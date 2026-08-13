#!/usr/bin/env python
"""Score a bonsai eval-render dir, reporting first8 / last20 SEPARATELY.

A3 (30/07) found the 28 holes split cleanly into two populations: the first 8 holes
(frames 10-710, a fast far-field sweep with 1.5 train cams within 0.5 units) average
62.00, the other 20 average 75.99. A scene-averaged number hides a 14-point gap, so
every bonsai arm from here on reports both.
"""
import argparse, os, re, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
FIRST8 = {10, 120, 190, 260, 440, 510, 630, 710}
def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0)
def sc(P, S, L):
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1))
ap = argparse.ArgumentParser()
ap.add_argument("--render_dir", required=True); ap.add_argument("--gt_dir", required=True)
ap.add_argument("--tag", default=""); ap.add_argument("--ref", type=float, default=71.9911)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
gt = {os.path.splitext(f)[0]: f for f in os.listdir(a.gt_dir)}
ren = [f for f in sorted(os.listdir(a.render_dir)) if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")]
assert {os.path.splitext(f)[0] for f in ren} == set(gt), "render/GT stem sets differ"
per = {}
with torch.no_grad():
    for f in ren:
        s = os.path.splitext(f)[0]
        r = load(os.path.join(a.render_dir, f)).to(dev); g = load(os.path.join(a.gt_dir, gt[s])).to(dev)
        assert r.shape == g.shape, f"shape mismatch {f}"
        mse = ((r - g) ** 2).mean().item()
        per[int(re.findall(r"\d+", s)[0])] = (10 * np.log10(1 / max(mse, 1e-12)),
                                              float(repo_ssim(r, g)), float(vgg(r * 2 - 1, g * 2 - 1).item()))
def agg(keys):
    if not keys: return None
    M = np.array([per[k] for k in keys])
    return sc(*M.mean(0)), M.mean(0)
allk = sorted(per); f8 = [k for k in allk if k in FIRST8]; l20 = [k for k in allk if k not in FIRST8]
S, m = agg(allk); S8, m8 = agg(f8); S20, m20 = agg(l20)
print(f"EVAL {a.tag:<18} n={len(allk):3d}  PSNR {m[0]:7.4f} SSIM {m[1]:.4f} LPIPS {m[2]:.4f}  "
      f"SCORE {S:.4f}   d_vs_ref {S - a.ref:+.4f}  [ref {a.ref}]", flush=True)
print(f"     first8  n={len(f8):2d}  PSNR {m8[0]:7.4f} SSIM {m8[1]:.4f} LPIPS {m8[2]:.4f}  SCORE {S8:.4f}", flush=True)
print(f"     last20  n={len(l20):2d}  PSNR {m20[0]:7.4f} SSIM {m20[1]:.4f} LPIPS {m20[2]:.4f}  SCORE {S20:.4f}", flush=True)
print("PERFRAME " + " ".join(f"{k}:{sc(*per[k]):.2f}" for k in allk), flush=True)
