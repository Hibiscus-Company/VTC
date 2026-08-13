#!/usr/bin/env python
"""STEP 6: CROSS-SCENE direction check.

HCM0181 is the ONLY scene on disk with >1 render variant at REAL test poses, so a true
production-regime cross-scene replicate does not exist. The next-best real data is the HCM0421
eval-split (a DIFFERENT tower, novel views, real held-out photos as GT) which has exactly two
members: the base UT render and the mip3d-filtered render -- the same two families production
blends. This is the DATA-STARVED proxy regime, so treat magnitude with suspicion and read only the
SIGN and the rough size against the k=2 production number (+0.0905).
"""
import os, sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lapfuse import fuse_image, _K
Image.MAX_IMAGE_PIXELS = None

GT = "/mnt/d/avv/evalsplit/HCM0421/eval_gt"
MEM = ["/mnt/d/avv/evalgen/HCM0421/eval_png", "/mnt/d/avv/mip3d/HCM0421_mip0.2/eval_png"]
ENERGY = dict(nalt=1, rule="energy", kw=dict(lam=1.0, win=5))

from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev = "cuda"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
gtf = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
stems = sorted(s for s in gtf if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
print(f"HCM0421 eval-split: {len(stems)} novel views x {len(MEM)} members (base UT + mip3d)")
k = _K.to(dev)


def sc(fuse):
    R = []
    for s in stems:
        st = torch.stack([torch.from_numpy(
            np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"), dtype=np.float32) / 255.
        ).permute(2, 0, 1) for d in MEM]).to(dev)
        img = fuse_image(st, ENERGY, 5, k) if fuse else st.mean(0, keepdim=True)
        a = (img.clamp(0, 1) * 255).round().to(torch.uint8).float() / 255.
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(GT, gtf[s])).convert("RGB"),
                                        dtype=np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            R.append((10 * np.log10(1. / max(((a - g) ** 2).mean().item(), 1e-12)),
                      float(repo_ssim(a, g)), float(vgg(a * 2 - 1, g * 2 - 1).item())))
    R = np.array(R)
    P, S, L = R.mean(0)
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1.)), P, S, L, R


b = sc(False); e = sc(True)
print(f"{'pixel-mean k=2':<20} {b[0]:9.4f}  PSNR {b[1]:.4f} SSIM {b[2]:.4f} LPIPS {b[3]:.4f}")
print(f"{'LAPFUSE energy':<20} {e[0]:9.4f}  PSNR {e[1]:.4f} SSIM {e[2]:.4f} LPIPS {e[3]:.4f}")
sv = lambda R: 100 * (0.4 * (1 - R[:, 2]) + 0.3 * R[:, 1] + 0.3 * R[:, 0] / 50)
d = sv(e[4]) - sv(b[4])
print(f"delta = {e[0]-b[0]:+.4f}   (per-view: {(d>0).sum()}/{len(d)} positive, "
      f"t={d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):.2f})")
print(f"production-regime HCM0181 k=2 for comparison: +0.0905")
