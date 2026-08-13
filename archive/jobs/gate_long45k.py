#!/usr/bin/env python
"""Diversity-matched, ENCODED k=2 paired gate -- the only surface calibrated against the LB (0.84x).

Both sides get exactly 2 members at matched seeds (42, 101), are averaged with the shipped
ensemble rule (float32 accumulate, single *255+0.5 round), pushed through the exact ship JPEG
encode, and scored per-frame so the comparison is PAIRED. A raw mixsweep overpredicts ~2.2x and
is not admissible here.
"""
import os, sys, io, numpy as np, torch
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from PIL import Image
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(6)
GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
SIDES = {
    "ref_churn15k8k": ["/mnt/d/avv/r36_shape/sr01/eval_png", "/mnt/d/avv/r38/sr01_s101/eval_png"],
    "long45k":        ["/mnt/d/avv/r43_bonsai/long45k/eval_png",
                       "/mnt/d/avv/r43_bonsai/long45k_s101/eval_png"],
}
ENC = dict(quality=100, subsampling=2, optimize=True, progressive=True)
FIRST8 = {10, 120, 190, 260, 440, 510, 630, 710}
gt = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
stems = sorted(gt)
vgg = lpips_pkg.LPIPS(net="vgg").eval()
def sc(P, S, L): return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1))
def mean_encode(dirs, stem):
    acc = None
    for d in dirs:
        a = np.asarray(Image.open(f"{d}/{stem}.png").convert("RGB"), dtype=np.float32)
        acc = a if acc is None else acc + a
    m = np.clip(acc / len(dirs) + 0.5, 0, 255).astype(np.uint8)   # single round, shipped rule
    b = io.BytesIO(); Image.fromarray(m).save(b, "JPEG", **ENC)   # exact ship encode
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.float32) / 255.
per = {k: {} for k in SIDES}
with torch.no_grad():
    for s in stems:
        g = torch.from_numpy(np.asarray(Image.open(f"{GT}/{gt[s]}").convert("RGB"),
                                        dtype=np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)
        for name, dirs in SIDES.items():
            r = torch.from_numpy(mean_encode(dirs, s)).permute(2, 0, 1).unsqueeze(0)
            mse = ((r - g) ** 2).mean().item()
            per[name][s] = (10 * np.log10(1 / max(mse, 1e-12)), float(repo_ssim(r, g)),
                            float(vgg(r * 2 - 1, g * 2 - 1).item()))
        print(f"  {s} done", flush=True)
def agg(name, keys):
    M = np.array([per[name][k] for k in keys]); return sc(*M.mean(0)), M.mean(0)
import re
f8 = [s for s in stems if int(re.findall(r"\d+", s)[0]) in FIRST8]
l20 = [s for s in stems if s not in f8]
print()
for name in SIDES:
    S, m = agg(name, stems); S8, _ = agg(name, f8); S20, _ = agg(name, l20)
    print(f"{name:16s} k=2 encoded  PSNR {m[0]:7.4f} SSIM {m[1]:.4f} LPIPS {m[2]:.4f}  "
          f"SCORE {S:.4f}   first8 {S8:.4f}  last20 {S20:.4f}", flush=True)
a, b = list(SIDES)
d = np.array([sc(*per[b][s]) - sc(*per[a][s]) for s in stems])
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
print(f"\nPAIRED delta ({b} - {a}) over {len(d)} holes:")
print(f"  mean {d.mean():+.4f}   sd {d.std(ddof=1):.4f}   t = {t:+.3f}   wins {int((d>0).sum())}/{len(d)}")
print(f"  forecast LB scene gain = {d.mean():+.4f} x 0.85 = {d.mean()*0.85:+.4f}")
print(f"  forecast LB TOTAL gain = {d.mean()*0.85/7:+.5f}   (r36 was +0.0124, r35 +0.0199)")
