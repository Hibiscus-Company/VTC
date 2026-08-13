#!/usr/bin/env python
"""Assert the hoisted algebra reproduces the PRODUCTION restore() bit-for-bit(ish), then time
one full-chain arm so the arm budget can be sized."""
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from div_lib import UT4, RD, ld, prep, combine, chain, load_field, stems
from energy_restore import restore
from lapfuse import _K
from fieldlib import warp

dev = "cuda" if torch.cuda.is_available() else "cpu"
K = _K.to(dev)
ss, gt_by = stems()
print("stems:", len(ss))
s = ss[0]
X = {m: ld(os.path.join(RD(m), s + ".png"), dev) for m in UT4}
P = {m: prep(X[m], K) for m in UT4}
L0 = {m: P[m][0] for m in UT4}
A = {m: P[m][1] for m in UT4}

for lam in (0.75, 1.0):
    mine = combine(X, L0, A, UT4, lam)
    ens = torch.stack([X[m] for m in UT4]).mean(0)
    prod = restore(ens, [X[m] for m in UT4], lam, 4, 3)
    d = (mine - prod).abs().max().item()
    print(f"lam={lam}  max|hoisted - production| = {d:.3e}   "
          f"(u8 quantum = {1/255:.3e})  {'OK' if d < 1e-4 else 'MISMATCH'}")

lens = load_field(1.30)
t0 = time.time()
for _ in range(3):
    j, nb = chain(combine(X, L0, A, UT4, 1.0), lens, warp)
print(f"chain (warp+jpeg) per arm-image: {(time.time()-t0)/3:.3f}s   jpeg {nb/1e6:.2f} MB")

from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
from PIL import Image
g = torch.from_numpy(np.asarray(Image.open(os.path.join("/mnt/d/avv/data/phase1/public_set/HCM0181/test/images", gt_by[s])).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
r = torch.from_numpy(j).permute(2, 0, 1).unsqueeze(0).to(dev)
torch.cuda.synchronize() if dev == "cuda" else None
t0 = time.time()
for _ in range(3):
    with torch.no_grad():
        a = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
        b = float(repo_ssim(r, g))
        c = float(vgg(r * 2 - 1, g * 2 - 1).item())
torch.cuda.synchronize() if dev == "cuda" else None
print(f"metrics per arm-image: {(time.time()-t0)/3:.3f}s   PSNR {a:.3f} SSIM {b:.4f} LPIPS {c:.4f}")

t0 = time.time()
_ = {m: ld(os.path.join(RD(m), ss[1] + ".png"), dev) for m in UT4}
print(f"png load per image: {(time.time()-t0)/4:.3f}s")
t0 = time.time()
_ = {m: prep(_[m], K) for m in UT4}
torch.cuda.synchronize() if dev == "cuda" else None
print(f"prep per image: {(time.time()-t0)/4:.3f}s")
