#!/usr/bin/env python
"""Does the 28-hole bonsai pool DESTROY the LPIPS gain that lpips_from produces?
Score solo members vs their pixel mean, same 28 holes, no encode."""
import os, sys, numpy as np
from PIL import Image
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg

GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
A = "/mnt/d/avv/bonsai_eval/K4_pC_seed7/eval_png"
B = "/mnt/d/avv/bonsai_eval/K4_pC_seed1k/eval_png"

dev = "cuda"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
stems = sorted(gt_by)

def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0)

acc = {k: [0.0,0.0,0.0] for k in ("A","B","MEAN")}
with torch.no_grad():
    for s in stems:
        g = load(os.path.join(GT, gt_by[s])).to(dev)
        a = load(os.path.join(A, s+".png")).to(dev)
        b = load(os.path.join(B, s+".png")).to(dev)
        m = (a+b)/2.0
        for k, r in (("A",a),("B",b),("MEAN",m)):
            mse = ((r-g)**2).mean().item()
            acc[k][0] += 10*np.log10(1.0/max(mse,1e-12))
            acc[k][1] += float(repo_ssim(r,g))
            acc[k][2] += float(vgg(r*2-1, g*2-1).item())
n = len(stems)
for k,(P,S,L) in acc.items():
    P,S,L = P/n, S/n, L/n
    sc = 100*(0.4*(1-L) + 0.3*S + 0.3*min(P/50,1))
    print(f"{k:5s} n={n} PSNR {P:.4f}  SSIM {S:.4f}  LPIPS {L:.4f}  SCORE {sc:.4f}")
