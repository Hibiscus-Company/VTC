#!/usr/bin/env python
"""STEP 1 DIAGNOSTIC (no metric, just physics): per pyramid level, how much energy does averaging
destroy, and where does the mean sit relative to the GT?  This tells us whether the lever has any
room at all, and at WHICH levels.

  E_mem   = rms of a single member's Laplacian
  E_mean  = rms of the k-member mean's Laplacian
  shrink  = E_mean/E_mem   (=1 if members identical, =1/sqrt(k)=0.500 if fully independent)
  E_gt    = rms of the GT's Laplacian
  gap     = E_mean/E_gt    (<1 -> mean is too smooth at this level, >1 -> too busy)
  rho     = mean pairwise correlation between member Laplacians
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lap_run import Harness
from lapfuse import lap_pyr, _K

H = Harness(["m1", "m2", "m3", "m4"], nlev=5)
k = _K.to("cuda")
NL = 5
acc = np.zeros((NL, 5))
n = len(H.stems)
for i in range(n):
    st = H.stack(i)
    g = H.gt(i)
    lm, _, _ = lap_pyr(st, NL, k)
    lg, _, _ = lap_pyr(g, NL, k)
    for l in range(NL):
        L = lm[l]                       # [4,3,H,W]
        Em = L.pow(2).mean(dim=(1, 2, 3)).sqrt().mean().item()
        mn = L.mean(0, keepdim=True)
        Eb = mn.pow(2).mean().sqrt().item()
        Eg = lg[l].pow(2).mean().sqrt().item()
        # mean pairwise correlation (zero-mean coefficients by construction)
        f = L.reshape(L.shape[0], -1)
        f = f / (f.pow(2).mean(1, keepdim=True).sqrt() + 1e-12)
        C = (f @ f.T) / f.shape[1]
        kk = L.shape[0]
        rho = (C.sum() - C.diag().sum()).item() / (kk * (kk - 1))
        acc[l] += [Em, Eb, Eg, rho, 1]
    del st, g, lm, lg
acc[:, :4] /= acc[:, 4:5]

print(f"\n{'lvl':>3} {'~cyc/px':>8} {'E_mem':>8} {'E_mean':>8} {'shrink':>7} {'E_gt':>8} "
      f"{'gap':>7} {'rho':>6}")
for l in range(NL):
    Em, Eb, Eg, rho, _ = acc[l]
    print(f"{l:>3} {1.0/2**(l+2):>8.3f} {Em:8.5f} {Eb:8.5f} {Eb/Em:7.4f} {Eg:8.5f} "
          f"{Eb/Eg:7.4f} {rho:6.3f}")
print(f"\nfully-independent shrink floor for k=4 = {1/np.sqrt(4):.4f}")
