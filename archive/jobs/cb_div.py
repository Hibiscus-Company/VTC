#!/usr/bin/env python
"""Is there a GT-FREE weighting rule (usable on private_set2, where no test GT exists)?

Identity, exact:  E||sum_i w_i e_i||^2 = sum_i w_i C_ii - (1/2) w' G w,
with G_ij = E||x_i - x_j||^2 the GT-FREE pairwise disagreement matrix.
=> if the members are EQUAL QUALITY (C_ii = const), minimising the ensemble error is
   exactly MAXIMISING w'Gw on the simplex, whose stationary point is w ~ G^-1 1.
   That is a pure DIVERSITY weighting and needs no ground truth at all.
Also evaluated: inverse-LOO-disagreement weights, and the honest split-half transfer of
each.  Everything here is a closed-form read of the covariance matrices already measured.
"""
import json, sys
import numpy as np

for pool in ("A", "B"):
    d = json.load(open(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/cb_cov_{pool}.json"))
    C = np.array(d["C"]); k = C.shape[0]; NB = d["nbig"]; one = np.ones(k)
    G = np.diag(C)[:, None] + np.diag(C)[None, :] - 2 * C      # GT-free disagreement
    w_flat = one / k

    def mse(w):
        return float(w @ C @ w)

    def norm(v):
        return v / v.sum()

    # 1. diversity weights  w ~ G^-1 1
    w_div = norm(np.linalg.solve(G + 1e-12 * np.eye(k), one))
    # 2. inverse LOO-disagreement:  D_i = ||x_i - mean_{j!=i} x_j||^2  (GT-free)
    D = np.array([C[i, i] - 2 * (C[i].sum() - C[i, i]) / (k - 1)
                  + (C.sum() - 2 * C[i].sum() + C[i, i]) / (k - 1) ** 2 for i in range(k)])
    w_inv = norm(1.0 / D)
    # 3. ORACLE GLS (needs GT)
    w_gls = norm(np.linalg.solve(C, one))
    m0 = mse(w_flat)
    print(f"\n=== pool {pool}  k={k}   member error PSNR spread "
          f"{10*np.log10(1/np.diag(C)).max()-10*np.log10(1/np.diag(C)).min():.3f} dB")
    for nm, w in (("flat", w_flat), ("diversity G^-1 1", w_div),
                  ("inv-LOO-disagree", w_inv), ("ORACLE GLS (uses GT)", w_gls)):
        dp = 10 * np.log10(m0 / mse(w))
        print(f"  {nm:<22} dPSNR {dp:+.4f} dB -> dScore(PSNR term) {0.6*dp:+.4f}   "
              f"w_small={w[NB:].sum():+.4f}  min_w={w.min():+.4f}")
    print(f"  diversity weights: " + " ".join(f"{v:+.3f}" for v in w_div))
