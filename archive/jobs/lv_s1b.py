#!/usr/bin/env python
"""S1b: two controls the first pass did not settle.

(1) FAIR STRUCTURE-ONLY CONTROL.  r = sqrt(E_members / E_mean) is ANTI-correlated with E_mean by
    construction (the denominator), so ordering the same histogram by +E_mean (the 'struct' run)
    inverts the map and is guaranteed to lose -- it is not a fair test.  The fair test orders the
    identical histogram by -E_mean: "give the biggest boost where the MEAN has the least local
    energy".  That predictor needs no ensemble at all.  If it works, the rule is a self-contained
    filter, we do not need member renders in production, and it would also apply to chair/bonsai.

(2) PRODUCTION-FEASIBLE MAP ESTIMATORS.  Production stores running ensemble MEANS, not all
    members; only some individual member renders survive.  Since
        mean_i E_i  =  E_mean + mean_i E(L_i - L_mean),
    the map can be rebuilt from the MEAN plus a SUBSET of members via the deviation energy:
        r_hat = sqrt(1 + (k/(k-1)) * mean_{i in S} E(L_i - L_mean) / E_mean)
    Measured at |S| = 1, 2 against the full-k map.
"""
import os, sys
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
import lv_main as LV
from lv_main import H, agg, dscore, tbl, HDR, DEV, UT, D, energy_map, hist_match
from lapfuse import lap_pyr, lap_recon, boxf, _K

K = _K.to(DEV)


def fuse2(st, lam, win=3, mode="raw", devsub=None, nlev=5):
    laps, res, sizes = lap_pyr(st, nlev, K)
    L0 = laps[0]
    mean0 = L0.mean(0, keepdim=True)
    Eb = boxf((mean0 ** 2).sum(1, keepdim=True), win)
    k = L0.shape[0]
    if devsub is None:
        Em = boxf((L0 ** 2).sum(1, keepdim=True), win).mean(0, keepdim=True)
        r = torch.sqrt((Em + 1e-10) / (Eb + 1e-10)).clamp(max=4.0)
    else:                                    # deviation estimator from a member subset
        dv = L0[devsub] - mean0
        V = boxf((dv ** 2).sum(1, keepdim=True), win).mean(0, keepdim=True) * (k / (k - 1.0))
        r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=4.0)
    if mode == "structinv":                  # same histogram, ordered by -E_mean (no ensemble)
        r = hist_match(r, -Eb)
    elif mode == "structpos":
        r = hist_match(r, Eb)
    out = [mean0 * (1.0 + lam * (r - 1.0))] + [l.mean(0, keepdim=True) for l in laps[1:]]
    return lap_recon(out, res.mean(0, keepdim=True), sizes, K), r, Eb


def run(h, lam, **kw):
    rows = []
    for i in range(len(h.stems)):
        st = h.stack(i)
        img = st.mean(0, keepdim=True) if lam == 0 else fuse2(st, lam, **kw)[0]
        a = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0)
        rows.append(h.eval_img(a, i))
    return np.array(rows)


if __name__ == "__main__":
    torch.manual_seed(0)
    h = H([D(t) for t in UT], names=UT)
    b = np.load(os.path.join(HERE, "lv_rows", "k4_png_base.npy"))
    print("\n" + "=" * 96)
    print("S1b. FAIR structure-only control + production-feasible map estimators (k=4, PNG)")
    print("=" * 96)
    print(HDR)
    tbl("pixel-mean BASELINE", b, b)
    tbl("energy lam1.0 REAL map", run(h, 1.0), b)
    tbl("  hist ordered by -E_mean", run(h, 1.0, mode="structinv"), b)
    tbl("  hist ordered by +E_mean", run(h, 1.0, mode="structpos"), b)
    print()
    tbl("dev-estimator, all 4 members", run(h, 1.0, devsub=[0, 1, 2, 3]), b)
    tbl("dev-estimator, members {1,2}", run(h, 1.0, devsub=[0, 1]), b)
    tbl("dev-estimator, member {1} only", run(h, 1.0, devsub=[0]), b)
    tbl("dev-estimator, member {3} only", run(h, 1.0, devsub=[2]), b)

    # ---- diagnostics: what does the map actually look like?
    st = h.stack(0)
    _, r, Eb = fuse2(st, 1.0)
    _, rd1, _ = fuse2(st, 1.0, devsub=[0])
    c = lambda a, bb: float(((a - a.mean()) * (bb - bb.mean())).mean() / (a.std() * bb.std()))
    lr, le = r.reshape(-1), torch.log(Eb.reshape(-1) + 1e-10)
    print(f"\n  corr( r , log E_mean )          = {c(lr, le):+.4f}   "
          f"(negative => the boost lands where the MEAN LOST energy, not on strong edges)")
    print(f"  corr( r_full , r_dev1member )   = {c(lr, rd1.reshape(-1)):+.4f}")
    print(f"  r: mean {float(r.mean()):.4f}  p50 {float(r.median()):.4f}  "
          f"p99 {float(torch.quantile(r.reshape(-1).float()[::37], 0.99)):.4f}  max {float(r.max()):.4f}")
