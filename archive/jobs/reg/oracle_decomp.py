#!/usr/bin/env python
"""DIAGNOSTIC (no shipping decision reads this): how much of the VIEW-CONSISTENT displacement
field at REAL TEST poses is explained by (a) the shipped train-fitted field, (b) that field with
one scalar gain, (c) that field with a radially-varying gain?

The shipped field f is the median DIS flow over TRAIN pairs.  Define the test-pose oracle
M = median over test views of the DIS flow (test photo -> production ensemble render).  M is the
view-CONSISTENT part of the residual at test poses -- exactly the thing a fixed field can chase.
M is measured against public test GT for DIAGNOSIS ONLY; nothing fitted here is ever shipped
(Rule 10: the shipped field stays the train-fitted one, optionally times a scalar chosen on the
public calibration set).

Reports:
  alpha*  = <M,f>/<f,f>              the amplitude the train fit misses
  R2(1.0) / R2(alpha*)               fraction of the consistent test field f explains
  radial / tangential alphas         is the deficit isotropic?
  alpha(rho) in radial bins          does the deficit have spatial shape?
  split-half stability of M          is M itself real, or flow noise?
"""
import os, sys, json
import numpy as np
import cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(4)

MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
DS = 8


def gray8(a):
    return (cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)


def main():
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    stack = []
    for i, s in enumerate(stems):
        r = np.mean([np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                                dtype=np.float32) / 255.0 for d in MEM], axis=0)
        g = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"),
                       dtype=np.float32) / 255.0
        H, W, _ = r.shape
        fl = np.clip(dis.calc(gray8(g), gray8(r), None), -6.0, 6.0)     # photo -> render
        stack.append(cv2.resize(fl, (W // DS, H // DS), interpolation=cv2.INTER_AREA))
        if i % 15 == 0:
            print(f"  flow {i}/{len(stems)}", flush=True)
    T = np.stack(stack).astype(np.float32)                              # (Nt,h,w,2)
    M = np.median(T, axis=0)                                            # test-pose consistent field
    f = np.median(np.load(f"{HERE}/flow/HCM0181.npz")["ds8"].astype(np.float32), axis=0)

    h, w, _ = f.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - (w - 1) / 2.0, yy - (h - 1) / 2.0
    rr = np.hypot(dx, dy); ux, uy = dx / (rr + 1e-9), dy / (rr + 1e-9)
    rho = rr / rr.max()

    def comp(F):
        p = F[..., 0] * ux + F[..., 1] * uy
        rad = np.stack([p * ux, p * uy], -1)
        return rad, F - rad

    def alpha_r2(A, B, m=None):
        """best scalar a minimising |B - a A|^2, and R2 of a A explaining B."""
        if m is None:
            m = np.ones(A.shape[:2], bool)

        a = float((A[m] * B[m]).sum() / max((A[m] * A[m]).sum(), 1e-12))
        r2 = lambda g: 1.0 - ((B[m] - g * A[m]) ** 2).sum() / max((B[m] ** 2).sum(), 1e-12)
        return a, r2(a), r2(1.0)

    out = {}
    out["mean_f"] = float(np.linalg.norm(f, axis=2).mean())
    out["mean_M"] = float(np.linalg.norm(M, axis=2).mean())
    out["mean_perview_test"] = float(np.linalg.norm(T, axis=3).mean())
    out["consistent_frac"] = float((M ** 2).sum() / (T ** 2).sum() * len(T))
    a, r2a, r21 = alpha_r2(f, M)
    out["alpha_star"], out["R2_alpha"], out["R2_one"] = a, r2a, r21
    fr, ft = comp(f); Mr, Mt = comp(M)
    out["alpha_radial"] = alpha_r2(fr, Mr)[0]
    out["alpha_tangential"] = alpha_r2(ft, Mt)[0]
    out["f_radial_energy_frac"] = float((fr ** 2).sum() / (f ** 2).sum())
    out["M_radial_energy_frac"] = float((Mr ** 2).sum() / (M ** 2).sum())
    # radial profile of the deficit
    prof = []
    edges = np.linspace(0, 1, 7)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rho >= lo) & (rho < hi + (1e-9 if hi == 1 else 0))
        aa, rr2, _ = alpha_r2(f, M, m)
        prof.append(dict(rho=[float(lo), float(hi)], alpha=float(aa), R2=float(rr2),
                         npx=int(m.sum()), mean_f=float(np.linalg.norm(f[m], axis=1).mean())))
    out["radial_profile"] = prof
    # split-half stability of the oracle M (is it real structure or flow noise?)
    idx = np.random.RandomState(0).permutation(len(T))
    M1 = np.median(T[idx[:len(T) // 2]], 0); M2 = np.median(T[idx[len(T) // 2:]], 0)
    out["M_splithalf_corr"] = float(np.corrcoef(M1.ravel(), M2.ravel())[0, 1])
    out["M_splithalf_alpha"] = float((M1 * M2).sum() / (M1 * M1).sum())
    # residual after the best scalar-gain field: how much consistent structure is LEFT?
    R = M - a * f
    out["resid_energy_frac_of_M"] = float((R ** 2).sum() / (M ** 2).sum())
    R1 = M1 - a * f; R2_ = M2 - a * f
    out["resid_splithalf_corr"] = float(np.corrcoef(R1.ravel(), R2_.ravel())[0, 1])
    out["n_test"] = len(T)

    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{HERE}/reg/oracle_decomp.json", "w"), indent=1)


if __name__ == "__main__":
    main()
