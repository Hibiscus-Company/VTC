#!/usr/bin/env python
"""IS THE TRAIN->TEST FIELD DEFICIT A LOW-ORDER LENS TERM RATHER THAN A PURE SCALE?

Story so far: warping the train renders by f1 and re-fitting leaves only noise, so f1 saturates
on TRAIN pairs; yet the test poses want alpha ~1.32-1.43 times f1.  The deficit is therefore the
misregistration the Gaussians ABSORBED at the views they were fit on.  A model can only absorb
what it can represent by moving Gaussians -- i.e. smooth, low-order warps -- which predicts the
deficit D = M - f1 is MORE low-order/radial than f1 itself, and that a Brown-Conrady radial
model should beat a scalar.

Three models per scene, each fit on a random half of the TEST VIEWS and scored on the held-out
half (so a model that only fits flow noise scores <= 0):
    scale     M ~ a*f1                              1 parameter   <- what the candidate ships
    scale+rad M ~ a*f1 + radial Brown basis         1 + 3         <- the hypothesis
    free      M ~ a*f1 + arbitrary dense residual   40590         <- the ceiling, known mirage
Also reports whether the extra coefficients agree ACROSS the five public towers, since only a
scene-independent correction is shippable to the private set.
"""
import os, sys, json
import numpy as np

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
OUT = os.path.join(HERE, "resid2")
SCENES = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]


def basis(h, w):
    """radial lens terms u*r^2, u*r^4, u*r^6 (and the v components), normalised coords."""
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    u = (u + 0.5) / w * 2 - 1
    v = (v + 0.5) / h * 2 - 1
    r2 = u * u + v * v
    B = []
    for p in (1, 2, 3):
        B.append(np.stack([u * r2 ** p, v * r2 ** p], -1))
    return np.stack(B)                        # (3,h,w,2)


def fit(cols, y):
    """least squares over flattened (h,w,2) design columns."""
    A = np.stack([c.ravel() for c in cols], 1)
    c, *_ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    return c, (A @ c).reshape(y.shape)


def main():
    rs = np.random.RandomState(0)
    allc, rows = {}, {}
    for tag in SCENES:
        mp = os.path.join(OUT, f"M_{tag}.npz")
        fp = os.path.join(HERE, "flow", f"{tag}.npz")
        if not os.path.exists(mp):
            print(f"skip {tag} (no M cache)"); continue
        T = np.load(mp)["T"].astype(np.float32)
        f1 = np.median(np.load(fp)["ds8"].astype(np.float32), 0)
        h, w, _ = f1.shape
        B = basis(h, w)
        g = {"scale": [], "rad": [], "free": []}
        coefs = []
        for rep in range(12):
            idx = rs.permutation(len(T))
            A, Bh = idx[:len(T) // 2], idx[len(T) // 2:]
            MA, MB = np.median(T[A], 0), np.median(T[Bh], 0)
            nrm = float((MB ** 2).sum())
            ca, pa = fit([f1], MA)
            g["scale"].append(1 - ((MB - pa) ** 2).sum() / nrm)
            cr, pr = fit([f1] + list(B), MA)
            g["rad"].append(1 - ((MB - pr) ** 2).sum() / nrm)
            g["free"].append(1 - ((MB - MA) ** 2).sum() / nrm)
            coefs.append(cr)
        C = np.stack(coefs)
        allc[tag] = C.mean(0).tolist()
        rows[tag] = {k: float(np.mean(v)) for k, v in g.items()}
        rows[tag]["rad_minus_scale"] = rows[tag]["rad"] - rows[tag]["scale"]
        rows[tag]["coef"] = C.mean(0).tolist()
        rows[tag]["coef_sd_over_folds"] = C.std(0).tolist()

    print(f"{'scene':>9} {'R2 scale':>9} {'R2 +radial':>11} {'delta':>8} {'R2 free':>9}   "
          f"{'a':>6} {'k1':>8} {'k2':>8} {'k3':>8}")
    for t, r in rows.items():
        c = r["coef"]
        print(f"{t:>9} {r['scale']:9.4f} {r['rad']:11.4f} {r['rad_minus_scale']:+8.4f} "
              f"{r['free']:9.4f}   {c[0]:6.3f} {c[1]:+8.4f} {c[2]:+8.4f} {c[3]:+8.4f}")
    if len(allc) > 1:
        C = np.array(list(allc.values()))
        print("\ncross-scene coefficient spread (mean +/- sd over scenes):")
        for i, n in enumerate(["a", "k1", "k2", "k3"]):
            print(f"  {n:>3}  {C[:,i].mean():+8.4f} +/- {C[:,i].std():.4f}   "
                  f"(|sd/mean| {abs(C[:,i].std()/(C[:,i].mean()+1e-12)):.2f})")
    json.dump(rows, open(os.path.join(OUT, "res_radial.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
