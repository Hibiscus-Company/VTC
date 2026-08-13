#!/usr/bin/env python
"""How much VIEW-CONSISTENT displacement structure is left after the shipped field, and can any
of it be reached WITHOUT test GT?

Everything here is a diagnostic; the oracle M is built from public test GT and is used only to
size prizes and to kill ideas.  Three questions:

  Q1 SCALE.   Does the train-only iterated field cum2 = f1 (+) f2 land on the oracle gain
              alpha* = <M,f1>/<f1,f1>?  If it does, the +0.14 scale win is shippable per-scene
              with no calibration constant.
  Q2 SHAPE.   After each field is given its OWN best scalar (so magnitude is factored out),
              does cum2 explain more of M than f1 does?  That is the only way a second-order
              fit can be worth anything beyond a number.
  Q3 RESIDUAL. R = M - alpha* f1.  Estimate R on half the test views, score it on the other
              half (VIEW hold-out).  A per-view field oracle already measured as a mirage under
              spatial hold-out; this asks the same question of the view-consistent part.
"""
import os, sys, json
import numpy as np
import cv2

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
OUT = os.path.join(HERE, "resid2")
cv2.setNumThreads(4)
TAG = os.environ.get("TAG", "HCM0181")


def r2(A, B):
    """best scalar a for B ~ a*A, plus R2 at that a and at a=1."""
    a = float((A * B).sum() / max((A * A).sum(), 1e-12))
    f = lambda g: 1.0 - ((B - g * A) ** 2).sum() / max((B ** 2).sum(), 1e-12)
    return a, float(f(a)), float(f(1.0))


def main():
    Z = np.load(os.path.join(OUT, f"M_{TAG}.npz"))
    T = Z["T"].astype(np.float32)
    M = np.median(T, 0)
    I = np.load(os.path.join(OUT, f"iter_{TAG}_gsplatB9ut.npz"))
    f1, f2, cum2 = I["f1"], I["f2"], I["cum2"]
    cum3 = I["cum3"]
    out = {"tag": TAG, "n_test": int(len(T)),
           "mean_M": float(np.linalg.norm(M, axis=2).mean()),
           "mean_f1": float(np.linalg.norm(f1, axis=2).mean()),
           "mean_cum2": float(np.linalg.norm(cum2, axis=2).mean())}

    # ---- Q1 / Q2
    for nm, F in (("f1", f1), ("cum2", cum2), ("cum3", cum3), ("f2", f2)):
        a, ra, r1 = r2(F, M)
        out[nm] = dict(alpha=a, R2_best=ra, R2_one=r1,
                       mag_ratio=float(np.linalg.norm(F, axis=2).mean() /
                                       np.linalg.norm(f1, axis=2).mean()))
    # does the train-only increment f2 point where the oracle deficit points?
    a1 = out["f1"]["alpha"]
    D = M - f1                                    # what the shipped field still owes
    out["corr_f2_deficit"] = float(np.corrcoef(f2.ravel(), D.ravel())[0, 1])
    out["proj_f2_on_deficit"] = float((f2 * D).sum() / max((D * D).sum(), 1e-12))

    # ---- Q3 residual after the best global scale, with VIEW hold-out
    R = M - a1 * f1
    out["resid_frac_of_M"] = float((R ** 2).sum() / (M ** 2).sum())
    rs = np.random.RandomState(0)
    hold = []
    for rep in range(8):
        idx = rs.permutation(len(T))
        A, B = idx[:len(T) // 2], idx[len(T) // 2:]
        MA, MB = np.median(T[A], 0), np.median(T[B], 0)
        aA = float((f1 * MA).sum() / (f1 * f1).sum())
        RA = MA - aA * f1                          # residual estimated on fold A
        aB = float((f1 * MB).sum() / (f1 * f1).sum())
        RB = MB - aB * f1
        # does adding RA to the fold-A-scaled field reduce the error against fold B's oracle?
        base = ((MB - aA * f1) ** 2).sum()
        with_r = ((MB - (aA * f1 + RA)) ** 2).sum()
        hold.append([float(1 - with_r / base), float(np.corrcoef(RA.ravel(), RB.ravel())[0, 1])])
        # smoothed residual (kill flow noise) at a few sigmas
    out["holdout_resid_gain"] = float(np.mean([h[0] for h in hold]))
    out["holdout_resid_corr"] = float(np.mean([h[1] for h in hold]))
    sm = {}
    for sig in (1, 2, 4, 8):
        g = []
        for rep in range(8):
            idx = rs.permutation(len(T))
            A, B = idx[:len(T) // 2], idx[len(T) // 2:]
            MA, MB = np.median(T[A], 0), np.median(T[B], 0)
            aA = float((f1 * MA).sum() / (f1 * f1).sum())
            RA = cv2.GaussianBlur(MA - aA * f1, (0, 0), sig)
            base = ((MB - aA * f1) ** 2).sum()
            g.append(float(1 - ((MB - (aA * f1 + RA)) ** 2).sum() / base))
        sm[f"sigma{sig}"] = float(np.mean(g))
    out["holdout_resid_gain_smoothed"] = sm

    # ---- radially-varying gain vs one global scalar (view hold-out too)
    h, w, _ = f1.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0)
    rho = rr / rr.max()
    nb = 8
    bins = np.clip((rho * nb).astype(int), 0, nb - 1)
    g = []
    for rep in range(8):
        idx = rs.permutation(len(T))
        A, B = idx[:len(T) // 2], idx[len(T) // 2:]
        MA, MB = np.median(T[A], 0), np.median(T[B], 0)
        aA = float((f1 * MA).sum() / (f1 * f1).sum())
        G = np.zeros_like(rho)
        for b in range(nb):
            m = bins == b
            G[m] = (f1[m] * MA[m]).sum() / max((f1[m] * f1[m]).sum(), 1e-12)
        Gs = cv2.GaussianBlur(G, (0, 0), 3)
        base = ((MB - aA * f1) ** 2).sum()
        g.append(float(1 - ((MB - Gs[..., None] * f1) ** 2).sum() / base))
    out["holdout_radial_gain"] = float(np.mean(g))

    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(OUT, f"diag_{TAG}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
