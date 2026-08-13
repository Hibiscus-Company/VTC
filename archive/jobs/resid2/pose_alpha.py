#!/usr/bin/env python
"""IS THE SCALE DEFICIT VIEW-DEPENDENT IN A WAY POSES PREDICT?

Established here: warping the TRAIN renders by the shipped field f1 and re-fitting leaves only
0.026 px (12% of |f1|, split-half -0.09 = noise).  So DIS is NOT shrinking -- f1 already
saturates on train pairs.  The alpha*=1.32 the TEST poses need is therefore not estimator bias
but TRAIN ABSORPTION: the Gaussians partly swallow the lens misregistration at the views they
were fit on, and a test view only gets that discount to the extent it sits near train views.

Prediction, if that story is right: the per-view scale alpha_i = <T_i,f1>/<f1,f1> should RISE
with a test view's distance from the train set, and fall to ~1 for a test view sitting on top
of one.  That would make alpha_i predictable from POSES ALONE -- no test GT -- which is
strictly more shippable than one hand-calibrated constant.

Sizes the ORACLE first (per-view alpha with each view's own GT).  If the oracle is small the
idea dies here for free.
"""
import os, sys, json
import numpy as np

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
OUT = os.path.join(HERE, "resid2")
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat                # noqa: E402

TAG = os.environ.get("TAG", "HCM0181")


def train_cams(tag):
    """TRAIN camera centres/axes only.

    train/sparse/0 holds 371 registered images -- all 240 train photos AND all 60 test poses
    (they were reconstructed together).  Leaving the test entries in makes every test view's
    distance-to-nearest-train-view exactly 0 and the covariate degenerate, so drop them by name.
    """
    p = f"/mnt/d/avv/data/phase1/public_set/{tag}/train/sparse/0/images.bin"
    trn = set(os.path.splitext(f)[0]
              for f in os.listdir(f"/mnt/d/avv/data/phase1/public_set/{tag}/train/images"))
    ex = read_extrinsics_binary(p)
    C, Z = [], []
    for k in ex:
        e = ex[k]
        if os.path.splitext(e.name)[0] not in trn:
            continue
        R = qvec2rotmat(e.qvec)
        t = np.asarray(e.tvec, dtype=np.float64)
        C.append(-R.T @ t)
        Z.append(R.T @ np.array([0.0, 0.0, 1.0]))     # optical axis in world
    return np.stack(C), np.stack(Z)


def test_cams(tag):
    p = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/test_poses.csv"
    rows = [l.strip().split(",") for l in open(p).read().strip().split("\n")[1:]]
    names, C, Z = [], [], []
    for r in rows:
        q = np.array([float(x) for x in r[1:5]])
        t = np.array([float(x) for x in r[5:8]])
        R = qvec2rotmat(q)
        names.append(os.path.splitext(r[0])[0])
        C.append(-R.T @ t)
        Z.append(R.T @ np.array([0.0, 0.0, 1.0]))
    return names, np.stack(C), np.stack(Z)


def main():
    Z = np.load(os.path.join(OUT, f"M_{TAG}.npz"))
    T = Z["T"].astype(np.float32)
    stems = list(Z["stems"])
    f1 = np.load(os.path.join(OUT, f"iter_{TAG}_gsplatB9ut.npz"))["f1"]

    ff = float((f1 * f1).sum())
    a_i = np.array([float((T[i] * f1).sum() / ff) for i in range(len(T))])
    a_glob = float((np.median(T, 0) * f1).sum() / ff)

    # ORACLE sizing: per-view alpha vs one global alpha, against each view's own flow
    e_glob = float(((T - a_glob * f1[None]) ** 2).sum())
    e_view = float(((T - a_i[:, None, None, None] * f1[None]) ** 2).sum())
    tot = float((T ** 2).sum())
    Mm = np.median(T, 0)
    consistent = float((Mm ** 2).sum() * len(T) / tot)

    # is the spread of alpha_i real structure or DIS noise?  fit alpha independently on the top
    # and the bottom half of the frame; if the per-view scale is a property of the VIEW the two
    # halves must agree across views.  (there is only one flow per view, so an image-domain
    # split is the only honest reliability check available.)
    h = f1.shape[0] // 2
    sp = []
    for sl in (slice(0, h), slice(h, None)):
        ffh = float((f1[sl] * f1[sl]).sum())
        sp.append(np.array([float((T[i][sl] * f1[sl]).sum() / ffh) for i in range(len(T))]))
    rel = float(np.corrcoef(sp[0], sp[1])[0, 1])

    out = {"tag": TAG, "n": len(T), "alpha_global": a_glob, "alpha_splithalf_rel": rel,
           "alpha_view_mean": float(a_i.mean()), "alpha_view_std": float(a_i.std()),
           "alpha_view_min": float(a_i.min()), "alpha_view_max": float(a_i.max()),
           "consistent_frac": consistent,
           "oracle_perview_gain_vs_global": float(1 - e_view / e_glob)}

    Ct, Zt = train_cams(TAG)
    ns, Cs, Zs = test_cams(TAG)
    idx = {n: i for i, n in enumerate(ns)}
    keep = [idx[s] for s in stems if s in idx]
    out["matched_poses"] = len(keep)
    if len(keep) == len(stems):
        Cs, Zs = Cs[keep], Zs[keep]
        d = np.linalg.norm(Cs[:, None, :] - Ct[None, :, :], axis=2)      # (Nt,Ntr)
        ang = np.degrees(np.arccos(np.clip(Zs @ Zt.T, -1, 1)))
        feats = {
            "d_nn": d.min(1),
            "d_k4": np.sort(d, 1)[:, :4].mean(1),
            "ang_nn": ang.min(1),
            "ang_k4": np.sort(ang, 1)[:, :4].mean(1),
            # combined pose distance: position gap scaled by scene extent + rotation gap
            "comb_nn": (d / max(d.mean(), 1e-9) + ang / max(ang.mean(), 1e-9)).min(1),
            "n_within_2deg": (ang < 2.0).sum(1).astype(float),
        }
        cors = {}
        for k, v in feats.items():
            cors[k] = (float(np.corrcoef(v, a_i)[0, 1]) if v.std() > 0 else float("nan"))
        out["pose_corr_with_alpha"] = cors
        # honest leave-one-view-out: best single feature, linear model
        best = max((k for k in cors if cors[k] == cors[k]), key=lambda k: abs(cors[k]))
        out["best_feat"] = best
        v = feats[best]
        pred = np.zeros_like(a_i)
        for i in range(len(a_i)):
            m = np.ones(len(a_i), bool); m[i] = False
            A = np.stack([np.ones(m.sum()), v[m]], 1)
            c, *_ = np.linalg.lstsq(A, a_i[m], rcond=None)
            pred[i] = c[0] + c[1] * v[i]
        e_pred = float((((T - pred[:, None, None, None] * f1[None]) ** 2)).sum())
        out["loo_posepred_gain_vs_global"] = float(1 - e_pred / e_glob)
        out["loo_pred_corr"] = float(np.corrcoef(pred, a_i)[0, 1])
        out["alpha_i"] = a_i.tolist()
        out["feat_best_vals"] = v.tolist()
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("alpha_i", "feat_best_vals")}, indent=1))
    json.dump(out, open(os.path.join(OUT, f"pose_{TAG}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
