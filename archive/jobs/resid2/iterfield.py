#!/usr/bin/env python
"""SECOND-ORDER FIELD FIT ON THE RESIDUAL AFTER THE FIRST FIELD -- train-only.

The shipped field f1 is a shrunk copy of what the test poses need: the oracle projection of
the test-pose consistent field M onto f1 is alpha*=1.318 (R2 0.901 -> 0.957).  Scaling by a
hand-picked 1.30 is worth +0.140 on the production harness (fieldrefit/res_combo.json), but
that constant was READ OFF the public test GT.  Two competing explanations, and they make
opposite predictions for the private set:

  H1 ESTIMATOR SHRINKAGE.  DIS regularises; it returns a fraction 1/alpha of the true
     displacement everywhere, on train pairs exactly as on test pairs.  Then re-fitting the
     flow on renders ALREADY WARPED by f1 recovers the missing part with NO test GT at all,
     per-scene, spatially adaptively -- and predicts |f2|/|f1| ~ 1 - 1/alpha ~ 0.24.
  H2 TRAIN ABSORPTION.  The Gaussians partially fit the misregistration at the TRAIN views,
     so the train-pose render/photo offset is genuinely smaller than the test-pose one.  Then
     the train residual after f1 is ~0 and iteration recovers NOTHING; the 1.30 is a
     train->test extrapolation constant that must be calibrated on GT.

This script measures |f2| and |f3| from TRAIN photos only.  It never opens test GT.
"""
import os, sys, time, json
import numpy as np
import cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
OUT = os.path.join(HERE, "resid2")
sys.path.insert(0, os.path.join(HERE, "lens"))
from fieldlib import upsample, warp                                    # noqa: E402

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(int(os.environ.get("NT", "8")))
TAG = os.environ.get("TAG", "HCM0181")
MEMBER = os.environ.get("MEMBER", "gsplatB9ut")
GT = f"/mnt/d/avv/data/phase1/public_set/{TAG}/train/images"
RD = f"/mnt/d/avv/output/{TAG}_{MEMBER}/train_renders"
DS = 8


def ld(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def gray8(x):
    return (cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)


def fit_round(stems, gt_by, field_full, H, W, dis):
    """median DIS flow(train photo -> render warped by field_full), pooled at 1/DS."""
    st = []
    for s in stems:
        r = ld(os.path.join(RD, s + ".png"))
        if field_full is not None:
            r = np.clip(warp(r, field_full, "lanczos"), 0, 1)
        g = ld(os.path.join(GT, gt_by[s]))
        fl = np.clip(dis.calc(gray8(g), gray8(r), None), -6.0, 6.0)
        st.append(cv2.resize(fl, (W // DS, H // DS), interpolation=cv2.INTER_AREA))
    return np.stack(st).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
    stems = sorted(s for s in (os.path.splitext(f)[0] for f in os.listdir(RD)
                               if f.lower().endswith(".png")) if s in gt_by)
    p0 = ld(os.path.join(RD, stems[0] + ".png"))
    H, W, _ = p0.shape
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    t0 = time.time()

    # round 1: reproduce the shipped field exactly from the cache (no recompute)
    z = np.load(os.path.join(HERE, "fieldrefit", f"flow_{TAG}_B9.npz")) \
        if TAG == "HCM0181" and MEMBER == "gsplatB9ut" else None
    if z is not None and sorted(list(z["stems"])) == stems:
        S1 = z["s8"].astype(np.float32)
    else:
        S1 = fit_round(stems, gt_by, None, H, W, dis)
    f1 = np.median(S1, 0)
    print(f"[{TAG}] n={len(stems)} {W}x{H}  |f1| {np.linalg.norm(f1,axis=2).mean():.4f} px "
          f"({time.time()-t0:.0f}s)", flush=True)

    res = {"tag": TAG, "member": MEMBER, "n": len(stems),
           "mean_f1": float(np.linalg.norm(f1, axis=2).mean())}
    fields = {"f1": f1}
    cum = f1.copy()
    for k in (2, 3):
        F = upsample(cum, H, W, "cubic")
        Sk = fit_round(stems, gt_by, F, H, W, dis)
        fk = np.median(Sk, 0)
        # projection of the new increment onto the CURRENT cumulative field: if this is pure
        # estimator shrinkage the increment is parallel to what is already there.
        pr = float((fk * cum).sum() / max((cum * cum).sum(), 1e-12))
        co = float(np.corrcoef(fk.ravel(), cum.ravel())[0, 1])
        cum = cum + fk
        fields[f"f{k}"] = fk
        fields[f"cum{k}"] = cum.copy()
        res[f"mean_f{k}"] = float(np.linalg.norm(fk, axis=2).mean())
        res[f"ratio_f{k}_f1"] = res[f"mean_f{k}"] / res["mean_f1"]
        res[f"proj_f{k}_on_cum"] = pr
        res[f"corr_f{k}_cum"] = co
        res[f"gain_cum{k}"] = float(np.linalg.norm(cum, axis=2).mean() / res["mean_f1"])
        # split-half stability of the increment: is f_k structure or flow noise?
        idx = np.random.RandomState(0).permutation(len(Sk))
        a = np.median(Sk[idx[:len(Sk) // 2]], 0); b = np.median(Sk[idx[len(Sk) // 2:]], 0)
        res[f"splithalf_f{k}"] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        print(f"  round {k}: |f{k}| {res[f'mean_f{k}']:.4f} px  "
              f"ratio {res[f'ratio_f{k}_f1']:.3f}  proj-on-cum {pr:+.4f}  corr {co:+.3f}  "
              f"splithalf {res[f'splithalf_f{k}']:.3f}  cumgain {res[f'gain_cum{k}']:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(os.path.join(OUT, f"iter_{TAG}_{MEMBER}.npz"), **fields)
    json.dump(res, open(os.path.join(OUT, f"iter_{TAG}_{MEMBER}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
