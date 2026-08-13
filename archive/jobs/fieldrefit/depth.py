#!/usr/bin/env python
"""Does a DEEPER fit pool (more members, more views) get the field any closer to the truth?
DIAGNOSTIC ONLY -- the oracle field is fit on test GT and is never shipped.

Ground truth = the field the k4 ensemble ACTUALLY needs at the test poses (DIS flow, real test GT,
median pooled at ds8). For each candidate train-fitted field f we report

    s*    = <f, f_oracle>/<f, f>          the LS-optimal rescale
    corr  = cos(f, f_oracle)              how much of the SHAPE is right
    resid = RMS|s* f - f_oracle| / RMS|f_oracle|    what is left after the best rescale

corr and resid are the parts a deeper pool could improve. s* is the part a scale constant fixes.
If resid is flat across pool depth, depth is worth nothing and the whole axis is closed.
"""
import os, sys, time, json
import numpy as np
import cv2
from PIL import Image

cv2.setNumThreads(2)
Image.MAX_IMAGE_PIXELS = None
HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
FR = os.path.join(HERE, "fieldrefit")
TAG = "HCM0181"
K4 = "/mnt/d/avv/prodharness/k4/png"
GTD = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"


def gray8(x):
    return (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)


def ld(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def oracle(ds=8):
    cp = os.path.join(FR, "oracle_k4_ds8.npy")
    if os.path.exists(cp):
        return np.load(cp)
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    st = []
    for f in sorted(os.listdir(K4)):
        s = os.path.splitext(f)[0]
        if not f.endswith(".png") or s not in gt_by:
            continue
        r, g = ld(os.path.join(K4, f)), ld(os.path.join(GTD, gt_by[s]))
        H, W, _ = g.shape
        fl = np.clip(dis.calc(gray8(g), gray8(r), None), -6.0, 6.0)
        st.append(cv2.resize(fl, (W // ds, H // ds), interpolation=cv2.INTER_AREA))
    o = np.median(np.stack(st), 0).astype(np.float32)
    np.save(cp, o)
    return o


def stats(f, o):
    num, den = float((f * o).sum()), float((f * f).sum())
    s = num / den
    c = num / np.sqrt(den * float((o * o).sum()))
    res = np.sqrt(((s * f - o) ** 2).sum() / (o ** 2).sum())
    return s, c, res, np.linalg.norm(f, axis=2).mean()


def main():
    t0 = time.time()
    o = oracle()
    print(f"oracle field (k4 ensemble at test poses vs real test GT): mean |f| "
          f"{np.linalg.norm(o, axis=2).mean():.4f} px   [{time.time()-t0:.0f}s]\n")
    med = lambda st: np.median(st.astype(np.float32), 0)
    Z = {t: np.load(os.path.join(FR, f"flow_{TAG}_{t}.npz")) for t in ("B9", "B11", "M2", "M2ER")}
    ZA = np.load(os.path.join(FR, f"flow_{TAG}_B11all.npz"))
    cand = {}
    for t in ("B9", "B11", "M2", "M2ER"):
        cand[f"{t} (60 views)"] = med(Z[t]["s8"])
    cand["avg(B9,B11) fields"] = 0.5 * (med(Z["B9"]["s8"]) + med(Z["B11"]["s8"]))
    rng = np.random.RandomState(0)
    for n in (15, 30, 60, 120):
        sel = np.arange(120) if n == 120 else rng.choice(120, n, replace=False)
        cand[f"B11 depth n={n:3d}"] = med(ZA["s8"][sel])
    for ds in (4, 16, 32):
        z = np.load(os.path.join(HERE, "lens", "cache", f"pub_{TAG}.npz"))
        if f"s{ds}" in z:
            f = med(z[f"s{ds}"])
            f = cv2.resize(f, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_CUBIC)
            cand[f"B9 ds={ds}"] = f

    print(f"{'candidate':>22} {'mean|f|':>8} {'s*':>7} {'corr':>7} {'resid':>7}")
    out = {}
    for k, f in cand.items():
        s, c, r, m = stats(f, o)
        out[k] = dict(s_star=s, corr=c, resid=r, mag=m)
        print(f"{k:>22} {m:8.4f} {s:7.3f} {c:7.3f} {r:7.3f}")
    json.dump(out, open(os.path.join(FR, "res_depth.json"), "w"), indent=1)
    print(f"\n{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
