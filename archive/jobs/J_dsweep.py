"""The largest error mode is LOCAL sub-pixel misregistration (oracle dense flow = +1.55).
Our shipped lens field pools DIS flow at ds=8 (165x123). Is it simply too coarse?
Fit on TRAIN photos only (zero leakage), score on REAL test GT, 5 scenes."""
import os, sys
import numpy as np, cv2, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from common import *
cv2.setNumThreads(6)
dev = "cuda"; lp = mlib.LP(dev)
SCENES = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
DSS = [16, 8, 4, 2]
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
rows = {}
for sc_ in SCENES:
    trd = f"/mnt/d/avv/output/{sc_}_gsplatB9ut/train_renders"
    tgd = f"/mnt/d/avv/data/phase1/public_set/{sc_}/train/images"
    tgm = {os.path.splitext(f)[0]: f for f in os.listdir(tgd)}
    fls = []
    for f in sorted(os.listdir(trd)):
        stem = os.path.splitext(f)[0]
        if stem not in tgm: continue
        r = cv2.cvtColor(mlib.load_u8(os.path.join(trd, f)), cv2.COLOR_RGB2GRAY)
        g = cv2.cvtColor(mlib.load_u8(os.path.join(tgd, tgm[stem])), cv2.COLOR_RGB2GRAY)
        fls.append(np.clip(dis.calc(g, r, None), -6, 6))
    FIELDS = {}
    for ds in DSS:
        S = np.stack([cv2.resize(f, (f.shape[1] // ds, f.shape[0] // ds),
                                 interpolation=cv2.INTER_AREA) for f in fls])
        FIELDS[("med", ds)] = np.median(S, 0)
        if ds == 8: FIELDS[("mean", ds)] = S.mean(0)
    shipped = np.load(f"/mnt/d/avv/fields/{sc_}.npy")
    m = np.sqrt((FIELDS[("mean", 8)] ** 2).sum(-1)).mean()
    print(f"{sc_}: {len(fls)} train pairs; refit mean-ds8 rms {m:.3f}px vs shipped "
          f"{np.sqrt((shipped**2).sum(-1)).mean():.3f}px", flush=True)

    ted = f"/mnt/d/avv/output/{sc_}_gsplatB9ut/test_poses_renders_png"
    gdir, gmp = gt_map(sc_)
    ss = [os.path.splitext(f)[0] for f in sorted(os.listdir(ted))]
    VAR = ["nofield", "shipped(mean,ds8)"] + [f"med_ds{d}" for d in DSS] + ["refit_mean_ds8"]
    A = {v: [0.0, 0.0, 0.0] for v in VAR}
    for stem in ss:
        R8 = mlib.load_u8(os.path.join(ted, stem + ".png"))
        G8 = mlib.load_u8(os.path.join(gdir, gmp[stem])); tg = mlib.to_t(G8, dev)
        outs = {"nofield": R8, "shipped(mean,ds8)": apply_field(R8, shipped),
                "refit_mean_ds8": apply_field(R8, FIELDS[("mean", 8)])}
        for d in DSS:
            outs[f"med_ds{d}"] = apply_field(R8, FIELDS[("med", d)])
        for v, a in outs.items():
            y = mlib.to_t(a, dev)
            A[v][0] += mlib.psnr(y, tg); A[v][1] += float(mlib.ssim(y, tg)); A[v][2] += lp(y, tg)
    n = len(ss); rows[sc_] = {}
    ref = None
    for v in VAR:
        P, S, Lv = [x / n for x in A[v]]
        s2 = mlib.score(P, S, Lv)
        if v == "shipped(mean,ds8)": ref = s2
        rows[sc_][v] = (P, S, Lv, s2)
    for v in VAR: rows[sc_][v] = rows[sc_][v] + (rows[sc_][v][3] - ref,)
    print("   " + "  ".join(f"{v}:{rows[sc_][v][4]:+.4f}" for v in VAR), flush=True)

print("\n=== LENS-FIELD RESOLUTION SWEEP (train-fitted, zero leakage) ===")
print("    delta score vs shipped mean-pooled ds=8 field")
VAR = list(rows[SCENES[0]].keys())
print(f"  {'variant':20s} " + " ".join(f"{s:>9s}" for s in SCENES) + f" {'MEAN':>9s}")
for v in VAR:
    ds_ = [rows[s][v][4] for s in SCENES]
    print(f"  {v:20s} " + " ".join(f"{d:+9.4f}" for d in ds_) + f" {np.mean(ds_):+9.4f}")
print("\n  absolute scores (shipped): " + " ".join(f"{s}={rows[s]['shipped(mean,ds8)'][3]:.4f}" for s in SCENES))
