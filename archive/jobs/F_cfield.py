"""Production-shaped, ZERO-LEAKAGE test: fit a fixed low-frequency COLOUR field on TRAIN
(photo - field-warped train render, median over train views), apply at TEST. 5 scenes.
Same logic that made the lens field worth +0.73 on the LB, but for photometry."""
import os, sys
import numpy as np, cv2, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from common import *
cv2.setNumThreads(8)
dev = "cuda"; lp = mlib.LP(dev)
SCENES = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
SIG = [8, 16, 32, 64]; GAIN = [0.5, 1.0]
DS = 8
rows = {}
for sc_ in SCENES:
    fld = np.load(f"/mnt/d/avv/fields/{sc_}.npy")
    trd = f"/mnt/d/avv/output/{sc_}_gsplatB9ut/train_renders"
    tgd = f"/mnt/d/avv/data/phase1/public_set/{sc_}/train/images"
    tgm = {os.path.splitext(f)[0]: f for f in os.listdir(tgd)}
    tf = sorted(os.listdir(trd))
    acc = []
    for f in tf:
        stem = os.path.splitext(f)[0]
        if stem not in tgm: continue
        r = apply_field(mlib.load_u8(os.path.join(trd, f)), fld).astype(np.float32) / 255.
        g = mlib.load_u8(os.path.join(tgd, tgm[stem])).astype(np.float32) / 255.
        d = g - r
        acc.append(cv2.resize(d, (d.shape[1] // DS, d.shape[0] // DS), interpolation=cv2.INTER_AREA))
    A = np.stack(acc)
    Cmed = np.median(A, 0); Cmean = A.mean(0)
    consist = 1 - (np.median(np.abs(A - Cmed), 0).mean() / np.abs(A).mean())
    print(f"{sc_}: {len(acc)} train pairs, |C_med| mean {np.abs(Cmed).mean()*255:.3f}/255, "
          f"view-consistency {consist:.3f}", flush=True)

    ted = f"/mnt/d/avv/output/{sc_}_gsplatB9ut/test_poses_renders_png"
    gdir, gmp = gt_map(sc_)
    ss = [os.path.splitext(f)[0] for f in sorted(os.listdir(ted))]
    variants = {"base": None}
    for s_ in SIG:
        for gn in GAIN:
            variants[f"med_s{s_}_g{gn}"] = ("med", s_, gn)
    variants["mean_s32_g1.0"] = ("mean", 32, 1.0)
    A_ = {v: [0.0, 0.0, 0.0] for v in variants}
    Cs = {}
    for s_ in SIG:
        Cs[("med", s_)] = cv2.GaussianBlur(Cmed, (0, 0), max(s_ / DS, 0.8))
        Cs[("mean", s_)] = cv2.GaussianBlur(Cmean, (0, 0), max(s_ / DS, 0.8))
    for stem in ss:
        K8 = apply_field(mlib.load_u8(os.path.join(ted, stem + ".png")), fld)
        K = K8.astype(np.float32) / 255.
        G8 = mlib.load_u8(os.path.join(gdir, gmp[stem]))
        tg = mlib.to_t(G8, dev)
        for v, cfg in variants.items():
            if cfg is None: out = K8
            else:
                kind, s_, gn = cfg
                C = cv2.resize(Cs[(kind, s_)], (K.shape[1], K.shape[0]), interpolation=cv2.INTER_CUBIC)
                out = np.clip(np.round((K + gn * C) * 255.), 0, 255).astype(np.uint8)
            y = mlib.to_t(out, dev)
            A_[v][0] += mlib.psnr(y, tg); A_[v][1] += float(mlib.ssim(y, tg)); A_[v][2] += lp(y, tg)
    n = len(ss)
    b0 = None; rows[sc_] = {}
    for v in variants:
        P, S, L = [x / n for x in A_[v]]
        s2 = mlib.score(P, S, L)
        if b0 is None: b0 = s2
        rows[sc_][v] = (P, S, L, s2, s2 - b0)
    print("   " + "  ".join(f"{v}:{rows[sc_][v][4]:+.4f}" for v in variants if v != "base"), flush=True)

print("\n=== TRAIN-FITTED LF COLOUR FIELD, delta score vs (B9ut + lens field) ===")
vs = [v for v in rows[SCENES[0]] if v != "base"]
print(f"  {'variant':16s} " + " ".join(f"{s:>9s}" for s in SCENES) + f" {'MEAN':>9s}")
for v in vs:
    ds = [rows[s][v][4] for s in SCENES]
    print(f"  {v:16s} " + " ".join(f"{d:+9.4f}" for d in ds) + f" {np.mean(ds):+9.4f}")
print("\n  base scores: " + " ".join(f"{s}={rows[s]['base'][3]:.4f}" for s in SCENES))
