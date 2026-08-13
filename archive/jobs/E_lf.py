"""ORACLE bound: how much score lives in the LOW-FREQUENCY (shading/colour) part of the residual?
corrected = K + LP_sigma(G-K).  Self-fitted => strict upper bound, tells us where to aim."""
import os, sys
import numpy as np, cv2, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from common import *
cv2.setNumThreads(8)
dev = "cuda"; lp = mlib.LP(dev)
SCENE = "HCM0181"; FLD = np.load(f"/mnt/d/avv/fields/{SCENE}.npy")
st = stems(); gd, gm = gt_map(SCENE)
SIG = [4, 8, 16, 32, 64, 128]
VAR = ["base", "globalmean", "globalaffine"] + [f"lp{s}" for s in SIG]
acc = {v: [0.0, 0.0, 0.0] for v in VAR}
rem = {f"lp{s}": 0.0 for s in SIG}; tote = 0.0
for i, s in enumerate(st):
    G8 = mlib.load_u8(os.path.join(gd, gm[s]))
    K8 = apply_field(mlib.load_u8(os.path.join(K4, s + ".png")), FLD)
    G = G8.astype(np.float32) / 255.; K = K8.astype(np.float32) / 255.
    R = G - K
    tote += (R ** 2).sum()
    outs = {"base": K, "globalmean": K + R.mean((0, 1), keepdims=True)}
    a = np.zeros(3); b = np.zeros(3)
    for c in range(3):
        A = np.stack([K[..., c].ravel(), np.ones(K[..., c].size, np.float32)], 1)
        co, *_ = np.linalg.lstsq(A, G[..., c].ravel(), rcond=None); a[c], b[c] = co
    outs["globalaffine"] = K * a + b
    for sg in SIG:
        lpr = cv2.GaussianBlur(R, (0, 0), sg)
        outs[f"lp{sg}"] = K + lpr
        rem[f"lp{sg}"] += ((R - lpr) ** 2).sum()
    tg = mlib.to_t(G8, dev)
    for v, arr in outs.items():
        u8 = np.clip(np.round(arr * 255.), 0, 255).astype(np.uint8)
        y = mlib.to_t(u8, dev)
        acc[v][0] += mlib.psnr(y, tg); acc[v][1] += float(mlib.ssim(y, tg)); acc[v][2] += lp(y, tg)
    if i % 15 == 0: print("img", i, flush=True)
n = len(st)
print("\n=== ORACLE low-frequency residual correction, HCM0181 k4+field, 60 real test views ===")
print(f"  {'variant':14s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s} {'score':>9s} {'dScore':>8s} {'%SEremoved':>11s}")
b0 = None
for v in VAR:
    P, S, L = [x / n for x in acc[v]]
    sc = mlib.score(P, S, L)
    if b0 is None: b0 = sc
    rr = 100 * (1 - rem[v] / tote) if v in rem else float('nan')
    print(f"  {v:14s} {P:8.4f} {S:8.5f} {L:8.5f} {sc:9.4f} {sc-b0:+8.4f} {rr:11.2f}")
