"""LEVER: ALIGN-THEN-MERGE (burst-photography align-and-merge applied to the GS ensemble).

MECHANISM: members disagree by ~0.15px of LOCAL sub-pixel geometry (measured, L1_diag).
Pixel-mean therefore averages misregistered texture and loses ~10% of gradient energy.
Fix: warp every member into the geometry of the pixel-mean (which IS the consensus geometry --
measured global translation ~0.007px, so the mean is not biased toward any member), THEN average.
Result should keep the mean's (correct, consensus) geometry but recover the members' texture.
NO GT is used to estimate the flow -- flow is render->render only. So no GT overfitting is possible.

CONTROL 'resample': identical cubic remap with a ZERO flow, so any pure resampling tax is isolated.
"""
import os, sys, json, time
import numpy as np, cv2, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
import agg_lib as A

POOL = sys.argv[1] if len(sys.argv) > 1 else "p7"
POOLS = {
 "p7": ["gsplatB9ut","gsplatB10ut8M","gsplatB11ut60k","gsplatB12ut8Ms7","m31b_taillpips","e17visnorm","gsplatB8pure"],
 "p4": ["gsplatB9ut","gsplatB10ut8M","gsplatB11ut60k","gsplatB12ut8Ms7"],
 "pB": ["gsplatB1","gsplatB2","gsplatB3","gsplatB4warm"],
 "pC": ["e15ceil95","e16app","e17visnorm","m31b_taillpips"],
 "pD": ["gsplatB5affine","gsplatB7ppisp2","gsplatB8pure","m31b_nolpips"],
}
V = POOLS[POOL]
stems, gt_by = A.stems_for(V)
dev = A.init("cuda")
print(f"pool={POOL} k={len(V)} views={len(stems)}", flush=True)

H, W = 989, 1320
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

def mkdis():
    d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    d.setFinestScale(0); d.setPatchSize(8); d.setPatchStride(3)
    d.setUseMeanNormalization(True); d.setUseSpatialPropagation(True)
    d.setVariationalRefinementIterations(5)
    return d
DIS = mkdis()

def g8(x): return cv2.cvtColor((np.clip(x,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

def warp(img, F, interp):
    return cv2.remap(img, gx + F[...,0], gy + F[...,1], interp,
                     borderMode=cv2.BORDER_REFLECT)

def aligned_stack(X, M, clamp=1.0, interp=cv2.INTER_CUBIC):
    gm = g8(M); out = np.empty_like(X)
    for i in range(X.shape[0]):
        F = DIS.calc(gm, g8(X[i]), None)
        m = np.sqrt((F**2).sum(-1, keepdims=True))
        F = F * np.minimum(1.0, clamp/np.maximum(m, 1e-6))
        out[i] = warp(X[i], F, interp)
    return out

sc = A.Scorer(dev)
per = {}
t0 = time.time()
for n, s in enumerate(stems):
    X = np.stack([A.load(A.mdir(v), s) for v in V]).astype(np.float32)
    M = X.mean(0)
    gt = A.gt_tensor(gt_by[s], dev)
    Z = np.zeros((H, W, 2), np.float32)
    ctrl = np.stack([warp(X[i], Z, cv2.INTER_CUBIC) for i in range(X.shape[0])]).mean(0)
    Ac = aligned_stack(X, M, 1.0, cv2.INTER_CUBIC).mean(0)
    Al = aligned_stack(X, M, 1.0, cv2.INTER_LANCZOS4).mean(0)
    cand = {
        "mean": M,
        "ctrl_resample": ctrl,
        "align_cub_l1.00": Ac,
        "align_cub_l0.75": 0.25*M + 0.75*Ac,
        "align_cub_l0.50": 0.50*M + 0.50*Ac,
        "align_cub_l0.25": 0.75*M + 0.25*Ac,
        "align_lz4_l1.00": Al,
        "align_lz4_l0.50": 0.50*M + 0.50*Al,
    }
    for k, im in cand.items():
        sc.add(k, im, gt)
        p, sm, l = A.metrics(im, gt, dev) if False else (None, None, None)
    if (n+1) % 10 == 0:
        print(f"  {n+1}/{len(stems)} ({time.time()-t0:.0f}s)", flush=True)

rows = sorted(sc.table(), key=lambda r: -r[1])
base = [r for r in rows if r[0] == "mean"][0][1]
print(f"\n{'variant':22s} {'SCORE':>9} {'d(mean)':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7}")
for k, s_, P, S, L, n_ in rows:
    print(f"{k:22s} {s_:9.4f} {s_-base:+9.4f} {P:8.4f} {S:7.4f} {L:7.4f}")
json.dump({"pool": POOL, "rows": [[r[0], r[1], r[2], r[3], r[4]] for r in rows],
           "acc": {k: v for k, v in sc.acc.items()}},
          open(f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/L1_{POOL}.json", "w"))
