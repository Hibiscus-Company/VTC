"""Decompose the residual into MISREGISTRATION vs MISSING CONTENT.
Oracle dense flow render->GT (unachievable at test time) bounds everything a geometric
correction could ever buy, and tells us how much of the EDGE error is registration."""
import os, sys
import numpy as np, cv2, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
from common import *
cv2.setNumThreads(8)
dev = "cuda"; lp = mlib.LP(dev)
SCENE = "HCM0181"; FLD = np.load(f"/mnt/d/avv/fields/{SCENE}.npy")
st = stems(); gd, gm = gt_map(SCENE)
GQ = np.load("/home/bkai/.claude/jobs/1c9cf7e9/tmp/B_out.npz")['GQ']
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
VAR = ["base", "oracle_flow", "oracle_flow_c1px", "oracle_shift"]
acc = {v: [0.0, 0.0, 0.0] for v in VAR}
# per-mask SE before / after oracle flow
NM = 4
mse_b = np.zeros(NM); mse_a = np.zeros(NM); mn = np.zeros(NM)
fl_rms = []
for i, s in enumerate(st):
    G8 = mlib.load_u8(os.path.join(gd, gm[s]))
    K8 = apply_field(mlib.load_u8(os.path.join(K4, s + ".png")), FLD)
    gg8 = cv2.cvtColor(G8, cv2.COLOR_RGB2GRAY); kk8 = cv2.cvtColor(K8, cv2.COLOR_RGB2GRAY)
    fl = dis.calc(gg8, kk8, None)
    fl_rms.append(np.sqrt((fl ** 2).sum(-1)).mean())
    H, W = gg8.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    def warp(f):
        return np.clip(cv2.remap(K8.astype(np.float32), (xx + f[..., 0]).astype(np.float32),
                                 (yy + f[..., 1]).astype(np.float32), cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REFLECT), 0, 255).astype(np.uint8)
    outs = {"base": K8, "oracle_flow": warp(fl), "oracle_flow_c1px": warp(np.clip(fl, -1, 1))}
    # oracle global translation
    best = None
    for dy in np.arange(-1.5, 1.6, 0.25):
        for dx in np.arange(-1.5, 1.6, 0.25):
            w = warp(np.stack([np.full((H, W), dx, np.float32), np.full((H, W), dy, np.float32)], -1))
            e = ((w.astype(np.float32) - G8.astype(np.float32)) ** 2).mean()
            if best is None or e < best[0]: best = (e, w)
    outs["oracle_shift"] = best[1]
    tg = mlib.to_t(G8, dev)
    for v, a in outs.items():
        y = mlib.to_t(a, dev)
        acc[v][0] += mlib.psnr(y, tg); acc[v][1] += float(mlib.ssim(y, tg)); acc[v][2] += lp(y, tg)
    G = G8.astype(np.float32) / 255.
    seb = ((G - K8.astype(np.float32) / 255.) ** 2).mean(-1)
    sea = ((G - outs["oracle_flow"].astype(np.float32) / 255.) ** 2).mean(-1)
    gray = gg8.astype(np.float32) / 255.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    gmag = np.sqrt(gx * gx + gy * gy)
    dist = cv2.distanceTransform(1 - (gmag > GQ[2]).astype(np.uint8), cv2.DIST_L2, 3)
    ml = [gmag < GQ[0], (gmag >= GQ[0]) & (gmag < GQ[2]), gmag >= GQ[2], dist > 12]
    for j, mk in enumerate(ml):
        mse_b[j] += seb[mk].sum(); mse_a[j] += sea[mk].sum(); mn[j] += mk.sum()
    if i % 15 == 0: print("img", i, flush=True)
n = len(st)
print(f"\nmean |oracle flow| = {np.mean(fl_rms):.3f} px")
print("=== ORACLE geometric correction (upper bound, self-fitted on test GT) ===")
b0 = None
for v in VAR:
    P, S, L = [x / n for x in acc[v]]
    s2 = mlib.score(P, S, L)
    if b0 is None: b0 = s2
    print(f"  {v:18s} PSNR {P:8.4f} SSIM {S:.5f} LPIPS {L:.5f} score {s2:9.4f} ({s2-b0:+.4f})")
print("\n=== per-mask MSE before/after oracle dense flow ===")
NAMES = ["flat(<p50)", "mid(p50-95)", "edge(>p95)", "far-from-edge(>12px)"]
for j, nm in enumerate(NAMES):
    print(f"  {nm:24s} MSE {mse_b[j]/mn[j]:.6f} -> {mse_a[j]/mn[j]:.6f}  "
          f"({100*(1-mse_a[j]/mse_b[j]):5.1f}% of its SE was registration)")
print(f"  {'ALL':24s} MSE {mse_b.sum()/mn.sum():.6f} -> {mse_a.sum()/mn.sum():.6f} "
      f"({100*(1-mse_a.sum()/mse_b.sum()):5.1f}%)")
