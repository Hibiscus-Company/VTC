#!/usr/bin/env python
"""GT-FREE: compare the production (r-1) map distribution on the PRIVATE shipped tower pool vs the
PUBLIC harness pool, and report how a gamma reshaping (r-1)^gam changes the delivered band-0 boost.
Renders only, no GT anywhere."""
import os, sys, numpy as np, torch
from PIL import Image
TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, TMP)
from lapfuse import lap_pyr, boxf, _K
Image.MAX_IMAGE_PIXELS = None


def ld(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.).permute(2, 0, 1)[None]


def rmap(ens, mems, k):
    laps, res, sizes = lap_pyr(ens, 5, _K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), 3)
    V = 0.0
    for m in mems:
        V = V + boxf(((lap_pyr(m, 5, _K)[0][0] - L0) ** 2).sum(1, keepdim=True), 3)
    V = V / len(mems) * (k / (k - 1.0))
    return (torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=4.0) - 1.0).clamp(min=0), L0, Eb


CASES = {}
T = "HCM0421"
pm = [f"/mnt/d/avv/r2r9/models/{T}_ut{s}/test_png" for s in (7, 13, 42, 77)] + \
     [f"/mnt/d/avv/r22_seed101/{T}/test_png", f"/mnt/d/avv/r25_mip3d/{T}/test_png"]
pm = [d for d in pm if os.path.isdir(d)]
CASES["PRIV_" + T] = (f"/mnt/d/avv/r29/tower_ens/{T}/png_ens", pm, 8)
SC = "HCM0181"
MEMN = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7", "gsplatB8pure",
        "e17visnorm", "e15ceil95", "e16app"]
CASES["PUB_" + SC] = (None, [f"/mnt/d/avv/output/{SC}_{t}/test_poses_renders_png" for t in MEMN], 8)

for name, (ensd, mems, k) in CASES.items():
    if not mems:
        print(f"{name}: NO MEMBER DIRS"); continue
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(mems[0]))[:4]
    R, W = [], []
    for s in stems:
        M = [ld(os.path.join(d, s + ".png")) for d in mems if os.path.exists(os.path.join(d, s + ".png"))]
        if len(M) < 2:
            continue
        ens = ld(os.path.join(ensd, s + ".png")) if ensd else torch.stack(M).mean(0)
        r1, L0, Eb = rmap(ens, M, k)
        R.append(r1.flatten().numpy()); W.append(Eb.flatten().numpy())
    if not R:
        print(f"{name}: no matching stems"); continue
    r1 = np.concatenate(R); w = np.concatenate(W)
    q = np.percentile(r1, [10, 50, 90, 99])
    print(f"\n{name}  n_img={len(R)} k={k} members={len(mems)}")
    print(f"  (r-1): mean {r1.mean():.4f}  p10 {q[0]:.4f}  p50 {q[1]:.4f}  p90 {q[2]:.4f}  p99 {q[3]:.4f}"
          f"  frac<0.25 {float((r1<0.25).mean()):.3f}  frac>1 {float((r1>1).mean()):.4f}")
    for g in (0.5, 0.7, 0.85, 1.0):
        # energy-weighted delivered boost:  sum Eb*(r-1)^g / sum Eb  (proportional to added HF amplitude)
        print(f"    gam={g:<5} unweighted mean (r-1)^g = {np.mean(r1**g):.4f}   "
              f"energy-weighted = {float((w*r1**g).sum()/w.sum()):.4f}")
