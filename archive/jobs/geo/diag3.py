#!/usr/bin/env python
"""DIAGNOSIS 3: per-pyramid-LEVEL correlation with GT, by geography.

All prior band-gain proofs are about L0 (the top octave). Levels 1..4 in FLAT regions have
never been characterised. If our flat-region MID-frequency content is spurious (3DGS sky
blotches) it will show up as matched ENERGY but LOW CORRELATION -- and the MSE-optimal gain
will be well below 1 at that level.
"""
import io, os, sys
import numpy as np
import cv2
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, _K
from energy_restore import restore
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NLEV = 5


def loadt(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def enc(x):
    b = io.BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.float32) / 255.0


gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
stems = stems[::max(1, len(stems) // N)][:N]
cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")

REG = ("flat", "mid", "edge")
st = {(r, l): np.zeros(3) for r in REG for l in range(NLEV + 1)}   # <X,G>, E_X, E_G
for si, s in enumerate(stems):
    mem = [loadt(os.path.join(d, s + ".png")) for d in MEM]
    ens = torch.stack(mem).mean(0)
    out = restore(ens, mem, 1.0, len(mem), 3).clamp(0, 1)
    X = np.clip(warp(out[0].permute(1, 2, 0).numpy(), lens, "lanczos"), 0, 1)
    X = enc(X)
    G = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"), dtype=np.float32) / 255.0

    xg = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(xg, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(xg, cv2.CV_32F, 0, 1, 3)
    gm = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 3.0)
    t_lo, t_hi = np.percentile(gm, [35.0, 73.0])
    base_masks = {"flat": gm <= t_lo, "edge": gm >= t_hi}
    base_masks["mid"] = ~base_masks["flat"] & ~base_masks["edge"]

    Xt = torch.from_numpy(np.ascontiguousarray(X)).permute(2, 0, 1).unsqueeze(0)
    Gt = torch.from_numpy(np.ascontiguousarray(G)).permute(2, 0, 1).unsqueeze(0)
    lx, rx, _ = lap_pyr(Xt, NLEV, _K)
    lg, rg, _ = lap_pyr(Gt, NLEV, _K)
    bands_x = [b[0].permute(1, 2, 0).numpy() for b in lx] + [rx[0].permute(1, 2, 0).numpy()]
    bands_g = [b[0].permute(1, 2, 0).numpy() for b in lg] + [rg[0].permute(1, 2, 0).numpy()]
    for l in range(NLEV + 1):
        h, w, _ = bands_x[l].shape
        for r in REG:
            M = cv2.resize(base_masks[r].astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            bx, bg_ = bands_x[l][M], bands_g[l][M]
            st[(r, l)] += [float((bx * bg_).sum()), float((bx * bx).sum()), float((bg_ * bg_).sum())]
    print(f"  {si+1}/{len(stems)}", flush=True)

print(f"\n=== per-LEVEL structure, HCM0181, n={len(stems)}, after full shipped chain ===")
print("level 0 = top octave (0.25-0.5 cyc/px), each level halves the frequency; 5 = LF residual")
for r in REG:
    print(f"\n[{r}]  {'level':>6} {'corr':>8} {'E_ours/E_GT':>12} {'MSE-opt gain':>13} "
          f"{'band MSE':>11} {'MSE at opt':>11} {'dMSE':>11}")
    for l in range(NLEV + 1):
        d, ex, eg = st[(r, l)]
        corr = d / np.sqrt(max(ex * eg, 1e-30))
        g = d / max(ex, 1e-30)
        mse = ex + eg - 2 * d
        mse_opt = eg - d * d / max(ex, 1e-30)
        print(f"      {l:>6} {corr:8.4f} {ex/eg:12.4f} {g:13.4f} {mse:11.4e} {mse_opt:11.4e} "
              f"{mse_opt-mse:11.4e}")
