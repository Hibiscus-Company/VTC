#!/usr/bin/env python
"""HOISTED SHARED COMPUTATION: build, once, the shipped-chain state just before the JPEG encode.

cache/<stem>.npz holds
   B   float16 HxWx3  = restore(mean of 4 members, lam=1.0) then median-lens-field lanczos4 warp
   G   float16 HxWx3  = real test GT
   D   float16 HxW    = distance (px) to the nearest strong GT edge  -> any flat mask is a threshold
Every arm afterwards is  B -> operator -> JPEG q100/ss2 -> score, so nothing shared is recomputed.
"""
import os, sys
import numpy as np, cv2, torch
from PIL import Image

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0, TMP)
sys.path.insert(0, os.path.join(TMP, "lens"))
from fieldlib import LooPool, upsample, warp
from energy_restore import restore
cv2.setNumThreads(8); Image.MAX_IMAGE_PIXELS = None

SC = sys.argv[1] if len(sys.argv) > 1 else "HCM0181"
POOL = {"HCM0181": ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]}.get(SC, ["gsplatB9ut"])
MEM = [f"/mnt/d/avv/output/{SC}_{t}/test_poses_renders_png" for t in POOL]
GTD = f"/mnt/d/avv/data/phase1/public_set/{SC}/test/images"
OUT = f"/mnt/d/avv/geoflat/cache_{SC}"; os.makedirs(OUT, exist_ok=True)
LAM = 1.0 if len(POOL) > 1 else 0.0


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
cache = np.load(f"{TMP}/lens/cache/pub_{SC}.npz")
lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
print(SC, len(stems), "stems, k =", len(POOL), "lam =", LAM, flush=True)

for i, s in enumerate(stems):
    f = os.path.join(OUT, s + ".npz")
    if os.path.exists(f):
        continue
    mem = [load(os.path.join(d, s + ".png")) for d in MEM]
    ens = torch.stack(mem).mean(0)
    B = restore(ens, mem, LAM, len(mem), 3).clamp(0, 1) if LAM > 0 else ens
    B = np.ascontiguousarray(B[0].permute(1, 2, 0).numpy())
    B = np.clip(warp(B, lens, "lanczos"), 0, 1)
    G = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"), dtype=np.float32) / 255.
    gy = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)
    gm = np.abs(cv2.Sobel(gy, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(gy, cv2.CV_32F, 0, 1, 3))
    edge = (gm > np.percentile(gm, 80)).astype(np.uint8)
    D = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 3)
    np.savez(f, B=B.astype(np.float16), G=G.astype(np.float16), D=D.astype(np.float16))
    print(i, end=" ", flush=True)
print("\ndone")
