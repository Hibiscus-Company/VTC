#!/usr/bin/env python
"""Measure pool disagreement (mean_i mean_px |m_i - mean| * 255) for candidate HCM0181 pools."""
import os, sys, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
O = "/mnt/d/avv/output"
SC = "HCM0181"
ALL10 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7", "gsplatB8pure",
         "e17visnorm", "e15ceil95", "e16app", "gsplatB5affine", "gsplatB6bilagrid"]
POOLS = {
    "k10_diverse": ALL10,
    "k8_diverse": ALL10[:8],
    "k4_ut": ALL10[:4],
    "k8_utfam": ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
                 "gsplatB1", "gsplatB2", "gsplatB3", "gsplatB4warm"],
    "k6_sh": ["sh0", "sh1", "sh2", "sh3", "gsplatB9ut", "gsplatB8pure"],
}
GTD = f"/mnt/d/avv/data/phase1/public_set/{SC}/test/images"
stems = sorted(os.path.splitext(f)[0] for f in os.listdir(GTD))[:8]
for name, mems in POOLS.items():
    ds = []
    for s in stems:
        A = []
        ok = True
        for m in mems:
            p = f"{O}/{SC}_{m}/test_poses_renders_png/{s}.png"
            if not os.path.exists(p):
                ok = False; break
            A.append(np.asarray(Image.open(p).convert("RGB"), np.float32))
        if not ok:
            ds = None; break
        A = np.stack(A)
        mu = A.mean(0)
        ds.append(np.abs(A - mu).mean())
    print(f"{name:14s} k={len(mems):2d} disagreement={np.mean(ds):.3f}/255" if ds else f"{name}: MISSING", flush=True)
