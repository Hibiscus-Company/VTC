#!/usr/bin/env python
"""Sanity: confirm /mnt/d/avv/prodharness/k4/png == uint8 mean of the 4 UT members."""
import os, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

K4 = "/mnt/d/avv/prodharness/k4/png"
MEMBERS = [
    "/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
    "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
    "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
    "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png",
]
GT = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"

stems = sorted(os.path.splitext(f)[0] for f in os.listdir(K4))
print("k4 pngs:", len(stems))
gtf = sorted(os.listdir(GT))
print("gt files:", len(gtf), gtf[:2])
im = Image.open(os.path.join(K4, stems[0] + ".png"))
print("k4 size/mode:", im.size, im.mode)
im2 = Image.open(os.path.join(GT, gtf[0]))
print("gt size/mode:", im2.size, im2.mode, gtf[0])

for s in stems[:3]:
    ms = [np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"), dtype=np.float32) for d in MEMBERS]
    mean = np.mean(ms, axis=0)
    k4 = np.asarray(Image.open(os.path.join(K4, s + ".png")).convert("RGB"), dtype=np.float32)
    print(s, "maxabs(round(mean)-k4) =", np.abs(np.round(mean) - k4).max(),
          " maxabs(mean-k4) =", np.abs(mean - k4).max())
