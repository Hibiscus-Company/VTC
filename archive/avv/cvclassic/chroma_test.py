#!/usr/bin/env python
"""DECISIVE CONTROL for the chroma-bandwidth claim.

The raw PNG-vs-JPEG comparison is confounded: GT is 4:2:0 so its chroma above
f=0.25 is pure upsampling interpolation. Fair test = put BOTH through the shipped
encode (q100, subsampling=2), then measure, then score the intervention.

Arms written as JPEG at the shipped profile:
  base            plain ensemble mean
  cN              chroma-only gaussian prefilter, sigma N, luma untouched
"""
import os, sys, argparse
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--outroot", required=True)
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.4, 0.7, 1.0, 1.5])
ap.add_argument("--q", type=int, default=100)
ap.add_argument("--ss", type=int, default=2)
a = ap.parse_args()

M = np.array([[0.299, 0.587, 0.114],
              [-0.168736, -0.331264, 0.5],
              [0.5, -0.418688, -0.081312]], dtype=np.float32)
Minv = np.linalg.inv(M)

files = sorted(f for f in os.listdir(a.src)
               if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg"))
for s in a.sigmas:
    d = os.path.join(a.outroot, f"c{s:g}")
    os.makedirs(d, exist_ok=True)
for f in files:
    rgb = np.asarray(Image.open(os.path.join(a.src, f)).convert("RGB"), dtype=np.float32)
    ycc = rgb @ M.T
    for s in a.sigmas:
        y = ycc.copy()
        if s > 0:
            k = int(2 * round(3 * s) + 1)
            y[..., 1] = cv2.GaussianBlur(y[..., 1], (k, k), s)
            y[..., 2] = cv2.GaussianBlur(y[..., 2], (k, k), s)
        out = np.clip(y @ Minv.T + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(out).save(
            os.path.join(a.outroot, f"c{s:g}", os.path.splitext(f)[0] + ".JPG"),
            quality=a.q, subsampling=a.ss, optimize=True)
print("wrote", len(files), "x", len(a.sigmas))
