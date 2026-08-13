#!/usr/bin/env python
"""Score the D5 registration oracle FULLY (LPIPS+SSIM+PSNR), not just dB.

The consult's second point: I declared registration "dead" from its dB gain (+2.13 -> 28.6),
but the metric rewards registration through SSIM, which is very sensitive to misregistration.
So the oracle's SCORE gain could be much larger than its dB gain implies.

Take our best public single render, fit dense flow vs TEST GT (<=6px, perfect per-image
registration = CHEATING, diagnosis only), warp, and score with the REAL competition metric.
This tells us the SCORE ceiling of perfect registration -- and whether SSIM/registration is a
bigger lever than the dB reading suggested.
"""
import os, sys
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None

SCENE = "HCM0181"
GT = os.path.expanduser(f"~/data/phase1/public_set/{SCENE}/test/images")
REND = f"/mnt/d/avv/output/{SCENE}_gsplatB11ut60k/test_poses_renders_png"
OUT = os.path.expanduser("~/densq/oracle_reg/" + SCENE)
os.makedirs(OUT, exist_ok=True)

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
n = 0
for f in sorted(os.listdir(REND)):
    s = os.path.splitext(f)[0]
    if s not in gt_by:
        continue
    r = np.asarray(Image.open(os.path.join(REND, f)).convert("RGB"), dtype=np.float32) / 255
    g = np.asarray(Image.open(os.path.join(GT, gt_by[s])).convert("RGB"), dtype=np.float32) / 255
    if r.shape != g.shape:
        continue
    H, W, _ = r.shape
    rg = (cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
    gg = (cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
    fl = np.clip(dis.calc(gg, rg, None), -6, 6)      # perfect per-image registration (cheat)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    w = cv2.remap(r, (xx + fl[..., 0]).astype(np.float32),
                  (yy + fl[..., 1]).astype(np.float32),
                  cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    arr = (np.clip(w, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT, s + ".png"))
    n += 1
print(f"wrote {n} oracle-registered renders -> {OUT}")
