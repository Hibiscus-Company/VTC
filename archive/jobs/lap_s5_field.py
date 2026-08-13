#!/usr/bin/env python
"""STEP 5: does the gain SURVIVE THE REST OF THE PRODUCTION PIPELINE?

Production tower recipe:  weighted pixel-mean of members -> apply median lens field -> JPEG q100 ss2
The lens field is a sub-pixel cv2.remap(INTER_CUBIC) resample -- exactly the kind of operation that
can eat a finest-pyramid-level texture boost. So measure the FULL composed chain.

Field used here was fit BY ME on HCM0181's TRAIN photos (Rule 10 clean: gt_dir is .../train/...,
zero stem overlap with the 60 test views), median estimator, mean|d| 0.211 px -- same recipe and
same magnitude as the shipped tower fields.
"""
import os, sys, io
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_field import apply_field
from lap_run import Harness, agg, JPEG_KW
from lapfuse import fuse_image, _K

FIELD = np.load("/home/bkai/.claude/jobs/1c9cf7e9/tmp/HCM0181_median.npy")
ROWS = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lap_rows"
CFGS = {"pixel-mean (SHIPPED)": None,
        "LAPFUSE lam0.50 win3": dict(nalt=1, rule="energy", kw=dict(lam=0.5, win=3)),
        "LAPFUSE lam0.75 win3": dict(nalt=1, rule="energy", kw=dict(lam=0.75, win=3)),
        "LAPFUSE lam1.00 win3": dict(nalt=1, rule="energy", kw=dict(lam=1.0, win=3)),
        "LAPFUSE lam1.25 win3": dict(nalt=1, rule="energy", kw=dict(lam=1.25, win=3))}
H = Harness(["m1", "m2", "m3", "m4"])
k = _K.to("cuda")


def chain(i, cfg, field, jpeg):
    st = H.stack(i)
    img = st.mean(0, keepdim=True) if cfg is None else fuse_image(st, cfg, H.nlev, k)
    a = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    if field:
        a = (np.clip(apply_field(a.astype(np.float32) / 255.0, FIELD), 0, 1) * 255).round().astype(np.uint8)
    if jpeg:
        buf = io.BytesIO(); Image.fromarray(a).save(buf, "JPEG", **JPEG_KW); buf.seek(0)
        a = np.asarray(Image.open(buf).convert("RGB"))
    r = torch.from_numpy(a).to("cuda").float().permute(2, 0, 1).unsqueeze(0) / 255.0
    g = H.gt(i)
    with torch.no_grad():
        return (10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12)),
                float(H.ssim(r, g)), float(H.vgg(r * 2 - 1, g * 2 - 1).item()))


res = {}
for nm, cfg in CFGS.items():
    for field in (False, True):
        for jpeg in (False, True):
            tag = nm.replace(" ", "").replace("(", "").replace(")", "")
            f = f"{ROWS}/chain_{tag}_{int(field)}{int(jpeg)}.npy"
            if os.path.exists(f):
                rows = np.load(f)
            else:
                rows = np.array([chain(i, cfg, field, jpeg) for i in range(len(H.stems))])
                np.save(f, rows)
            res[(nm, field, jpeg)] = rows

order = [(False, False, "raw PNG"), (True, False, "+field  PNG"),
         (False, True, "+JPEG"), (True, True, "+field +JPEG  <<< SHIPPED CHAIN")]
for field, jpeg, label in order:
    print("\n" + "=" * 84)
    print(f"  {label}")
    print("=" * 84)
    print(f"{'variant':<26} {'SCORE':>9} {'delta':>8} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8}")
    b = agg(res[("pixel-mean (SHIPPED)", field, jpeg)])
    for nm in CFGS:
        s, P, S, L = agg(res[(nm, field, jpeg)])
        print(f"{nm:<26} {s:9.4f} {s-b[0]:+8.4f} {P:8.4f} {S:7.4f} {L:8.4f}")
