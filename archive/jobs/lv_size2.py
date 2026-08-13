#!/usr/bin/env python
"""Byte cost of the RECOMMENDED configs (dev-estimator map), since the 350 MB cap is the binding
constraint. Reported as a PERCENTAGE so it can be applied to the real per-scene tower sizes."""
import os, sys, io
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track")
sys.path.insert(0, HERE)
from lv_s1b import fuse2
from fit_field import apply_field
Image.MAX_IMAGE_PIXELS = None

DEV = "cuda"
JPEG_KW = dict(quality=100, subsampling=2, optimize=True, progressive=True)
GT = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
MEM = [f"/mnt/d/avv/output/HCM0181_{t}/test_poses_renders_png" for t in
       ("gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7")]
FIELD = np.load(os.path.join(HERE, "HCM0181_median.npy"))

CFG = [("baseline", 0.0, None), ("full lam0.75", 0.75, None),
       ("dev2 lam0.50", 0.5, [0, 1]), ("dev2 lam0.75", 0.75, [0, 1]),
       ("dev2 lam1.00", 1.0, [0, 1]), ("dev1 lam0.75", 0.75, [0])]
stems = sorted(os.path.splitext(f)[0] for f in os.listdir(GT))
tot = {c[0]: 0 for c in CFG}
for s in stems:
    st = torch.stack([torch.from_numpy(np.asarray(Image.open(os.path.join(d, s + ".png")
                                                             ).convert("RGB"), dtype=np.float32) / 255.
                                       ).permute(2, 0, 1) for d in MEM]).to(DEV)
    for nm, lam, sub in CFG:
        img = st.mean(0, keepdim=True) if lam == 0 else fuse2(st, lam, devsub=sub)[0]
        a = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        a = (np.clip(apply_field(a.astype(np.float32) / 255.0, FIELD), 0, 1) * 255).round().astype(np.uint8)
        b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", **JPEG_KW)
        tot[nm] += b.tell()
    del st

base = tot["baseline"]
TOWER_MB, ZIP_MB, ALL_MB = 298.6, 345.0, 345.0
print(f"{'config':<14} {'MB/60':>8} {'delta%':>8} {'towers only':>13} {'all 7 scenes':>14}")
for nm, lam, sub in CFG:
    p = (tot[nm] - base) / base
    print(f"{nm:<14} {tot[nm]/1e6:8.3f} {100*p:+8.2f}% "
          f"{ZIP_MB + TOWER_MB*p:12.1f} MB {ALL_MB*(1+p):13.1f} MB")
