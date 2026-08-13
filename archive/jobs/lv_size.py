#!/usr/bin/env python
"""BUDGET CHECK. r27 is 345.1 MB against a 350 MB cap. Adding high-frequency energy makes JPEG
files BIGGER. If the restoration inflates the encode past the cap the change is unshippable at
q100, so measure the exact byte cost per lambda on real production-quality renders."""
import os, sys, io
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track")
sys.path.insert(0, HERE)
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fit_field import apply_field
Image.MAX_IMAGE_PIXELS = None

DEV = "cuda"
K = _K.to(DEV)
JPEG_KW = dict(quality=100, subsampling=2, optimize=True, progressive=True)
GT = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
MEM = [f"/mnt/d/avv/output/HCM0181_{t}/test_poses_renders_png" for t in
       ("gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7")]
FIELD = np.load(os.path.join(HERE, "HCM0181_median.npy"))


def fuse(st, lam, win=3):
    laps, res, sizes = lap_pyr(st, 5, K)
    L0 = laps[0]; m0 = L0.mean(0, keepdim=True)
    Eb = boxf((m0 ** 2).sum(1, keepdim=True), win)
    Em = boxf((L0 ** 2).sum(1, keepdim=True), win).mean(0, keepdim=True)
    r = torch.sqrt((Em + 1e-10) / (Eb + 1e-10)).clamp(max=4.0)
    out = [m0 * (1.0 + lam * (r - 1.0))] + [l.mean(0, keepdim=True) for l in laps[1:]]
    return lap_recon(out, res.mean(0, keepdim=True), sizes, K)


stems = sorted(os.path.splitext(f)[0] for f in os.listdir(GT))
LAMS = (0.0, 0.5, 0.75, 1.0, 1.25)
tot = {l: 0 for l in LAMS}
clip = {l: 0.0 for l in LAMS}
for s in stems:
    st = torch.stack([torch.from_numpy(np.asarray(Image.open(os.path.join(d, s + ".png")
                                                             ).convert("RGB"), dtype=np.float32) / 255.
                                       ).permute(2, 0, 1) for d in MEM]).to(DEV)
    for lam in LAMS:
        img = st.mean(0, keepdim=True) if lam == 0 else fuse(st, lam)
        clip[lam] += float(((img < 0) | (img > 1)).float().mean())
        a = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        a = (np.clip(apply_field(a.astype(np.float32) / 255.0, FIELD), 0, 1) * 255).round().astype(np.uint8)
        b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", **JPEG_KW)
        tot[lam] += b.tell()
    del st

print(f"{'lam':>5} {'MB/60 imgs':>12} {'vs lam0':>9} {'x5 towers, 386-img zip est':>30} {'clipped px':>12}")
base = tot[0.0]
for lam in LAMS:
    mb = tot[lam] / 1e6
    # r27 zip = 345.0 MB total; the 5 towers are 300 images / 298.6 MB of it (60 per tower)
    scale = 300.0 / 60.0
    print(f"{lam:>5.2f} {mb:12.3f} {100*(tot[lam]-base)/base:+8.2f}% "
          f"{345.1 + (tot[lam]-base)/1e6*scale:>29.1f} MB {100*clip[lam]/len(stems):11.4f}%")
