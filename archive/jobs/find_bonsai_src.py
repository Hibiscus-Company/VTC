#!/usr/bin/env python
"""r14/bonsai_ens/png does NOT reproduce r21's shipped bonsai bytes. Find what does, before we
rebuild bonsai for r23 from the wrong pixels. Try (a) other quality settings against that dir,
(b) every other candidate png dir on disk. Also report how DIFFERENT the pixels are, so we know
whether this is merely an encode-setting mismatch or genuinely different renders."""
import io, os, zipfile
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
R21 = "/mnt/d/avv/submissions/sub_round21_chairdepth_hcm0674ema.zip"
z = zipfile.ZipFile(R21)
arc = sorted(n for n in z.namelist() if n.startswith("bonsai/"))
print(f"{len(arc)} bonsai entries in r21")
ref0 = z.read(arc[0])
im0 = Image.open(io.BytesIO(ref0))
print(f"shipped bonsai[0]: {im0.size} mode={im0.mode} bytes={len(ref0)}")
ship_px = np.asarray(im0.convert("RGB"), np.float32)

cands = []
for root, dirs, files in os.walk("/mnt/d/avv"):
    if os.path.basename(root) in ("png", "png_ens", "eval_png", "test_png", "ens"):
        pngs = [f for f in files if f.lower().endswith(".png")]
        if len(pngs) == len(arc):
            stems = {os.path.splitext(f)[0] for f in pngs}
            want = {os.path.splitext(os.path.basename(a))[0] for a in arc}
            if stems == want:
                cands.append(root)
    if root.count(os.sep) > 7:
        dirs[:] = []
print(f"\n{len(cands)} candidate dirs with matching filename set:")

stem0 = os.path.splitext(os.path.basename(arc[0]))[0]
for d in cands:
    p = os.path.join(d, stem0 + ".png")
    px = np.asarray(Image.open(p).convert("RGB"), np.float32)
    if px.shape != ship_px.shape:
        print(f"  {d}: SHAPE {px.shape} != {ship_px.shape}"); continue
    mae = np.abs(px - ship_px).mean()
    print(f"  {d}: mean|px-shipped| = {mae:.4f}")

print("\n=== quality probe on the closest dirs (does an encode setting reproduce the bytes?) ===")
for d in cands:
    im = Image.open(os.path.join(d, stem0 + ".png")).convert("RGB")
    for q in (100, 99, 98, 97, 96, 95):
        for ss in (0, 2):
            for prog in (True, False):
                b = io.BytesIO()
                kw = dict(quality=q, subsampling=ss, optimize=True)
                if prog:
                    kw["progressive"] = True
                im.save(b, "JPEG", **kw)
                if b.getvalue() == ref0:
                    print(f"  MATCH: {d}  q{q} ss{ss} prog={prog}")
