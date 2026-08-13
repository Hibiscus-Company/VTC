#!/usr/bin/env python
"""jpegalloc STAGE 1: exact BYTE cost of every encode setting, on the REAL r29 masters.

No scoring here -- this is the cost side of the knapsack only. Score side is ja_score.py.
Samples SAMP images per scene, extrapolates to the scene's true image count, and reports the
delta against what r29 actually shipped (Q99 for HCM0421, Q100 elsewhere, all subsampling=2).
"""
import io, os, sys, json
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SHIP = dict(optimize=True, progressive=True)
SCENES = {
    "HCM0421": ("/mnt/d/avv/r29/tower_ens/HCM0421/png", 60, 99),
    "HCM0539": ("/mnt/d/avv/r29/tower_ens/HCM0539/png", 60, 100),
    "HCM0540": ("/mnt/d/avv/r29/tower_ens/HCM0540/png", 60, 100),
    "HCM0644": ("/mnt/d/avv/r29/tower_ens/HCM0644/png", 60, 100),
    "HCM0674": ("/mnt/d/avv/r29/tower_ens/HCM0674/png", 60, 100),
    "chair":   ("/mnt/d/avv/r29/video_ens/chair/png", 58, 100),
    "bonsai":  ("/mnt/d/avv/r29/video_ens/bonsai/png", 28, 100),
}
ARMS = [("q100ss2", 100, 2), ("q99ss2", 99, 2), ("q98ss2", 98, 2), ("q97ss2", 97, 2),
        ("q95ss2", 95, 2), ("q100ss1", 100, 1), ("q100ss0", 100, 0), ("q98ss0", 98, 0),
        ("q97ss0", 97, 0), ("q95ss0", 95, 0)]
SAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 14

out = {}
for sc, (d, n, qship) in SCENES.items():
    stems = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))
    idx = np.linspace(0, len(stems) - 1, min(SAMP, len(stems))).astype(int)
    ims = [Image.open(os.path.join(d, stems[i])).convert("RGB") for i in idx]
    row = {}
    for name, q, ss in ARMS:
        tot = 0
        for im in ims:
            b = io.BytesIO()
            im.save(b, "JPEG", quality=q, subsampling=ss, **SHIP)
            tot += b.tell()
        row[name] = tot / len(ims) * n
    out[sc] = dict(bytes=row, n=n, shipped=f"q{qship}ss2")
    print(f"{sc:<9} shipped=q{qship}ss2  " + "  ".join(
        f"{k}:{v/1e6:.1f}" for k, v in row.items()), flush=True)

json.dump(out, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/ja_bytes.json", "w"), indent=1)

ship = sum(out[s]["bytes"][out[s]["shipped"]] for s in SCENES)
print(f"\nmodelled r29 total {ship/1e6:.2f} MB (actual zip payload 349.72 MB)")
print(f"\n{'scene':<9}" + "".join(f"{a:>10}" for a, _, _ in ARMS))
for sc in SCENES:
    b = out[sc]["bytes"]
    base = b[out[sc]["shipped"]]
    print(f"{sc:<9}" + "".join(f"{(b[a]-base)/1e6:>+10.2f}" for a, _, _ in ARMS))
print("(delta MB vs what r29 shipped for that scene)")
