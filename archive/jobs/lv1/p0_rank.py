#!/usr/bin/env python
"""Rank the 21 HCM0181 test-pose render variants by raw PSNR on a 10-image subsample,
so the k-sweep pool is built from comparable-quality members (CPU only, no GPU touch)."""
import os, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = "/mnt/d/avv/output"
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}

variants = sorted(d for d in os.listdir(ROOT)
                  if d.startswith("HCM0181_")
                  and os.path.isdir(os.path.join(ROOT, d, "test_poses_renders_png"))
                  and len(os.listdir(os.path.join(ROOT, d, "test_poses_renders_png"))) == 60)
stems = sorted(gt_by)
sub = stems[::6][:10]
print(f"{len(variants)} variants, {len(sub)} sample images")

gts = {}
for s in sub:
    gts[s] = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"), np.float32) / 255.

rows = []
for v in variants:
    d = os.path.join(ROOT, v, "test_poses_renders_png")
    ps = []
    for s in sub:
        p = os.path.join(d, s + ".png")
        if not os.path.exists(p):
            ps = None
            break
        r = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
        if r.shape != gts[s].shape:
            ps = None
            break
        ps.append(10 * np.log10(1.0 / max(float(((r - gts[s]) ** 2).mean()), 1e-12)))
    if ps:
        rows.append((float(np.mean(ps)), v))
rows.sort(reverse=True)
for p, v in rows:
    print(f"{p:7.3f}  {v}")
