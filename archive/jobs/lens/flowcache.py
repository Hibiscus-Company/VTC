#!/usr/bin/env python
"""Compute the DIS render<-photo flow stack ONCE per scene and cache multi-resolution
downsamples of it.  Everything downstream (ds sweep, parametric fits, interp variants,
LOVO) reads this cache, so every variant sees the SAME flow realisation -- differences
are then attributable to the field model, not to flow noise.

Identical flow computation to gsplat_track/fit_field.py (DIS MEDIUM preset, clip 6 px,
INTER_AREA downsample of the full-res flow) -- verified byte-for-byte against it.
"""
import argparse, os, sys
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
DSL = (2, 4, 8, 16, 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir)}
    files = sorted(f for f in os.listdir(args.render_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = {f"s{d}": [] for d in DSL}
    stems = []
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem not in gt_by:
            continue
        r = np.asarray(Image.open(os.path.join(args.render_dir, f)).convert("RGB"),
                       dtype=np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(args.gt_dir, gt_by[stem])).convert("RGB"),
                       dtype=np.float32) / 255.0
        if r.shape != g.shape:
            continue
        H, W, _ = r.shape
        rg = (cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        gg = (cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        fl = np.clip(dis.calc(gg, rg, None), -6.0, 6.0)
        for d in DSL:
            out[f"s{d}"].append(
                cv2.resize(fl, (W // d, H // d), interpolation=cv2.INTER_AREA).astype(np.float16))
        stems.append(stem)
    if not stems:
        sys.exit("no matched pairs")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, stems=np.array(stems),
                        HW=np.array([H, W]),
                        **{k: np.stack(v) for k, v in out.items()})
    print(f"{args.out}: {len(stems)} pairs, {H}x{W}, ds={DSL}")


if __name__ == "__main__":
    main()
