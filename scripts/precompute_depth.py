#!/usr/bin/env python
"""Precompute Depth-Anything-V2 monocular depth for every train image of a scene, cache as .npy
(raw float predicted_depth, keyed by image stem). Frozen generic depth net -> Rule-10 legal
(same class as VGG/AlexNet). Used by train_gsplat --depth_prior."""
import argparse, os, glob
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir of train images")
    ap.add_argument("--out", required=True, help="output dir for <stem>.npy")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pipe = pipeline("depth-estimation", model=args.model, device=args.device)
    files = sorted(glob.glob(os.path.join(args.images, "*")))
    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        outp = os.path.join(args.out, stem + ".npy")
        if os.path.exists(outp):
            continue
        im = Image.open(f).convert("RGB")
        with torch.no_grad():
            out = pipe(im)
        d = out["predicted_depth"]           # torch tensor, raw (disparity-like: near=large)
        if hasattr(d, "detach"):
            d = d.detach().float().cpu().numpy()
        d = np.asarray(d, dtype=np.float32).squeeze()
        np.save(outp, d)
        if i % 25 == 0:
            print(f"[{i}/{len(files)}] {stem} depth {d.shape} range {d.min():.2f}-{d.max():.2f}")
    print(f"done: {len(files)} depth maps -> {args.out}")


if __name__ == "__main__":
    main()
