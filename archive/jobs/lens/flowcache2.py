#!/usr/bin/env python
"""Alternative FLOW ESTIMATORS for the same field, cached in the same format as flowcache.py.

The shipped field is cv2 DIS at PRESET_MEDIUM, whose finest pyramid level is 1 (half
resolution).  Our displacements are ~0.2 px, an order of magnitude below what that
configuration is designed to resolve, so the estimator itself is a candidate bottleneck.

  dis0  : DIS MEDIUM with finest_scale=0 (full-resolution finest level) + 25 variational
          refinement iterations
  dis2x : DIS MEDIUM on both images upsampled 2x (Lanczos), flow halved -- buys sub-pixel
          resolution the pyramid otherwise throws away
  lk    : block Lucas-Kanade.  Solves the brightness-constancy normal equations over each
          block, so it is analytically sub-pixel with no pyramid quantisation at all.
"""
import argparse, os, sys
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(1)      # this box is shared; DIS otherwise grabs every core
DSL = (4, 8, 16, 32)


def make_dis(kind):
    d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    if kind == "dis0":
        d.setFinestScale(0)
        d.setVariationalRefinementIterations(25)
    return d


def lk_field(gg, rg, block, H, W):
    """dense least-squares optical flow on `block`-sized tiles (sub-pixel by construction)."""
    a = gg.astype(np.float32) / 255.0
    b = rg.astype(np.float32) / 255.0
    Ix = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    It = b - a
    k = (block, block)
    sxx = cv2.boxFilter(Ix * Ix, -1, k, normalize=False)
    syy = cv2.boxFilter(Iy * Iy, -1, k, normalize=False)
    sxy = cv2.boxFilter(Ix * Iy, -1, k, normalize=False)
    sxt = cv2.boxFilter(Ix * It, -1, k, normalize=False)
    syt = cv2.boxFilter(Iy * It, -1, k, normalize=False)
    det = sxx * syy - sxy * sxy
    reg = 1e-3 * (sxx + syy).mean()
    det = det + reg * reg
    u = (-syy * sxt + sxy * syt) / det
    v = (sxy * sxt - sxx * syt) / det
    return np.clip(np.stack([u, v], axis=2), -6.0, 6.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--kind", required=True, choices=("dis0", "dis2x", "lk"))
    ap.add_argument("--block", type=int, default=17)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir)}
    files = sorted(f for f in os.listdir(args.render_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    dis = make_dis("dis0" if args.kind == "dis0" else "med")
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
        if args.kind == "dis2x":
            rg2 = cv2.resize(rg, (W * 2, H * 2), interpolation=cv2.INTER_LANCZOS4)
            gg2 = cv2.resize(gg, (W * 2, H * 2), interpolation=cv2.INTER_LANCZOS4)
            fl2 = dis.calc(gg2, rg2, None) * 0.5
            fl = cv2.resize(fl2, (W, H), interpolation=cv2.INTER_AREA)
            fl = np.clip(fl, -6.0, 6.0)
        elif args.kind == "lk":
            fl = lk_field(gg, rg, args.block, H, W)
        else:
            fl = np.clip(dis.calc(gg, rg, None), -6.0, 6.0)
        for d in DSL:
            out[f"s{d}"].append(
                cv2.resize(fl, (W // d, H // d), interpolation=cv2.INTER_AREA).astype(np.float16))
        stems.append(stem)
    np.savez_compressed(args.out, stems=np.array(stems), HW=np.array([H, W]),
                        **{k: np.stack(v) for k, v in out.items()})
    print(f"{args.out}: {len(stems)} pairs {args.kind}")


if __name__ == "__main__":
    main()
