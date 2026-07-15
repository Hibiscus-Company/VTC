#!/usr/bin/env python
"""Fit the systematic render->photo displacement field of a scene.

Our renders are misregistered against the photos by a small, FIXED, image-
independent 2D field (mean |d| ~0.2-0.3 px, up to ~2 px at the frame edge).
It is a camera/pipeline property: COLMAP gives us SIMPLE_RADIAL, i.e. a single
k1, and gsplat holds that constant (its distortion coeffs take no gradient --
verified), so every higher-order radial + tangential term is pinned to zero and
the residual shows up as this field.

Correcting it is worth ~+0.9 dB. Two folds of the test set agree on the field to
0.14 px (r=0.94), and a field fit on TRAIN photos alone recovers 95% of the gain
an oracle test-fitted field would -- so it is fittable WITHOUT ever touching
test GT, which is what makes it usable on the private set.

Fit it against TRAIN photos only. Never against test GT.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None


def fit_field(render_dir, gt_dir, ds=8, clip=6.0, verbose=True):
    """mean dense flow (GT -> render) over all matched pairs, at 1/ds resolution"""
    gt_by_stem = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)}
    files = sorted(f for f in os.listdir(render_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    acc, n = None, 0
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem not in gt_by_stem:
            continue
        r = np.asarray(Image.open(os.path.join(render_dir, f)).convert("RGB"),
                       dtype=np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(gt_dir, gt_by_stem[stem])).convert("RGB"),
                       dtype=np.float32) / 255.0
        if r.shape != g.shape:
            continue
        H, W, _ = r.shape
        rg = (cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        gg = (cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        # flow from photo to render: sampling the render at (x+flow) lands it on the photo
        fl = np.clip(dis.calc(gg, rg, None), -clip, clip)
        s = cv2.resize(fl, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
        acc = s if acc is None else acc + s
        n += 1
    if n == 0:
        raise SystemExit(f"no matched pairs between {render_dir} and {gt_dir}")
    field = acc / n
    if verbose:
        mag = np.linalg.norm(field, axis=2)
        print(f"  fit on {n} pairs -> field {field.shape[1]}x{field.shape[0]}  "
              f"mean |d| {mag.mean():.3f} px  max {mag.max():.2f} px")
    return field


def apply_field(img, field):
    """img float32 HxWx3 in [0,1] -> corrected"""
    H, W, _ = img.shape
    fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, (xx + fu[..., 0]).astype(np.float32),
                     (yy + fu[..., 1]).astype(np.float32),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True, help="renders at TRAIN poses")
    ap.add_argument("--gt_dir", required=True, help="TRAIN photos (never test GT)")
    ap.add_argument("--out", required=True, help=".npy to write")
    ap.add_argument("--ds", type=int, default=8)
    args = ap.parse_args()

    # RULE 10 GUARD. The D5-D8 diagnostics fitted ORACLE fields against public TEST
    # GT to size the prize; those .npy files exist on disk. Fitting against test GT
    # and shipping the result would be a real violation, so make it unreachable
    # rather than merely unintended -- a submission must never depend on us
    # remembering which directory we typed.
    gt_abs = os.path.abspath(args.gt_dir)
    assert os.sep + "test" + os.sep not in gt_abs + os.sep, (
        f"field must be fit on TRAIN photos, got a test dir: {gt_abs}")

    f = fit_field(args.render_dir, args.gt_dir, ds=args.ds)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, f)
    # provenance sidecar: apply_field --strict refuses any field that cannot prove
    # it was fit on train photos
    with open(args.out + ".meta.json", "w") as fh:
        json.dump({"gt_dir": gt_abs,
                   "render_dir": os.path.abspath(args.render_dir),
                   "ds": args.ds}, fh, indent=2)
    print(f"saved -> {args.out} (+ .meta.json)")


if __name__ == "__main__":
    main()
