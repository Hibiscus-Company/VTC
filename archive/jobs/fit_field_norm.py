#!/usr/bin/env python
"""fit_field with PER-PAIR PHOTOMETRIC NORMALIZATION (audit r13 item 3/6).

The pipeline's fit_field runs DIS on raw grayscale pairs. On the video scenes the
train photos have auto-exposure drift up to 63/255 (~25%) -- DIS on an exposure-shifted
pair produces spurious flow, and the bonsai field came out 1.03px mean, 5x the tower
range, DIRECTIONALLY UNCORRELATED with the COLMAP keypoint field (corr -0.07/+0.17 vs
+0.8 on towers) = contamination, not registration.

Fix: affine-match the render's grayscale mean/std to the photo's before DIS.
Everything else identical to gsplat_track/fit_field.py (flow gt->render, mean over
pairs, 1/ds grid, Rule-10 guard, .meta.json provenance so apply_field --strict accepts).
"""
import argparse, json, os
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--gt_dir", required=True, help="TRAIN photos (never test GT)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ds", type=int, default=8)
    args = ap.parse_args()

    gt_abs = os.path.abspath(args.gt_dir)
    assert os.sep + "test" + os.sep not in gt_abs + os.sep, (
        f"field must be fit on TRAIN photos, got a test dir: {gt_abs}")

    gt_by_stem = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir)}
    files = sorted(f for f in os.listdir(args.render_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    acc, n = None, 0
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem not in gt_by_stem:
            continue
        r = np.asarray(Image.open(os.path.join(args.render_dir, f)).convert("RGB"),
                       dtype=np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(args.gt_dir, gt_by_stem[stem])).convert("RGB"),
                       dtype=np.float32) / 255.0
        if r.shape != g.shape:
            continue
        H, W, _ = r.shape
        rg = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) * 255
        gg = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255
        # photometric normalization: affine-match render stats to the photo's so DIS
        # sees only geometry, not the photo's auto-exposure state
        rs = rg.std()
        if rs > 1e-3:
            rg = (rg - rg.mean()) * (gg.std() / rs) + gg.mean()
        rg8 = np.clip(rg, 0, 255).astype(np.uint8)
        gg8 = np.clip(gg, 0, 255).astype(np.uint8)
        fl = np.clip(dis.calc(gg8, rg8, None), -6.0, 6.0)
        s = cv2.resize(fl, (W // args.ds, H // args.ds), interpolation=cv2.INTER_AREA)
        acc = s if acc is None else acc + s
        n += 1
    if n == 0:
        raise SystemExit("no matched pairs")
    field = acc / n
    mag = np.linalg.norm(field, axis=2)
    print(f"  fit on {n} pairs (photonorm) -> {field.shape[1]}x{field.shape[0]}  "
          f"mean |d| {mag.mean():.3f} px  max {mag.max():.2f} px")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, field)
    with open(args.out + ".meta.json", "w") as fh:
        json.dump({"gt_dir": gt_abs, "render_dir": os.path.abspath(args.render_dir),
                   "ds": args.ds, "photonorm": True}, fh, indent=2)
    print(f"saved -> {args.out} (+ .meta.json)")


if __name__ == "__main__":
    main()
