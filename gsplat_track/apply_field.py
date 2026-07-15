#!/usr/bin/env python
"""Apply a fitted displacement field (see fit_field.py) to a directory of renders.

Writes lossless PNG. Feed the output to build_submission_zip.py, which does the
single-generation JPEG encode -- never re-encode an already-encoded JPEG.

Three guards, because every one of these fails SILENTLY (the output looks fine and
just scores worse, which we would not catch until the leaderboard):
  --strict    : the field must carry provenance proving it was fit on TRAIN photos.
                Rule 10. The D5-D8 oracle fields (fit on test GT) are on disk and
                must be unable to reach a submission.
  PNG-only in : pointing at the JPEG renders instead of the PNG archive would cost
                a second JPEG generation (measured: -0.14..-0.26).
  no re-apply : warping an already-warped dir doubles the displacement, INCREASING
                error by about what the correction removed.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image
from fit_field import apply_field

Image.MAX_IMAGE_PIXELS = None
STAMP = "field_applied.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="LOSSLESS PNG renders")
    ap.add_argument("--field", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="require train-provenance on the field (use for submissions)")
    args = ap.parse_args()

    meta_path = args.field + ".meta.json"
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        gt = meta.get("gt_dir", "")
        assert os.sep + "test" + os.sep not in gt + os.sep, (
            f"REFUSING: this field was fit against a TEST dir ({gt}). Rule 10.")
    elif args.strict:
        sys.exit(f"REFUSING (--strict): {args.field} has no .meta.json provenance; "
                 "cannot prove it was fit on TRAIN photos. Re-fit with fit_field.py.")

    if os.path.exists(os.path.join(args.in_dir, STAMP)):
        sys.exit(f"REFUSING: {args.in_dir} is already field-corrected "
                 "(double-warping would roughly double the displacement error).")

    files = sorted(f for f in os.listdir(args.in_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    bad = [f for f in files if not f.lower().endswith(".png")]
    if bad:
        sys.exit(f"REFUSING: {len(bad)} non-PNG inputs in {args.in_dir} (e.g. {bad[0]}). "
                 "Point --in_dir at the lossless PNG archive; re-encoding a JPEG "
                 "costs a second generation.")

    field = np.load(args.field)
    os.makedirs(args.out_dir, exist_ok=True)
    for f in files:
        img = np.asarray(Image.open(os.path.join(args.in_dir, f)).convert("RGB"),
                         dtype=np.float32) / 255.0
        out = np.clip(apply_field(img, field), 0.0, 1.0)
        arr = (out * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(arr).save(
            os.path.join(args.out_dir, os.path.splitext(f)[0] + ".png"))

    with open(os.path.join(args.out_dir, STAMP), "w") as fh:
        json.dump({"field": os.path.abspath(args.field),
                   "n": len(files), "source": os.path.abspath(args.in_dir),
                   "field_meta": meta}, fh, indent=2)
    print(f"corrected {len(files)} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
