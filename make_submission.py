#
# Package rendered test images into the competition submission zip.
#
# Expects a base directory containing one subdirectory per scene (named after
# the scene), each holding the rendered images for that scene's test_poses.csv.
# Validates count, names and image sizes against the CSVs before zipping.
#
# Usage:
#   python make_submission.py --renders_root submission_renders \
#       --data_roots ~/data/phase1/public_set ~/data/phase1/private_set1 \
#       --out submission.zip --ext jpg
#

import os
import csv
import glob
import argparse
import zipfile
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders_root", required=True)
    parser.add_argument("--data_roots", nargs="+", required=True)
    parser.add_argument("--out", default="submission.zip")
    parser.add_argument("--ext", default=None, help="override output extension, e.g. png (default: keep csv name)")
    args = parser.parse_args()

    scenes = {}
    for root in args.data_roots:
        root = os.path.expanduser(root)
        for scene_dir in sorted(glob.glob(os.path.join(root, "*"))):
            csv_path = os.path.join(scene_dir, "test", "test_poses.csv")
            if os.path.isdir(scene_dir) and os.path.exists(csv_path):
                scenes[os.path.basename(scene_dir)] = csv_path

    print(f"{len(scenes)} scenes with test_poses.csv found")
    errors = []
    entries = []
    for scene, csv_path in sorted(scenes.items()):
        rdir = os.path.join(args.renders_root, scene)
        if not os.path.isdir(rdir):
            errors.append(f"{scene}: renders dir missing ({rdir})")
            continue
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            name = row["image_name"]
            if args.ext:
                name = os.path.splitext(name)[0] + "." + args.ext.lstrip(".")
            fpath = os.path.join(rdir, name)
            if not os.path.exists(fpath):
                errors.append(f"{scene}: missing {name}")
                continue
            w, h = Image.open(fpath).size
            if (w, h) != (int(row["width"]), int(row["height"])):
                errors.append(f"{scene}/{name}: size {w}x{h} != csv {row['width']}x{row['height']}")
                continue
            entries.append((fpath, f"{scene}/{name}"))

    if errors:
        print(f"\n{len(errors)} problems:")
        for e in errors[:30]:
            print("  " + e)
        print("Submission NOT written.")
        return 1

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as zf:
        for fpath, arcname in entries:
            zf.write(fpath, arcname)
    print(f"Wrote {args.out}: {len(entries)} images, {len(scenes)} scenes, "
          f"{os.path.getsize(args.out)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
