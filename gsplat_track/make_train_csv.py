#!/usr/bin/env python
"""Emit a train_poses.csv in the exact format of the competition's test_poses.csv.

Needed to render the FastGS gate members at TRAIN poses -- the only way to validate
the P6 FoV mask, since no PUBLIC scene has negative k1 (0% masked) and so the ring
defect cannot be reproduced on public data at all. Train photos are the only ground
truth available for it.
"""
import argparse, csv, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary

COLS = ["image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
        "fx", "fy", "cx", "cy", "width", "height"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="scene train/ dir (has sparse/0, images/)")
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    sparse = os.path.join(args.source, "sparse", "0")
    cam = list(read_intrinsics_binary(os.path.join(sparse, "cameras.bin")).values())[0]
    p = list(cam.params)
    f, cx, cy = float(p[0]), float(p[1]), float(p[2])
    imgs = read_extrinsics_binary(os.path.join(sparse, "images.bin"))

    img_dir = os.path.join(args.source, args.images)
    rows = []
    for im in imgs.values():
        # the sparse holds train + test + dropped poses; keep only those with a file
        if not os.path.exists(os.path.join(img_dir, im.name)):
            continue
        q, t = im.qvec, im.tvec
        rows.append({"image_name": im.name,
                     "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
                     "tx": t[0], "ty": t[1], "tz": t[2],
                     "fx": f, "fy": f, "cx": cx, "cy": cy,
                     "width": cam.width, "height": cam.height})
    rows.sort(key=lambda r: r["image_name"])
    rows = rows[:: args.stride]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=COLS)
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"wrote {len(rows)} train poses -> {args.out}")


if __name__ == "__main__":
    main()
