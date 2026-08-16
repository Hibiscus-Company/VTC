#
# Pixel-mean render ensemble (exp18 lever, audit-round-3 hardened).
#
# Averages the renders of N independently trained models of the same scene.
# Audit-confirmed rules baked in:
#   - accumulate in float32, single "*255 + 0.5" round at the end
#     (uint8 integer division costs -0.009; measured)
#   - average in sRGB (linear-RGB averaging measured -0.012)
#   - mean, not median (median measured -0.08)
#   - prefer lossless PNG archives as sources over q100 JPEGs (~+0.005)
#   - only ensemble members within ~0.15 pts of the best single model
#     (a -0.97 member measured -0.09 on the ensemble) -- caller's job
#
# Usage:
#   python ensemble_renders.py --dirs A/test_poses_renders_png B/..._png C/..._png \
#       --out OUT_DIR [--png_dir OUT_PNG_DIR] [--names_from test_poses.csv]
#       [--jpeg_quality 100] [--jpeg_subsampling 0]
#
# Output filenames follow --names_from (csv image_name column, e.g. *.JPG);
# without it, the stems of the first source dir with .JPG extension.
#
import os
import csv
import glob
import argparse
import numpy as np
from PIL import Image


def load_stems(d):
    # reverse preference order: later globs overwrite, so PNG wins over JPEG
    # when a dir holds both (audit round 4: concatenation order had this inverted)
    files = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.JPG"))
                   + glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.PNG")))
    return {os.path.splitext(os.path.basename(f))[0]: f for f in files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True, help="member render dirs (PNG archives preferred)")
    p.add_argument("--out", required=True, help="output dir for ensembled JPEGs")
    p.add_argument("--png_dir", default=None, help="also write lossless PNG copies here")
    p.add_argument("--names_from", default=None, help="test_poses.csv to take output filenames from")
    p.add_argument("--jpeg_quality", type=int, default=100)
    p.add_argument("--jpeg_subsampling", type=int, default=0, help="0=4:4:4, 2=4:2:0")
    p.add_argument("--weights", type=float, nargs="+", default=None,
                   help="per-member weights, same order as --dirs (normalized "
                        "internally; default uniform). Audit round-9: only "
                        "meaningful across model FAMILIES with complementary "
                        "error profiles; equal-quality same-family weighting "
                        "measured worthless (round 3)")
    p.add_argument("--masks", nargs="+", default=None,
                   help="per-member .npy validity mask (or 'none'), same order as "
                        "--dirs. Weight is zeroed where a member was never supervised "
                        "and the remaining members are renormalized per pixel. See "
                        "gsplat_track/fov_mask.py")
    args = p.parse_args()

    members = [load_stems(d) for d in args.dirs]
    stems = set(members[0])
    for d, m in zip(args.dirs, members):
        assert set(m) == stems, f"{d} file stems differ from {args.dirs[0]}"
        n_png = sum(f.lower().endswith(".png") for f in m.values())
        if n_png < len(m):
            print(f"WARNING: {d}: {len(m) - n_png}/{len(m)} sources are JPEG, not lossless PNG")

    out_names = {s: s + ".JPG" for s in stems}
    if args.names_from:
        with open(args.names_from, newline="") as f:
            rows = list(csv.DictReader(f))
        csv_names = {os.path.splitext(r["image_name"])[0]: r["image_name"] for r in rows}
        assert set(csv_names) == stems, "csv image stems differ from render stems"
        out_names = csv_names

    os.makedirs(args.out, exist_ok=True)
    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)

    w = np.ones(len(members), dtype=np.float64)
    if args.weights is not None:
        assert len(args.weights) == len(members), "--weights count != --dirs count"
        w = np.asarray(args.weights, dtype=np.float64)
    w = w / w.sum()

    # per-member spatial validity (P6). On HNI0131/HNI0265 (k1=-0.115) the same-K
    # undistortion crops the outer ring, so the FastGS members were NEVER supervised
    # on 11.9% of the frame -- yet they carry 0.4 of the weight there. Zero them
    # outside their supervised region and renormalize per pixel, so the UT members
    # (supervised across the whole frame, in distorted space) carry that ring alone.
    masks = [None] * len(members)
    if args.masks is not None:
        assert len(args.masks) == len(members), "--masks count != --dirs count"
        for i, mp in enumerate(args.masks):
            if mp and mp.lower() != "none":
                masks[i] = np.load(mp).astype(np.float32)
                print(f"  mask {os.path.basename(mp)} on {args.dirs[i]}: "
                      f"{100 * (masks[i] < 0.5).mean():.2f}% suppressed")

    for s in sorted(stems):
        acc, wsum = None, None
        for wi, m, mk in zip(w, members, masks):
            im = Image.open(m[s])
            if im.mode == "P":
                print(f"WARNING: {m[s]} is palette-quantized (lossy source, not the --png_dir archive)")
            a = np.asarray(im.convert("RGB"), dtype=np.float32)
            ww = np.float32(wi) if mk is None else (np.float32(wi) * mk)[..., None]
            acc = a * ww if acc is None else acc + a * ww
            wsum = np.broadcast_to(ww, a.shape).copy() if wsum is None \
                else wsum + np.broadcast_to(ww, a.shape)
        # renormalize: with a mask the weights no longer sum to 1 at every pixel
        mean = acc / np.maximum(wsum, 1e-6)
        assert wsum.min() > 1e-3, ("some pixel has no valid member -- at least one "
                                   "unmasked member must cover the whole frame")
        img = Image.fromarray(np.clip(mean + 0.5, 0, 255).astype(np.uint8))
        if args.png_dir:
            img.save(os.path.join(args.png_dir, s + ".png"))
        img.save(os.path.join(args.out, out_names[s]),
                 quality=args.jpeg_quality, subsampling=args.jpeg_subsampling, optimize=True)
    print(f"Ensembled {len(stems)} images from {len(members)} members -> {args.out}")


if __name__ == "__main__":
    main()
