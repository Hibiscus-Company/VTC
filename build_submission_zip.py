#
# Build the competition submission zip from lossless PNG archives
# (audit round 3: the packaged JPEGs must be SINGLE-GENERATION -- encoded once,
# from PNG, at final quality; never re-encode an existing JPEG).
#
# Walks a quality ladder until the zip fits --max_mb: encodes every scene at
# qualities[0]; if too big, steps the largest scenes down one rung at a time.
# Validates the file list against each scene's test_poses.csv (exact names,
# including .JPG case) and runs a CRC self-test before reporting.
#
# Usage:
#   python build_submission_zip.py \
#       --scene_dirs "SCENE1=/path/to/renders_png" "SCENE2=..." \
#       --data_root ~/data/phase1/private_set1 \
#       --out /mnt/d/avv/submissions/sub_XXX.zip \
#       [--qualities 98 97 96] [--subsampling 2] [--max_mb 350]
#
import os
import io
import csv
import glob
import argparse
import zipfile
from PIL import Image


def encode_scene(png_dir, names, sizes, quality, subsampling, keep_rgb=False):
    blobs = {}
    for name in names:
        stem = os.path.splitext(name)[0]
        src = os.path.join(png_dir, stem + ".png")
        assert os.path.exists(src), f"missing PNG source: {src}"
        im = Image.open(src)
        # guard against a stale/wrong-run png_dir (e.g. old supersample output):
        # wrong-size images would pass every name/CRC check but score 0
        assert im.size == sizes[name], f"{src}: {im.size} != csv {sizes[name]}"
        buf = io.BytesIO()
        # progressive: identical pixels at the same quality, ~5% smaller
        # (measured 343.7->327.0MB on the 8-scene ensemble at q98ss2) --
        # that headroom is what lets small scenes ride at q100/q99
        #
        # keep_rgb (audit 26/07): at q100 the quant table is all-1s, so the residual JPEG loss is
        # the RGB->YCbCr->RGB roundtrip plus DCT rounding. keep_rgb=True stores JPEG in RGB and
        # removes the colour transform entirely. On bonsai that recovered +1.0088 of a measured
        # -1.0802 tax (LPIPS .2695 -> .2461), landing within 0.071 of lossless PNG. It is a REAL
        # jpeg -- no format spoofing. Costs bytes, so it is worth it only on high-frequency
        # content (the video scenes); towers measured a ~0 tax and stay on the cheap path.
        kw = dict(quality=quality, subsampling=subsampling, optimize=True, progressive=True)
        if keep_rgb:
            kw["keep_rgb"] = True
        im.convert("RGB").save(buf, "JPEG", **kw)
        blobs[name] = buf.getvalue()
    return blobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dirs", nargs="+", required=True,
                   help="SCENE=png_dir pairs; SCENE must match a <data_root>/<SCENE>")
    p.add_argument("--data_root", required=True, help="dir holding <scene>/test/test_poses.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--qualities", type=int, nargs="+", default=[100, 99, 98, 97, 96, 95])
    p.add_argument("--subsampling", type=int, default=2, help="0=4:4:4, 2=4:2:0")
    p.add_argument("--max_mb", type=float, default=350.0)
    p.add_argument("--hq_scenes", nargs="*", default=[],
                   help="scenes to encode with subsampling=0 + keep_rgb=True (no RGB->YCbCr "
                        "roundtrip). Worth ~+1.0 on high-frequency video scenes; costs bytes, "
                        "so leave the towers off it.")
    p.add_argument("--hq_min_quality", type=int, default=96,
                   help="the size ladder may not push an --hq_scenes scene below this quality")
    p.add_argument("--hq_quality", type=int, default=None,
                   help="PIN --hq_scenes to exactly this quality; they are then excluded from both "
                        "the step-down and the reclaim passes. Needed because both passes assume "
                        "higher quality == better, which is FALSE for keep_rgb video: measured "
                        "bonsai q98 keep_rgb 71.8115 BEATS q100 keep_rgb 71.7935 while being 26% "
                        "smaller (mild quantization denoises in a way LPIPS rewards).")
    args = p.parse_args()

    scenes = {}
    for sd in args.scene_dirs:
        scene, png_dir = sd.split("=", 1)
        csv_path = os.path.join(os.path.expanduser(args.data_root), scene, "test", "test_poses.csv")
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        names = [r["image_name"] for r in rows]
        assert len(names) == len(set(names)), f"{csv_path}: duplicate image_name"
        sizes = {r["image_name"]: (int(r["width"]), int(r["height"])) for r in rows}
        hq = scene in args.hq_scenes
        pinned = hq and args.hq_quality is not None
        scenes[scene] = {"png_dir": png_dir, "names": names, "sizes": sizes,
                         "q": args.hq_quality if pinned else args.qualities[0],
                         "hq": hq, "pinned": pinned,
                         "ss": 0 if hq else args.subsampling}
    for s in args.hq_scenes:
        assert s in scenes, f"--hq_scenes names {s}, which is not in --scene_dirs"

    def enc(s, q):
        i = scenes[s]
        return encode_scene(i["png_dir"], i["names"], i["sizes"], q, i["ss"], keep_rgb=i["hq"])

    # first pass at top quality
    for s, info in scenes.items():
        info["blobs"] = enc(s, info["q"])

    def total_mb():
        return sum(len(b) for i in scenes.values() for b in i["blobs"].values()) / 1e6

    # ladder: step the currently-largest scene down one rung until it fits
    def can_step_down(s):
        i = scenes[s]
        if i["pinned"]:
            return False
        qi = args.qualities.index(i["q"])
        if qi >= len(args.qualities) - 1:
            return False
        # an --hq_scenes scene is high-quality *on purpose* (it is where the LPIPS points are);
        # never let the byte ladder quietly undo that below the floor
        return not (i["hq"] and args.qualities[qi + 1] < args.hq_min_quality)

    while total_mb() > args.max_mb:
        candidates = [s for s in scenes if can_step_down(s)]
        assert candidates, (
            f"cannot fit {args.max_mb}MB: every scene is at its floor "
            f"(hq scenes are pinned at >= q{args.hq_min_quality})")
        big = max(candidates, key=lambda s: sum(len(b) for b in scenes[s]["blobs"].values()))
        scenes[big]["q"] = args.qualities[args.qualities.index(scenes[big]["q"]) + 1]
        print(f"{total_mb():.1f}MB > {args.max_mb}MB, re-encoding {big} at q{scenes[big]['q']}")
        scenes[big]["blobs"] = enc(big, scenes[big]["q"])

    # RECLAIM: the descent above stops at the first rung that FITS, so the final step
    # usually overshoots and throws the leftover headroom away. R8 landed at 342.7MB of
    # a 350MB budget -- 7.3MB (2%) of quality discarded, after it had already knocked an
    # untouched scene from q99 to q98 to get there. Step scenes back UP while they fit.
    # Greedy on smallest-cost-first, so the cheapest upgrades land before the budget runs out.
    improved = True
    while improved:
        improved = False
        for s in sorted(scenes, key=lambda s: sum(len(b) for b in scenes[s]["blobs"].values())):
            if scenes[s]["pinned"]:
                continue  # pinned at its MEASURED optimum; "up" would be worse, not better
            qi = args.qualities.index(scenes[s]["q"])
            if qi == 0:
                continue
            up_q = args.qualities[qi - 1]
            trial = enc(s, up_q)
            cur_b = sum(len(b) for b in scenes[s]["blobs"].values())
            new_b = sum(len(b) for b in trial.values())
            # zip overhead is ~136 B/entry (ZIP_STORED), so a FIXED 0.5MB reserve stops
            # covering it past ~3700 files -- and a bigger release could exceed that.
            # Scale the margin with the file count (audit r11).
            n_files = sum(len(i["names"]) for i in scenes.values())
            margin = max(0.5, 0.0002 * n_files)
            if total_mb() + (new_b - cur_b) / 1e6 <= args.max_mb - margin:
                scenes[s]["q"], scenes[s]["blobs"] = up_q, trial
                print(f"reclaim: {s} back up to q{up_q}  (total {total_mb():.1f}MB)")
                improved = True

    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for s in sorted(scenes):
            for name in scenes[s]["names"]:
                z.writestr(f"{s}/{name}", scenes[s]["blobs"][name])

    # verify: CRC + exact arcname set vs CSVs
    with zipfile.ZipFile(args.out) as z:
        assert z.testzip() is None, "CRC check failed"
        got = set(z.namelist())
    exp = {f"{s}/{n}" for s, i in scenes.items() for n in i["names"]}
    assert got == exp, f"arcname mismatch: missing={exp - got} extra={got - exp}"

    size_mb = os.path.getsize(args.out) / 1e6
    # blob budget ignores ~150B/entry of zip overhead; catch the edge here
    assert size_mb <= args.max_mb, f"final zip {size_mb:.1f}MB exceeds {args.max_mb}MB"
    print(f"OK: {args.out}  {size_mb:.1f}MB  {len(exp)} files")
    for s in sorted(scenes):
        i = scenes[s]
        print(f"  {s}: q{i['q']} ss{i['ss']}{' keep_rgb' if i['hq'] else ''}")


if __name__ == "__main__":
    main()
