#!/usr/bin/env python
"""Carve a scene's TRAIN set into train-sub / eval, where eval MIMICS the hidden test.

The round-2 video scenes (bonsai, chair) hand us a test set that is a set of SINGLE
HELD-OUT GRID POINTS: the capture is sampled at a fixed video stride (bonsai 10, chair 5),
and every test frame is one missing grid point flanked by present train frames one stride
away (verified: 28/28 bonsai, 58/58 chair strictly interleaved, nearest-train index gap =
one stride, pose gap ~0.05 / ~4 deg). A model's job on test is exactly "fill a one-stride
hole." So a HONEST eval set is the same shape: punch isolated holes in the train grid, each
flanked by remaining train frames one stride away. Random holdout would be dishonest --
adjacent removals make two-stride holes (harder than test), and leaving near-duplicate
neighbours makes it trivially easy.

This is model-SELECTION only: pick the recipe that fills eval holes best, then RETRAIN the
final model on the FULL train set with that recipe before rendering test poses.

RULE 10: eval GT is TRAIN photos (the held-out train frames). Never test GT. The eval poses
come from images.bin (they are train frames, in the BA). No test pixel is ever read.

Outputs under <out>/:
  train_sub/images/  symlinks to train frames MINUS the eval holes
  train_sub/sparse/  symlink to the scene's sparse/0
  eval_poses.csv     render-CSV (image_name,q,t,fx,fy,cx,cy,w,h) for the eval holes
  eval_gt/           symlinks to the eval holes' train photos (scoring GT)
  split.json         provenance: strides, counts, the exact eval frame list
"""
import argparse, csv, json, os, struct
import numpy as np


def qvec(name_bin, want):
    """read images.bin -> {name: (qwxyz, txyz)} for names in `want` (or all if None)"""
    out = {}
    with open(name_bin, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<i", f.read(4))
            q = struct.unpack("<4d", f.read(32))
            t = struct.unpack("<3d", f.read(24))
            struct.unpack("<i", f.read(4))
            nm = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                nm += c
            npt = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * npt, 1)
            nm = nm.decode()
            if want is None or nm in want:
                out[nm] = (q, t)
    return out


def read_cam(cameras_bin):
    MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
              3: ("RADIAL", 5)}
    with open(cameras_bin, "rb") as f:
        struct.unpack("<Q", f.read(8))
        cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
        model, npar = MODELS[mid]
        par = struct.unpack("<" + "d" * npar, f.read(8 * npar))
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fx = fy = par[0]; cx, cy = par[1], par[2]
    else:
        fx, fy, cx, cy = par[0], par[1], par[2], par[3]
    return fx, fy, cx, cy, w, h


def fidx(name):
    b = os.path.splitext(name)[0]
    if b.startswith("frame_"):
        try:
            return int(b.split("_")[1])
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="scene dir with train/ and test/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_eval", type=int, default=0,
                    help="0 = match the test-set size")
    args = ap.parse_args()

    train_dir = os.path.join(args.scene, "train")
    sp = os.path.join(train_dir, "sparse", "0")
    imdir = os.path.join(train_dir, "images")
    train = sorted(os.listdir(imdir))
    with open(os.path.join(args.scene, "test", "test_poses.csv")) as f:
        n_test = sum(1 for _ in csv.DictReader(f))
    n_eval = args.n_eval or n_test

    # CAREFUL SPLIT (one rule, both regimes): eval = ISOLATED, evenly-spaced frames along
    # the capture sequence. The hidden test set is itself an interleaved holdout of the same
    # capture, so mirroring that structure reproduces its novel-view difficulty (validated on
    # public HCM0181: isolated-every-k matches test gap 0.121 vs 0.116, p75 0.176 vs 0.168;
    # random has too-fat a tail, arcs/widening overshoot). Never remove adjacent frames.
    idxs = {n: fidx(n) for n in train}
    is_video = all(v is not None for v in idxs.values())
    if is_video:
        # video: capture index is the video frame number; stride = grid step
        order = sorted(train, key=lambda n: idxs[n])
        seq = [idxs[n] for n in order]
        stride = int(np.median(np.diff(seq)))
        # a frame is "isolated-able" iff both one-stride neighbours are present train frames
        present = set(seq)
        cand = [n for n in order if (idxs[n] - stride in present) and (idxs[n] + stride in present)]
        min_sep = 2 * stride
        sep = lambda a, b: idxs[a] - idxs[b]
    else:
        # drone: no frame index; capture order = filename sort (DJI timestamps). Use ordinal
        # position as the sequence coordinate; every frame is flanked, so all are candidates.
        order = sorted(train)
        pos = {n: i for i, n in enumerate(order)}
        stride = 1
        cand = order
        min_sep = 2                       # never remove two adjacent captures
        sep = lambda a, b: pos[a] - pos[b]

    if len(cand) < n_eval:
        raise SystemExit(f"only {len(cand)} isolated-hole candidates, need {n_eval}")
    pick, last = [], None
    step = len(cand) / n_eval
    i = 0.0
    while len(pick) < n_eval and int(i) < len(cand):
        c = cand[int(i)]
        if last is None or sep(c, last) >= min_sep:
            pick.append(c); last = c
        i += step
    # audit r14 bug 2: min_sep rejections can under-fill silently; the len(cand) guard alone
    # is insufficient (real requirement ~2x). Fail loudly instead of shipping a thin eval set.
    assert len(pick) == n_eval, (
        f"eval under-fill: picked {len(pick)}/{n_eval} (cand {len(cand)}, min_sep {min_sep}) "
        "-- scene too dense for isolated holes at this count; lower --n_eval")
    pick = set(pick)
    eval_names = [n for n in train if n in pick]
    sub_names = [n for n in train if n not in pick]

    os.makedirs(os.path.join(args.out, "train_sub", "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "eval_gt"), exist_ok=True)
    # symlink train-sub images + sparse
    for n in sub_names:
        d = os.path.join(args.out, "train_sub", "images", n)
        if not os.path.lexists(d):
            os.symlink(os.path.abspath(os.path.join(imdir, n)), d)
    sld = os.path.join(args.out, "train_sub", "sparse")
    if not os.path.lexists(sld):
        os.symlink(os.path.abspath(os.path.join(train_dir, "sparse")), sld)
    for n in eval_names:
        d = os.path.join(args.out, "eval_gt", n)
        if not os.path.lexists(d):
            os.symlink(os.path.abspath(os.path.join(imdir, n)), d)

    # eval_poses.csv in render_gsplat format
    fx, fy, cx, cy, w, h = read_cam(os.path.join(sp, "cameras.bin"))
    poses = qvec(os.path.join(sp, "images.bin"), set(eval_names))
    with open(os.path.join(args.out, "eval_poses.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
                     "fx", "fy", "cx", "cy", "width", "height"])
        for n in eval_names:
            q, t = poses[n]
            wr.writerow([n, *q, *t, fx, fy, cx, cy, w, h])

    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump({"scene": os.path.abspath(args.scene), "stride": stride,
                   "n_train_full": len(train), "n_train_sub": len(sub_names),
                   "n_eval": len(eval_names), "eval_frames": sorted(pick),
                   "note": "eval = isolated grid holes mimicking held-out test; "
                           "select recipe here, retrain on FULL train for test render"},
                  f, indent=2)
    print(f"{os.path.basename(args.scene)}: stride {stride}  full {len(train)} -> "
          f"sub {len(sub_names)} + eval {len(eval_names)} isolated holes "
          f"(cand pool {len(cand)})  -> {args.out}")


if __name__ == "__main__":
    main()
