#!/usr/bin/env python
"""D11: the DISTRIBUTION of test-pose proximity to train poses (not the mean).

D3 measured the MEAN parallax to the nearest train view (11.8 deg) and closed photo-reuse.
But the score averages PSNR PER IMAGE, so the mean is the wrong statistic: if a TAIL of
test poses sits very close to a train pose, those specific images could be handled
completely differently, and a few very-high-PSNR images lift a per-image mean.

Poses only -- no test GT, no images. Fully legal.

Reports, per scene and pooled:
  - angular distance to the nearest train view (deg)
  - camera-centre distance to the nearest train view, normalized by scene scale
  - how many test views fall under increasingly tight thresholds
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "..", "..",
                                "mnt", "c", "Users", "BKAI", "an_plaza2", "FastGS"))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat


def cams_from_sparse(sparse, img_dir):
    out = []
    for v in read_extrinsics_binary(os.path.join(sparse, "images.bin")).values():
        if not os.path.exists(os.path.join(img_dir, v.name)):
            continue                      # train images only
        R = qvec2rotmat(v.qvec)
        C = -R.T @ v.tvec                 # camera centre in world
        out.append((R, C))
    return out


def cams_from_csv(path):
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            q = np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])])
            t = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])
            R = qvec2rotmat(q)
            out.append((R, -R.T @ t, r["image_name"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    scenes = args.scenes or sorted(d for d in os.listdir(root)
                                   if os.path.isdir(os.path.join(root, d)))
    all_ang, all_dist = [], []
    print(f"{'scene':10s} {'n':>4s}  {'ang: min':>8s} {'p10':>6s} {'med':>6s} {'max':>6s}"
          f"   {'dist/scale: min':>15s} {'med':>7s}")
    for s in scenes:
        sp = os.path.join(root, s, "train", "sparse", "0")
        imd = os.path.join(root, s, "train", "images")
        csvp = os.path.join(root, s, "test", "test_poses.csv")
        if not (os.path.isdir(sp) and os.path.exists(csvp)):
            continue
        tr = cams_from_sparse(sp, imd)
        te = cams_from_csv(csvp)
        if not tr or not te:
            continue
        trC = np.stack([c for _, c in tr])
        scale = float(np.linalg.norm(trC - trC.mean(0), axis=1).mean())

        angs, dists = [], []
        for R, C, _n in te:
            # angular distance between viewing directions (camera +z in world)
            z_t = R.T @ np.array([0.0, 0.0, 1.0])
            best_a = 180.0
            for Rr, _Cr in tr:
                z_r = Rr.T @ np.array([0.0, 0.0, 1.0])
                cos = float(np.clip(np.dot(z_t, z_r), -1, 1))
                best_a = min(best_a, np.degrees(np.arccos(cos)))
            angs.append(best_a)
            dists.append(float(np.linalg.norm(trC - C, axis=1).min()) / max(scale, 1e-9))
        angs = np.array(angs); dists = np.array(dists)
        all_ang.append(angs); all_dist.append(dists)
        print(f"{s:10s} {len(te):4d}  {angs.min():8.2f} {np.percentile(angs,10):6.2f} "
              f"{np.median(angs):6.2f} {angs.max():6.2f}   "
              f"{dists.min():15.3f} {np.median(dists):7.3f}")

    A = np.concatenate(all_ang); D = np.concatenate(all_dist)
    print(f"\npooled: {len(A)} test views")
    print(f"  nearest-train ANGLE (deg): min {A.min():.2f}  p1 {np.percentile(A,1):.2f}  "
          f"p5 {np.percentile(A,5):.2f}  p10 {np.percentile(A,10):.2f}  "
          f"median {np.median(A):.2f}  max {A.max():.2f}")
    print("\n  how many test views are CLOSE to a train view?")
    for thr in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        n = int((A < thr).sum())
        print(f"    angle < {thr:4.1f} deg : {n:4d} / {len(A)}  ({100*n/len(A):5.1f}%)")
    print("\n  => a TAIL of near-duplicate poses would be worth exploiting per-image.")
    print("     NO tail => photo-reuse stays dead and the mean was the right statistic.")


if __name__ == "__main__":
    main()
