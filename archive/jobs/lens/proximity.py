#!/usr/bin/env python
"""HOW FAR IS EACH TEST POSE FROM THE NEAREST TRAIN POSE, PER SCENE?

Diagnostic D3 killed photo-reuse / IBR on the drone towers with one number: the nearest train
pose is 11.8 deg of view angle away, so "paste the nearest train photo" scored 9.4 dB against
our render's 24.5. That verdict was measured on a TOWER and then applied to the whole set.

The two video scenes are a different capture entirely -- a handheld walk sampled every 5th
(chair) or 10th (bonsai) video frame, then split into train/test. If a test frame's nearest
train frame is a fraction of a degree away rather than 11.8 deg, the reasoning that killed IBR
does not carry over to them, and they are 2/7 of the score with no lens field and no
photo-reuse ever attempted.

Rule 10: reads train COLMAP poses and the GIVEN test_poses.csv. No test imagery.
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat

DATA = "/mnt/d/avv/data/phase1"


def centres(qwxyz, t):
    R = qvec2rotmat(np.array(qwxyz))
    return -R.T @ np.array(t), R[2]        # camera centre, and the +z (viewing) axis in world


def scene_poses(root):
    sp = os.path.join(root, "train", "sparse", "0")
    ex = read_extrinsics_binary(os.path.join(sp, "images.bin"))
    # images.bin carries MORE entries than there are train photos (e.g. 350 vs 240) -- the test
    # poses are registered in the same reconstruction. Keep only poses whose photo is on disk,
    # otherwise every test pose finds ITSELF and the nearest distance is exactly 0.
    imgd = os.path.join(root, "train", "images")
    have = {os.path.splitext(f)[0] for f in os.listdir(imgd)}
    tr = {}
    for im in ex.values():
        stem = os.path.splitext(im.name)[0]
        if stem not in have:
            continue
        c, z = centres(im.qvec, im.tvec)
        tr[stem] = (c, z)
    te = {}
    with open(os.path.join(root, "test", "test_poses.csv")) as fh:
        for r in csv.DictReader(fh):
            c, z = centres([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])],
                           [float(r["tx"]), float(r["ty"]), float(r["tz"])])
            te[os.path.splitext(r["image_name"])[0]] = (c, z)
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="private_set2/HCM0421,private_set2/HCM0674,"
                                        "private_set2/chair,private_set2/bonsai,"
                                        "public_set/HCM0181")
    args = ap.parse_args()
    print(f"{'scene':>26} {'n_tr':>5} {'n_te':>5} {'radius':>8} "
          f"{'nearest dist':>13} {'as % radius':>12} {'view angle deg':>15}")
    for s in args.scenes.split(","):
        root = os.path.join(DATA, s)
        try:
            tr, te = scene_poses(root)
        except Exception as e:
            print(f"{s:>26}  ERROR {e}")
            continue
        TC = np.stack([v[0] for v in tr.values()])
        TZ = np.stack([v[1] for v in tr.values()])
        rad = float(np.linalg.norm(TC - TC.mean(0), axis=1).mean())
        d_all, a_all = [], []
        for name, (c, z) in te.items():
            d = np.linalg.norm(TC - c, axis=1)
            j = int(np.argmin(d))
            d_all.append(float(d[j]))
            # view-angle difference to the nearest-in-position train camera
            cosang = float(np.clip(TZ[j] @ z, -1, 1))
            a_all.append(np.degrees(np.arccos(cosang)))
        d_all, a_all = np.array(d_all), np.array(a_all)
        print(f"{s:>26} {len(tr):>5} {len(te):>5} {rad:8.3f} "
              f"{d_all.mean():13.4f} {100*d_all.mean()/rad:11.2f}% {a_all.mean():15.2f}")
        print(f"{'':>26} {'':>5} {'':>5} {'':>8} "
              f"median {np.median(d_all):.4f}  p90 {np.percentile(d_all,90):.4f}"
              f"   angle median {np.median(a_all):.2f}  p90 {np.percentile(a_all,90):.2f}")


if __name__ == "__main__":
    main()
