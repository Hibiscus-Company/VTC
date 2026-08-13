"""A3 follow-up: is the first-quarter failure a camera-coverage / baseline effect?
Uses ONLY train_sub poses (legal) + the provided eval_poses.csv."""
import sys, os, json, csv
import numpy as np
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from scipy import stats
from scene.colmap_loader import read_extrinsics_binary, read_points3D_binary

OUT = "/mnt/d/avv/r42_bonsai78/a3_diag"
SP = "/mnt/d/avv/evalsplit/bonsai/train_sub/sparse_filtered/0"
ex = read_extrinsics_binary(f"{SP}/images.bin")


def qvec2R(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])


train = {}
for k, im in ex.items():
    R = qvec2R(im.qvec); t = np.array(im.tvec)
    C = -R.T @ t
    fwd = R.T @ np.array([0, 0, 1.0])
    train[im.name] = (C, fwd, int(os.path.splitext(im.name)[0].split("_")[-1]))
print(f"train_sub cameras: {len(train)}")

ev = {}
with open("/mnt/d/avv/evalsplit/bonsai/eval_poses.csv") as f:
    for r in csv.DictReader(f):
        q = np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])])
        R = qvec2R(q); t = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])
        ev[r["image_name"]] = (-R.T @ t, R.T @ np.array([0, 0, 1.0]),
                               int(os.path.splitext(r["image_name"])[0].split("_")[-1]))

P, _rgb, _err = read_points3D_binary(f"{SP}/points3D.bin")
P = np.asarray(P)
print(f"sparse points: {len(P)}")

rows = sorted(json.load(open(f"{OUT}/bonsai_sr01_perframe.json")), key=lambda r: r["frame"])
TC = np.array([v[0] for v in train.values()])
TF = np.array([v[1] for v in train.values()])
Tn = np.array([v[2] for v in train.values()])
scale = np.median(np.linalg.norm(TC - TC.mean(0), axis=0))

out = []
for r in rows:
    nm = r["stem"] + ".jpg"
    C, F, n = ev[nm]
    d = np.linalg.norm(TC - C, axis=1)
    o = np.argsort(d)
    ang = np.degrees(np.arccos(np.clip(TF @ F, -1, 1)))
    # local trajectory speed: distance to the two temporal neighbours (+-10 frames)
    nb = [np.linalg.norm(TC[i] - C) for i in range(len(Tn)) if abs(Tn[i] - n) <= 10]
    # scene depth: median distance from camera to sparse points in front of it
    v = P - C
    z = v @ F
    fr = z[(z > 0)]
    med_depth = float(np.median(fr)) if len(fr) else np.nan
    out.append(dict(stem=r["stem"], frame=n, score=r["score"], lpips=r["lpips"],
                    psnr=r["psnr"], ssim=r["ssim"],
                    d_nearest=float(d[o[0]]), d_2nd=float(d[o[1]]),
                    d_mean5=float(d[o[:5]].mean()),
                    ang_nearest=float(ang[o[0]]),
                    n_within_0p2=int((d < 0.2).sum()), n_within_0p5=int((d < 0.5).sum()),
                    temporal_baseline=float(np.mean(nb)) if nb else np.nan,
                    med_scene_depth=med_depth,
                    baseline_over_depth=(float(np.mean(nb)) / med_depth) if nb and med_depth else np.nan))
    print(f"  {r['stem']} sc{r['score']:6.2f} dnear {out[-1]['d_nearest']:.4f} "
          f"tempbase {out[-1]['temporal_baseline']:.4f} depth {med_depth:.3f} "
          f"b/d {out[-1]['baseline_over_depth']:.4f} n<0.2 {out[-1]['n_within_0p2']:2d} "
          f"angnear {out[-1]['ang_nearest']:.2f}deg")

sc = np.array([o["score"] for o in out])
print("\n=== geometry vs score (spearman) ===")
for k in ["d_nearest", "d_2nd", "d_mean5", "ang_nearest", "n_within_0p2", "n_within_0p5",
          "temporal_baseline", "med_scene_depth", "baseline_over_depth"]:
    v = np.array([o[k] for o in out], float)
    if np.all(np.isfinite(v)) and v.std() > 0:
        rr, pp = stats.spearmanr(v, sc)
        print(f"  {k:20s}: {rr:+.3f} (p={pp:.4f})   first8 {v[:8].mean():.4f} last20 {v[8:].mean():.4f}")
with open(f"{OUT}/bonsai_geom.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader()
    [w.writerow(o) for o in out]
json.dump(out, open(f"{OUT}/bonsai_geom.json", "w"), indent=1)
print(f"\nwrote {OUT}/bonsai_geom.csv")
