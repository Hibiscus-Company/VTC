"""A3 analysis: (a) per-frame distribution + worst5 lift, (b) sharpness correlations,
(c) spatial concentration. Reads the .npy/.json produced by a3_perframe.py."""
import json, sys, os
import numpy as np
from scipy import stats

OUT = "/mnt/d/avv/r42_bonsai78/a3_diag"
TAG = sys.argv[1] if len(sys.argv) > 1 else "bonsai_sr01"
rows = json.load(open(f"{OUT}/{TAG}_perframe.json"))
rows.sort(key=lambda r: r["frame"])
sc = np.array([r["score"] for r in rows])
lp = np.array([r["lpips"] for r in rows])
ps = np.array([r["psnr"] for r in rows])
ss = np.array([r["ssim"] for r in rows])
lv = np.array([r["gt_lapvar"] for r in rows])
rlv = np.array([r["rn_lapvar"] for r in rows])
st = [r["stem"] for r in rows]
n = len(rows)

print("=== (a) PER-FRAME DISTRIBUTION ===")
print(f"n={n}  mean score {sc.mean():.4f}  (PSNR {ps.mean():.4f} SSIM {ss.mean():.4f} LPIPS {lp.mean():.4f})")
for nm, v in [("score", sc), ("psnr", ps), ("ssim", ss), ("lpips", lp)]:
    q = np.percentile(v, [0, 25, 50, 75, 100])
    print(f"{nm:6s} min {q[0]:.4f} p25 {q[1]:.4f} med {q[2]:.4f} p75 {q[3]:.4f} max {q[4]:.4f} "
          f"sd {v.std(ddof=1):.4f}")
o = np.argsort(sc)
print("\nWORST 5:")
for i in o[:5]:
    print(f"  {st[i]} score {sc[i]:.3f} P {ps[i]:.3f} S {ss[i]:.4f} L {lp[i]:.4f} gt_lapvar {lv[i]*1e4:.1f}e-4")
print("BEST 5:")
for i in o[-5:][::-1]:
    print(f"  {st[i]} score {sc[i]:.3f} P {ps[i]:.3f} S {ss[i]:.4f} L {lp[i]:.4f} gt_lapvar {lv[i]*1e4:.1f}e-4")

med = float(np.median(sc))
gain5 = (5 * med - sc[o[:5]].sum()) / n
deficit = 78.0 - sc.mean()
print(f"\nmedian per-frame score {med:.4f}; deficit to 78 = {deficit:.4f}")
print(f"worst-5 -> median lifts the mean by {gain5:.4f} = {100*gain5/deficit:.1f}% of the deficit")
for k in (3, 5, 8, 10, 14):
    gk = (k * med - sc[o[:k]].sum()) / n
    print(f"  worst-{k:2d} -> median: +{gk:.4f} ({100*gk/deficit:.1f}% of deficit)")
# what if ALL frames reached the current best frame?
print(f"  every frame -> current best frame ({sc.max():.3f}): mean {sc.max():.3f} "
      f"({100*(sc.max()-sc.mean())/deficit:.1f}% of deficit)")
print(f"  uniform lift needed on all 28: +{deficit:.4f} score/frame")

print("\n=== (b) SHARPNESS COUPLING (GT laplacian variance) ===")
for nm, v in [("score", sc), ("lpips", lp), ("psnr", ps), ("ssim", ss)]:
    r_s, p_s = stats.spearmanr(lv, v)
    r_p, p_p = stats.pearsonr(np.log(lv), v)
    print(f"  {nm:6s} vs GT lapvar : spearman {r_s:+.3f} (p={p_s:.4f})   "
          f"pearson(vs log lapvar) {r_p:+.3f} (p={p_p:.4f})")
r_s, p_s = stats.spearmanr(lv, rlv)
print(f"  render lapvar vs GT lapvar: spearman {r_s:+.3f} (p={p_s:.4g})")
print(f"  render/GT lapvar ratio: median {np.median(rlv/lv):.4f} min {np.min(rlv/lv):.4f} max {np.max(rlv/lv):.4f}")
print(f"  GT lapvar: p10 {np.percentile(lv,10)*1e4:.1f}e-4 med {np.median(lv)*1e4:.1f}e-4 "
      f"p90 {np.percentile(lv,90)*1e4:.1f}e-4  (p90/p10 = {np.percentile(lv,90)/np.percentile(lv,10):.2f}x)")
# tercile means
idx = np.argsort(lv)
t = n // 3
for lab, sel in [("blurriest 1/3", idx[:t]), ("middle 1/3", idx[t:2*t]), ("sharpest 1/3", idx[2*t:])]:
    print(f"  {lab:14s}: score {sc[sel].mean():7.3f}  lpips {lp[sel].mean():.4f}  "
          f"psnr {ps[sel].mean():.3f}  ssim {ss[sel].mean():.4f}  gtlapv {lv[sel].mean()*1e4:.0f}e-4")

print("\n=== (c) SPATIAL CONCENTRATION ===")
tl = np.load(f"{OUT}/{TAG}_tiles_lpips.npy")   # n, ny, nx  (mean lpips density per tile)
t1 = np.load(f"{OUT}/{TAG}_tiles_l1.npy")
co = json.load(open(f"{OUT}/{TAG}_tile_coords.json"))
ys, xs, H, W, T = co["ys"], co["xs"], co["H"], co["W"], co["tile"]
mlp = np.load(f"{OUT}/{TAG}_mean_lpips_map.npy")
ml1 = np.load(f"{OUT}/{TAG}_mean_l1_map.npy")
print(f"  tile grid {len(ys)}x{len(xs)} of {T}px, image {H}x{W}; "
      f"mean-map LPIPS {mlp.mean():.4f} vs scalar {lp.mean():.4f}")
tm = tl.mean(0)   # ny,nx averaged over frames
flat = tm.ravel()
srt = np.sort(flat)[::-1]
for frac in (0.05, 0.10, 0.25, 0.50):
    k = max(1, int(round(frac * flat.size)))
    print(f"  worst {int(frac*100):2d}% of tiles ({k:3d}/{flat.size}) carry "
          f"{100*srt[:k].sum()/flat.sum():.1f}% of total LPIPS "
          f"(uniform would be {100*frac:.0f}%)")
t1m = t1.mean(0).ravel(); s1 = np.sort(t1m)[::-1]
for frac in (0.10, 0.25):
    k = max(1, int(round(frac * t1m.size)))
    print(f"  worst {int(frac*100):2d}% of tiles carry {100*s1[:k].sum()/t1m.sum():.1f}% of total L1")
# gini
def gini(x):
    x = np.sort(x); i = np.arange(1, len(x) + 1)
    return float((2 * (i * x).sum()) / (len(x) * x.sum()) - (len(x) + 1) / len(x))
print(f"  Gini(tile LPIPS) = {gini(flat):.3f}   Gini(tile L1) = {gini(t1m):.3f}")

# where: centre vs periphery, and top tiles' coords
yy, xx = np.meshgrid(np.array(ys) + T / 2, np.array(xs) + T / 2, indexing="ij")
rn = np.sqrt(((yy - H / 2) / (H / 2)) ** 2 + ((xx - W / 2) / (W / 2)) ** 2) / np.sqrt(2)
r_s, p_s = stats.spearmanr(rn.ravel(), flat)
print(f"  tile LPIPS vs normalised radius from image centre: spearman {r_s:+.3f} (p={p_s:.2g})")
for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]:
    m = (rn.ravel() >= lo) & (rn.ravel() < hi)
    if m.sum():
        print(f"    radius [{lo:.2f},{hi:.2f}) : {m.sum():3d} tiles, mean LPIPS density {flat[m].mean():.4f}")
print("  top-12 tiles (y,x of top-left corner) by mean LPIPS:")
ordt = np.argsort(flat)[::-1][:12]
for i in ordt:
    iy, ix = divmod(i, len(xs))
    print(f"    y={ys[iy]:4d} x={xs[ix]:4d}  lpips {flat[i]:.4f}  l1 {t1m[i]:.4f}  "
          f"({flat[i]/flat.mean():.2f}x mean)")
# rows / cols marginals
print("  row-band mean LPIPS density (top->bottom):", " ".join(f"{v:.4f}" for v in tm.mean(1)))
print("  col-band mean LPIPS density (left->right):", " ".join(f"{v:.4f}" for v in tm.mean(0)))

# is the spatial pattern stable across frames? corr of each frame's tile map to the mean
cs = [stats.pearsonr(tl[i].ravel(), flat)[0] for i in range(tl.shape[0])]
print(f"  per-frame tile-map correlation to the 28-frame mean: median {np.median(cs):.3f} "
      f"min {np.min(cs):.3f} max {np.max(cs):.3f}")

# save CSV
import csv
with open(f"{OUT}/{TAG}_perframe.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
np.save(f"{OUT}/{TAG}_tilemean_lpips.npy", tm)
np.save(f"{OUT}/{TAG}_tilemean_l1.npy", t1.mean(0))
print(f"\nwrote {OUT}/{TAG}_perframe.csv")
