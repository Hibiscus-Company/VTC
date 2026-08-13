"""A3 consolidation: merge everything into one summary CSV + final numbers."""
import json, csv, os
import numpy as np
from scipy import stats

OUT = "/mnt/d/avv/r42_bonsai78/a3_diag"
pf = {r["stem"]: r for r in json.load(open(f"{OUT}/bonsai_sr01_perframe.json"))}
gm = {r["stem"]: r for r in json.load(open(f"{OUT}/bonsai_geom.json"))}
cz = {r["stem"]: r for r in json.load(open(f"{OUT}/bonsai_sr01_cause.json"))}
stems = sorted(pf)
merged = []
for s in stems:
    d = dict(pf[s])
    d.update({k: v for k, v in gm[s].items() if k not in d})
    d.update({k: v for k, v in cz[s].items() if k not in d})
    merged.append(d)
with open(f"{OUT}/bonsai_sr01_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(merged[0].keys())); w.writeheader()
    [w.writerow(m) for m in merged]

sc = np.array([m["score"] for m in merged]); lp = np.array([m["lpips"] for m in merged])
lv = np.array([m["gt_lapvar"] for m in merged]); dm5 = np.array([m["d_mean5"] for m in merged])
dep = np.array([m["med_scene_depth"] for m in merged])
nw = np.array([m["n_within_0p5"] for m in merged], float)

print("=== confound chain ===")
print(f"  gt_lapvar vs med_scene_depth : spearman {stats.spearmanr(lv, dep)[0]:+.3f}")
print(f"  gt_lapvar vs d_mean5         : spearman {stats.spearmanr(lv, dm5)[0]:+.3f}")
print(f"  score     vs d_mean5         : spearman {stats.spearmanr(dm5, sc)[0]:+.3f}")


def pspear(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, _ = stats.pearsonr(ex, ey)
    df = len(x) - 3
    t = r * np.sqrt(df / max(1e-12, 1 - r * r))
    return r, 2 * (1 - stats.t.cdf(abs(t), df))


print(f"  score ~ gt_lapvar | d_mean5  : {pspear(lv, sc, dm5)[0]:+.3f} (p={pspear(lv,sc,dm5)[1]:.3f})")
print(f"  score ~ d_mean5   | gt_lapvar: {pspear(dm5, sc, lv)[0]:+.3f} (p={pspear(dm5,sc,lv)[1]:.3f})")
print(f"  lpips ~ gt_lapvar | d_mean5  : {pspear(lv, lp, dm5)[0]:+.3f} (p={pspear(lv,lp,dm5)[1]:.3f})")
print(f"  lpips ~ d_mean5   | gt_lapvar: {pspear(dm5, lp, lv)[0]:+.3f} (p={pspear(dm5,lp,lv)[1]:.3f})")
# R^2 of log(d_mean5) alone
X = np.stack([np.log(dm5), np.ones_like(dm5)], 1)
b, *_ = np.linalg.lstsq(X, sc, rcond=None)
r2 = 1 - ((sc - X @ b) ** 2).sum() / ((sc - sc.mean()) ** 2).sum()
print(f"  linear fit score ~ log(d_mean5): R^2 = {r2:.3f}")
X2 = np.stack([np.log(dm5), np.log(lv), np.ones_like(dm5)], 1)
b2, *_ = np.linalg.lstsq(X2, sc, rcond=None)
r22 = 1 - ((sc - X2 @ b2) ** 2).sum() / ((sc - sc.mean()) ** 2).sum()
print(f"  + log(gt_lapvar)              : R^2 = {r22:.3f} (adds {r22-r2:+.3f})")

print("\n=== spatial split of LPIPS mass ===")
m = np.load(f"{OUT}/bonsai_sr01_mean_lpips_map.npy").astype(np.float64)
H, W = m.shape
tot = m.sum()
print(f"  top    third rows: {100*m[:H//3].sum()/tot:.1f}% of LPIPS mass (33.3% if uniform)")
print(f"  middle third rows: {100*m[H//3:2*H//3].sum()/tot:.1f}%")
print(f"  bottom third rows: {100*m[2*H//3:].sum()/tot:.1f}%")
print(f"  bottom/top row-band density ratio: {m[2*H//3:].mean()/m[:H//3].mean():.2f}x")
b = 192
per_frac = 1 - (H - 2*b) * (W - 2*b) / (H * W)
inner = m[b:-b, b:-b].sum()
print(f"  outer {b}px border ({100*per_frac:.0f}% of pixels): {100*(tot-inner)/tot:.1f}% of LPIPS mass")
for nm in ["first8", "last20"]:
    t = np.load(f"{OUT}/bonsai_sr01_tilemean_lpips_{nm}.npy")
    print(f"  {nm}: bottom3/top3 tile-row ratio {t[-3:].mean()/t[:3].mean():.2f}x")

print("\n=== headline arithmetic ===")
med = float(np.median(sc)); o = np.argsort(sc)
print(f"  mean {sc.mean():.4f}  median {med:.4f}  deficit to 78 = {78-sc.mean():.4f}")
print(f"  first8 mean {sc[:8].mean():.4f}  last20 mean {sc[8:].mean():.4f}  gap {sc[8:].mean()-sc[:8].mean():.4f}")
print(f"  worst8 == first8: {sorted(o[:8].tolist()) == list(range(8))}")
for tgt, lab in [(med, "median 74.05"), (sc[8:].mean(), "last-20 mean 75.99"),
                 (float(sc.max()), "best frame 81.98")]:
    new = (8 * tgt + sc[8:].sum()) / 28
    print(f"  if first8 -> {lab:20s}: mean {new:.3f} (+{new-sc.mean():.3f}, "
          f"{100*(new-sc.mean())/(78-sc.mean()):.0f}% of deficit)")
print(f"  need first8 at {(78*28 - sc[8:].sum())/8:.2f} for the scene to hit 78 with last20 unchanged")
print(f"  or a uniform +{78-sc.mean():.3f} on every frame")
