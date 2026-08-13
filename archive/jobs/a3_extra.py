"""A3 extra: disentangle 'sharp GT' from 'early in the sequence'; per-frame band ratios;
content-normalised sharpness; edge/texture localisation of the error map."""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_common import *  # noqa
import numpy as np
from scipy import stats
import cv2

OUT = "/mnt/d/avv/r42_bonsai78/a3_diag"
RENDER = "/mnt/d/avv/r36_shape/sr01/eval_png"
GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
rows = json.load(open(f"{OUT}/bonsai_sr01_perframe.json"))
rows.sort(key=lambda r: r["frame"])
pairs = {s: (rp, gp) for s, rp, gp in pair_list(RENDER, GT)}

edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def bands_of(g, rbin, NB, cnt):
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2((g - g.mean()) * win))
    P = F.real ** 2 + F.imag ** 2
    s = np.bincount(rbin, weights=P.ravel(), minlength=NB)
    rc = (np.arange(NB) + 0.5) / NB
    return np.array([s[(rc >= a) & (rc < b)].sum() for a, b in zip(edges[:-1], edges[1:])])


NB = 256
rbin = None
per = []
for r in rows:
    rp, gp = pairs[r["stem"]]
    gR = gray(load(rp)); gG = gray(load(gp))
    if rbin is None:
        H, W = gG.shape
        fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None] / 0.5
        fx = np.fft.fftshift(np.fft.fftfreq(W))[None, :] / 0.5
        rbin = np.clip((np.sqrt(fy**2 + fx**2) * NB).astype(np.int64), 0, NB - 1).ravel()
        cnt = np.bincount(rbin, minlength=NB)
    bR = bands_of(gR, rbin, NB, cnt); bG = bands_of(gG, rbin, NB, cnt)
    hf_gt = float(bG[2:].sum() / bG.sum())      # content-normalised GT sharpness
    hf_rn = float(bR[2:].sum() / bR.sum())
    per.append(dict(**r, hf_frac_gt=hf_gt, hf_frac_render=hf_rn,
                    **{f"ratio_b{i}": float(bR[i] / bG[i]) for i in range(5)}))
    print(f"{r['stem']} sc{r['score']:6.2f} hf_gt {hf_gt:.4f} hf_rn {hf_rn:.4f} "
          f"ratios " + " ".join(f"{bR[i]/bG[i]:.3f}" for i in range(5)), flush=True)

sc = np.array([p["score"] for p in per]); lp = np.array([p["lpips"] for p in per])
lv = np.array([p["gt_lapvar"] for p in per]); fr = np.array([p["frame"] for p in per], float)
hf = np.array([p["hf_frac_gt"] for p in per])
b34 = np.array([(p["ratio_b3"] + p["ratio_b4"]) / 2 for p in per])

print("\n=== CONFOUND: sharpness vs position in the sequence ===")
for nm, v in [("gt_lapvar", lv), ("hf_frac_gt", hf), ("score", sc), ("lpips", lp)]:
    print(f"  {nm:12s} vs frame index: spearman {stats.spearmanr(fr, v)[0]:+.3f} "
          f"(p={stats.spearmanr(fr, v)[1]:.4f})")


def partial_spearman(x, y, z):
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, _ = stats.pearsonr(ex, ey)
    n = len(x); df = n - 3
    t = r * np.sqrt(df / max(1e-12, 1 - r * r))
    return r, 2 * (1 - stats.t.cdf(abs(t), df))


print("\n  partial (rank) correlations:")
for nm, v in [("score", sc), ("lpips", lp)]:
    r1, p1 = partial_spearman(lv, v, fr)
    r2, p2 = partial_spearman(fr, v, lv)
    r3, p3 = partial_spearman(hf, v, fr)
    print(f"    {nm:6s} ~ gt_lapvar  | frame index : {r1:+.3f} (p={p1:.4f})")
    print(f"    {nm:6s} ~ hf_frac_gt | frame index : {r3:+.3f} (p={p3:.4f})")
    print(f"    {nm:6s} ~ frame index| gt_lapvar   : {r2:+.3f} (p={p2:.4f})")

print("\n  score vs content-normalised sharpness hf_frac_gt: "
      f"spearman {stats.spearmanr(hf, sc)[0]:+.3f} (p={stats.spearmanr(hf, sc)[1]:.4f})")
print("  lpips vs hf_frac_gt: "
      f"spearman {stats.spearmanr(hf, lp)[0]:+.3f} (p={stats.spearmanr(hf, lp)[1]:.4f})")
print(f"  hf_frac_gt range: {hf.min():.4f} .. {hf.max():.4f} (median {np.median(hf):.4f}); "
      f"render {np.min([p['hf_frac_render'] for p in per]):.4f} .. "
      f"{np.max([p['hf_frac_render'] for p in per]):.4f}")
print(f"  per-frame HF band ratio (0.6-1.0 Nyq) render/GT: median {np.median(b34):.3f} "
      f"min {b34.min():.3f} max {b34.max():.3f}")
print(f"  score vs HF band ratio: spearman {stats.spearmanr(b34, sc)[0]:+.3f}")

# split first 8 vs last 20
a, b = np.arange(8), np.arange(8, 28)
print(f"\n  frames 10-710   (n=8) : score {sc[a].mean():6.3f} lpips {lp[a].mean():.4f} "
      f"gtlapv {lv[a].mean()*1e4:.0f}e-4 hf_gt {hf[a].mean():.4f} hfratio {b34[a].mean():.3f}")
print(f"  frames 810-2650 (n=20): score {sc[b].mean():6.3f} lpips {lp[b].mean():.4f} "
      f"gtlapv {lv[b].mean()*1e4:.0f}e-4 hf_gt {hf[b].mean():.4f} hfratio {b34[b].mean():.3f}")

with open(f"{OUT}/bonsai_sr01_perframe.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
    w.writeheader()
    [w.writerow(p) for p in per]
json.dump(per, open(f"{OUT}/bonsai_sr01_perframe.json", "w"), indent=1)
print(f"\nrewrote {OUT}/bonsai_sr01_perframe.csv with band ratios")
