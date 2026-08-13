#!/usr/bin/env python
"""Aggregate a (possibly partial) p3/p5 per-image json into the blended-score table with
paired standard errors against a chosen reference arm."""
import json, sys
import numpy as np

path, ref = sys.argv[1], sys.argv[2]
d = json.load(open(path))
per = d["per_image"] if "per_image" in d else d
arms = list(next(iter(per.values())).keys())
n = len(per)
stems = sorted(per)


def sc(P, S, L):
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))


P = {a: np.mean([per[s][a][0] for s in stems]) for a in arms}
S = {a: np.mean([per[s][a][1] for s in stems]) for a in arms}
L = {a: np.mean([per[s][a][2] for s in stems]) for a in arms}
MB = {a: sum(per[s][a][3] for s in stems) / 1e6 / n * 60 for a in arms}
pim = {a: np.array([sc(*per[s][a][:3]) for s in stems]) for a in arms}
agg = {a: sc(P[a], S[a], L[a]) for a in arms}
base = "base" if "base" in arms else arms[0]
print(f"n={n}  ref={ref}")
print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
      f"{'vs base':>9} {'vs ref':>9} {'pairSE':>8} {'t':>7} {'MB/60':>7}")
for a in arms:
    dlt = pim[a] - pim[ref]
    se = dlt.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    t = dlt.mean() / se if se > 0 else float("nan")
    print(f"{a:>12} {agg[a]:9.4f} {P[a]:8.4f} {S[a]:7.4f} {L[a]:8.4f} "
          f"{agg[a]-agg[base]:+9.4f} {agg[a]-agg[ref]:+9.4f} {se:8.4f} {t:7.2f} {MB[a]:7.2f}")
