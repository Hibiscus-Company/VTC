"""Which MEMBER does winsor1 actually clip?

Production's tower ensemble is a WEIGHTED 7-way mean: mip3d at w=0.200 and each of the
six base members at 0.8/6 = 0.1333 (build_r25.sh:11-13). The mip3d member is deliberately
UP-weighted because it is the most decorrelated one. A winsorised mean clips the per-pixel
min and max -- so if the most decorrelated member is also the one most often at an extreme,
winsor1 silently DOWN-weights exactly the member production pays extra to include.

This measures, per member: the fraction of (pixel, channel) slots where it is the min or
the max of the 7 and therefore gets replaced by winsorisation.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_lib as A
from agg_run import POOLS

POOL = sys.argv[1] if len(sys.argv) > 1 else "p7"
NV = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def main():
    variants = POOLS[POOL]
    k = len(variants)
    gt_by = {os.path.splitext(f)[0]: 1 for f in os.listdir(A.GT)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(A.mdir(v), s + ".png")) for v in variants))
    stems = stems[:: max(1, len(stems) // NV)][:NV]
    clip = np.zeros(k)
    dist = np.zeros(k)
    tot = 0
    for s in stems:
        X = np.stack([A.load(A.mdir(v), s) for v in variants], 0)
        mn = X.argmin(0)
        mx = X.argmax(0)
        n = mn.size
        for i in range(k):
            clip[i] += (mn == i).sum() + (mx == i).sum()
            # mean |member - consensus|, i.e. how decorrelated / far out this member sits
            dist[i] += np.abs(X[i] - np.median(X, axis=0)).mean() * n
        tot += n
        del X
    clip /= tot
    dist /= tot
    exp = 2.0 / k
    print(f"pool={POOL}  k={k}  views={len(stems)}   expected clip rate if members were "
          f"exchangeable = 2/k = {exp:.3f}\n")
    print(f"{'member':<20}{'clip rate':>11}{'vs 2/k':>10}{'mean|x-med|':>13}")
    for i in np.argsort(-clip):
        print(f"{variants[i]:<20}{clip[i]:11.3f}{clip[i]/exp:9.2f}x{dist[i]:13.5f}")
    print(f"\nspread: max/min clip rate = {clip.max()/clip.min():.2f}x")
    print(f"correlation(clip rate, distance-from-consensus) = "
          f"{np.corrcoef(clip, dist)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
