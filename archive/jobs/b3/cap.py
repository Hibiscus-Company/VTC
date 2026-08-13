#!/usr/bin/env python
"""B3 Job-2: does the bonsai capacity ladder actually extrapolate past 5M?"""
import math
# 17/07 UT-recipe ladder, PRE-drift scorer. a4 measured the drift at -0.947 on capD.
lad = [(0.5e6, 69.5706), (1e6, 70.2975), (2e6, 70.6599), (5e6, 71.1563)]
DRIFT = -0.9474
print("17/07 cap ladder (UT recipe), as logged and re-based onto today's scorer:")
for n, s in lad:
    print(f"  {n/1e6:5.1f}M   logged {s:.4f}   today-equivalent {s+DRIFT:.4f}")
print("\nmarginal gain per DOUBLING of cap:")
for (n0, s0), (n1, s1) in zip(lad, lad[1:]):
    d = math.log2(n1 / n0)
    print(f"  {n0/1e6:>4.1f}M -> {n1/1e6:>4.1f}M  ({d:.3f} doublings)  d={s1-s0:+.4f}  "
          f"per-doubling {(s1-s0)/d:+.4f}")
slope = (lad[3][1] - lad[2][1]) / math.log2(lad[3][0] / lad[2][0])
print(f"\nlocal slope at the top of the ladder (2M->5M): {slope:+.4f} / doubling")
for cap in (8e6, 10e6, 16e6):
    d = math.log2(cap / 5e6)
    print(f"  extrapolated 5M -> {cap/1e6:4.1f}M ({d:.3f} doublings): predicted {slope*d:+.4f}")

print("\n--- THE DIRECT TEST (single variable, same recipe generation, same scorer) ---")
print("  5M  bonsai_eval/K1_noUT_aa  --iters 30000 --cap_max 5000000 --refine_stop 15000"
      " --noise_stop 8000 --lpips_from 12000   PNG   71.9029")
print("  8M  r33_bonsai/eval_s42     IDENTICAL except --cap_max 8000000                "
      "  JPEG  70.0864")
print("  (identical command strings verified from the two train_args.txt / launch scripts;")
print("   the ONLY confound is delivery format: 5M scored on PNG, 8M scored on JPEG q100/4:4:4)")
