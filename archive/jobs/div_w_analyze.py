#!/usr/bin/env python
"""Weight axis for a foreign-family member. Round-13 shipped 3 FastGS-family members on the
set2 towers at 0.40 total weight and graded -0.5961; this locates the optimum share."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f"{HERE}/div_wsweep.json"))
L, n = J["arms"], J["n"]
b = L["base_k4UT"]["score"]
print(f"HCM0181 n={n} lam={J['lam']} gain={J['gain']} FULL SHIPPED CHAIN, base(4 UT)={b:.4f}")
print(f"{'arm':>22} {'share f':>8} {'SCORE':>9} {'vs base':>9} {'dPSNR':>7} "
      f"{'dSSIM pp':>9} {'dLPIPS pp':>10}")
bp, bs, bl = (L["base_k4UT"][k] for k in ("psnr", "ssim", "lpips"))
best = {}
for a in L:
    if a == "base_k4UT":
        continue
    f = float(a.split("f=")[1])
    nm = a.split(" f=")[0]
    r = L[a]
    d = r["score"] - b
    best.setdefault(nm, []).append((f, d))
    print(f"{nm:>22} {f:8.2f} {r['score']:9.4f} {d:+9.4f} {r['psnr']-bp:+7.3f} "
          f"{100*(r['ssim']-bs):+9.4f} {100*(r['lpips']-bl):+10.4f}")
print()
for nm, rows in best.items():
    rows.sort()
    fs = np.array([r[0] for r in rows]); ds = np.array([r[1] for r in rows])
    i = int(ds.argmax())
    # quadratic through the 3 points around the argmax -> continuous optimum
    j = max(1, min(len(fs) - 2, i))
    c = np.polyfit(fs[j - 1:j + 2], ds[j - 1:j + 2], 2)
    fstar = -c[1] / (2 * c[0]) if c[0] < 0 else fs[i]
    print(f"{nm:>22}  grid argmax f={fs[i]:.2f} ({ds[i]:+.4f}); "
          f"quadratic optimum f*={fstar:.3f}, peak {np.polyval(c, fstar):+.4f}; "
          f"uniform-1/5 share=0.20 gives {ds[list(fs).index(0.20)]:+.4f}"
          if 0.20 in list(fs) else "")
