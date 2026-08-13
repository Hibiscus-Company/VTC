#!/usr/bin/env python
"""The r29 tower pool is 6 plain members + 2 mip3d members at wB=0.333 (1.5x per member).
Reproduce that SHAPE on the harness and find the exact MSE optimum, plus the value of pool
COMPOSITION versus pool WEIGHTING."""
import numpy as np
from a1_solve import NAMES, psnr, ls_weights
from a2_family import IX, ev, odd

UT = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
A6 = UT + ["e15ceil95", "gsplatB7ppisp2"]        # "plain" family, 6 members
B2 = ["m31b_taillpips", "e17visnorm"]            # "second family", 2 members
ia, ib = [IX[x] for x in A6], [IX[x] for x in B2]
idx = ia + ib

print("PRODUCTION SHAPE: 6 + 2, sweep the total weight on the 2-member family")
g = np.arange(0, 1.001, 0.005)
ps = np.array([psnr(np.concatenate([np.full(6, (1 - w) / 6), np.full(2, w / 2)]), idx) for w in g])
j = int(ps.argmax())
pu = ps[np.argmin(np.abs(g - 0.25))]
pprod = ps[np.argmin(np.abs(g - 0.3333))]
print(f"   uniform(1/8 each)  wB=0.250  PSNR {pu:.4f}")
print(f"   r29 ships          wB=0.333  PSNR {pprod:.4f}   ({pprod-pu:+.4f} dB vs uniform)")
print(f"   MSE optimum        wB={g[j]:.3f}  PSNR {ps[j]:.4f}   ({ps[j]-pu:+.4f} dB vs uniform)")
r = (g[j] / 2) / ((1 - g[j]) / 6)
print(f"   => per-member ratio at the optimum {r:.2f}x  (r29 assumes 1.50x)")
print("   curve:", "  ".join(f"{v:.2f}:{ps[np.argmin(np.abs(g-v))]:.3f}"
                             for v in (0.0, 0.1, 0.2, 0.25, 0.3, 0.333, 0.4, 0.5, 0.6)))

print("\nCOMPOSITION vs WEIGHTING (all exact, MSE/PSNR):")
u4 = psnr(np.full(4, 0.25), [IX[x] for x in UT])
u8 = psnr(np.full(8, 0.125), idx)
w8 = ls_weights(idx, sel=ev)
u8o = psnr(np.full(8, 0.125), idx, sel=odd)
l8o = psnr(w8, idx, sel=odd)
print(f"   4 UT members, uniform            {u4:.4f} dB")
print(f"   8 members (4 families), uniform  {u8:.4f} dB   composition gain {u8-u4:+.4f} dB")
print(f"   8 members, LS weights (held out) {l8o-u8o:+.4f} dB   weighting gain on top")
print(f"   => composition is worth {(u8-u4)/max(l8o-u8o,1e-9):.0f}x the weighting")

print("\nSENSITIVITY of the family weight (how flat is the optimum?):")
for d in (0.05, 0.10, 0.15):
    lo = ps[np.argmin(np.abs(g - max(g[j] - d, 0)))]
    hi = ps[np.argmin(np.abs(g - min(g[j] + d, 1)))]
    print(f"   +/-{d:.2f} around the optimum costs {lo-ps[j]:+.4f} / {hi-ps[j]:+.4f} dB "
          f"= {0.6*(lo-ps[j]):+.4f} / {0.6*(hi-ps[j]):+.4f} blended points")
