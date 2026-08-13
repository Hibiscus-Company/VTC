#!/usr/bin/env python
"""Family weighting, exact in MSE, from the cached Gram. Also emits the pass-2 arm spec."""
import json
import numpy as np
from a1_solve import Gd, bd, SS, P, NAMES, FAM, NIM, psnr, ls_weights, mse_per_image

IX = {n: i for i, n in enumerate(NAMES)}
odd = np.arange(NIM) % 2 == 1
ev = ~odd


def blend(A, B, tag, grid=None):
    ia, ib = [IX[x] for x in A], [IX[x] for x in B]
    idx = ia + ib
    kA, kB = len(A), len(B)
    g = grid if grid is not None else np.arange(0, 1.0001, 0.01)
    ps = []
    for wB in g:
        w = np.concatenate([np.full(kA, (1 - wB) / kA), np.full(kB, wB / kB)])
        ps.append(psnr(w, idx))
    ps = np.array(ps)
    j = int(ps.argmax())
    uni = kB / (kA + kB)
    pu = ps[np.argmin(np.abs(g - uni))]
    prod = 1.5 * kB / (kA + 1.5 * kB)                     # production's 1.5x-per-member heuristic
    pp = ps[np.argmin(np.abs(g - prod))]
    ratio = (g[j] / kB) / max((1 - g[j]) / kA, 1e-9)
    print(f"\n[{tag}] A={A} | B={B}")
    print(f"   uniform  wB={uni:5.3f}  PSNR {pu:8.4f}")
    print(f"   prod1.5x wB={prod:5.3f}  PSNR {pp:8.4f}  ({pp-pu:+.4f} dB vs uniform)")
    print(f"   OPTIMUM  wB={g[j]:5.3f}  PSNR {ps[j]:8.4f}  ({ps[j]-pu:+.4f} dB vs uniform)"
          f"   => per-member ratio {ratio:5.2f}x")
    show = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]
    print("   curve:", "  ".join(f"{v:.2f}:{ps[np.argmin(np.abs(g-v))]:.3f}" for v in show))
    return dict(uni=uni, prod=prod, opt=float(g[j]), ratio=float(ratio),
                d_prod=float(pp - pu), d_opt=float(ps[j] - pu), idx=idx, kA=kA, kB=kB)


def pool_report(names, tag):
    idx = [IX[n] for n in names]
    k = len(idx)
    uni = np.full(k, 1.0 / k)
    pu = psnr(uni, idx)
    wls = ls_weights(idx)
    we, wo = ls_weights(idx, sel=ev), ls_weights(idx, sel=odd)
    cv = 0.5 * (psnr(we, idx, sel=odd) + psnr(wo, idx, sel=ev))
    cvu = 0.5 * (psnr(uni, idx, sel=odd) + psnr(uni, idx, sel=ev))
    print(f"\n[{tag}] k={k}  uniform {pu:.4f} | LS oracle {psnr(wls,idx):.4f} "
          f"({psnr(wls,idx)-pu:+.4f}) | LS 2-fold CV {cv-cvu:+.4f} dB honest")
    print("   w_LS =", np.array2string(wls, precision=3, suppress_small=True))
    return idx, wls, we, wo, pu


UT = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]

print("=" * 100)
print("A. Does the OPTIMAL family weight track solo quality? (the production 1.5x claim)")
r1 = blend(UT, ["m31b_taillpips"], "UT4 | m31b_taillpips  (B BETTER solo than 3/4 of A)")
r2 = blend(UT, ["e17visnorm"], "UT4 | e17visnorm  (B slightly WORSE solo)")
r3 = blend(UT, ["gsplatB7ppisp2"], "UT4 | gsplatB7ppisp2 (B worse solo, different family)")
r4 = blend(UT, ["gsplatB8pure"], "UT4 | gsplatB8pure  (B non-UT, worse solo)")
r5 = blend(UT, ["m31b_taillpips", "e17visnorm"], "UT4 | 2-member other family")
r6 = blend(UT[:3], [UT[3]], "UT3 | UT4th  (SAME-family control)")
r7 = blend(UT, ["gsplatB6bilagrid"], "UT4 | bilagrid (BROKEN member, 17.5 dB solo)")

print("\n" + "=" * 100)
print("B. Does per-member LS weighting beat uniform on a sane pool?")
P8 = UT + ["m31b_taillpips", "e17visnorm", "e15ceil95", "gsplatB7ppisp2"]
i8, w8, w8e, w8o, pu8 = pool_report(P8, "P8 sane pool")
P5 = UT + ["m31b_taillpips"]
i5, w5, w5e, w5o, pu5 = pool_report(P5, "P5 = UT4 + m31b")

print("\n" + "=" * 100)
print("C. cross-fold weight stability (does the LS answer move between image halves?)")
for tag, a, b in [("P8", w8e, w8o), ("P5", w5e, w5o)]:
    print(f"   {tag}: max|w_even - w_odd| = {np.abs(a-b).max():.4f}  "
          f"corr = {np.corrcoef(a,b)[0,1]:.4f}")

# ---------------------------------------------------------------- pass-2 arm spec
kA, kB = 4, 1
def famw(wB, kA=4, kB=1):
    return list(np.concatenate([np.full(kA, (1 - wB) / kA), np.full(kB, wB / kB)]))

spec = dict(members=P5, arms=[
    dict(name="uniform_CONTROL", w=famw(0.2)),
    dict(name="wB0.00_UTonly", w=famw(0.0)),
    dict(name="wB0.273_prod1.5x", w=famw(1.5 / 5.5)),
    dict(name=f"wB{r1['opt']:.2f}_MSEopt", w=famw(r1["opt"])),
    dict(name="wB0.50", w=famw(0.5)),
    dict(name="LS_oracle", w=list(w5)),
    dict(name="LS_heldout_even", w=list(w5e)),
])
json.dump(spec, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/wtq/arms_P5.json", "w"), indent=1)
spec8 = dict(members=P8, arms=[
    dict(name="uniform8_CONTROL", w=[1 / 8] * 8),
    dict(name="LS8_oracle", w=list(w8)),
    dict(name="LS8_heldout_even", w=list(w8e)),
    dict(name="UT4_only", w=[0.25] * 4 + [0.0] * 4),
])
json.dump(spec8, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/wtq/arms_P8.json", "w"), indent=1)
print("\nwrote arms_P5.json / arms_P8.json")
print("P5 arms:", [(a["name"], [round(x, 3) for x in a["w"]]) for a in spec["arms"]])
print("P8 arms:", [(a["name"], [round(x, 3) for x in a["w"]]) for a in spec8["arms"]])
