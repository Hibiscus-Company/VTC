#!/usr/bin/env python
"""Marginal value of the NEXT member vs pool size k, same-family vs foreign-family,
and the honest extrapolation to the production tower pool (k=8 -> 9)."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f"{HERE}/div_ladder.json"))
L, n = J["arms"], J["n"]

print(f"HCM0181 n={n}  lam={J['lam']}  fieldgain={J['gain']}  FULL SHIPPED CHAIN\n")
print("SAME-FAMILY DEPTH LADDER (pool grown by UT-family members only)")
print(f"{'k':>3} {'SCORE':>9} {'marginal':>9}  members")
prev, same = None, {}
for k in range(2, 7):
    a = L[f"Q{k}_UTonly"]
    d = "" if prev is None else f"{a['score']-prev:+9.4f}"
    if prev is not None:
        same[k] = a["score"] - prev
    print(f"{k:>3} {a['score']:9.4f} {d:>9}  {a['members'][-1]}")
    prev = a["score"]

print("\nMARGINAL OF THE (k+1)-th MEMBER: same-family vs foreign-family, identical k step")
print(f"{'step':>8} {'sameFAM(UT)':>12} {'+AA(B8pure)':>12} {'+FGS(e17)':>11} {'AA-same':>9}")
rows_aa, rows_fg = [], []
for k in (2, 3, 4, 5, 6):
    b = L[f"Q{k}_UTonly"]["score"]
    aa = L.get(f"Q{k}+AA(B8pure)", {}).get("score")
    fg = L.get(f"Q{k}+FGS(e17)", {}).get("score")
    sf = same.get(k + 1)
    f2 = lambda v: f"{v:+12.4f}" if v is not None else f"{'--':>12}"
    print(f"{k}->{k+1:<5} {f2(sf)} {f2(aa - b if aa else None)} "
          f"{(fg - b) if fg else float('nan'):+11.4f} "
          f"{(aa - b - sf) if (aa and sf is not None) else float('nan'):+9.4f}")
    if aa:
        rows_aa.append((k, aa - b))
    if fg:
        rows_fg.append((k, fg - b))

def fit(rows, tag):
    kk = np.array([r[0] for r in rows], float)
    dd = np.array([r[1] for r in rows], float)
    ok = dd > 0
    if ok.sum() < 3:
        print(f"\n{tag}: fewer than 3 positive rungs, no extrapolation")
        return
    p = np.polyfit(np.log(kk[ok]), np.log(dd[ok]), 1)
    r2 = np.corrcoef(np.log(kk[ok]), np.log(dd[ok]))[0, 1] ** 2
    print(f"\n{tag}: marginal(k) = {np.exp(p[1]):.4f} * k^({p[0]:+.2f})   R2={r2:.3f}")
    for kq in (6, 7, 8):
        print(f"    extrapolated marginal {kq}->{kq+1}: {np.exp(p[1]) * kq ** p[0]:+.4f}")

fit(rows_aa, "AA foreign add")
fit(rows_fg, "FGS foreign add")
sk = sorted(same)
if len(sk) >= 3:
    fit([(k - 1, same[k]) for k in sk], "same-family add")

b6 = L["Q6_UTonly"]["score"]
for extra in ("Q6+AA+FGS", "Q6+4foreign"):
    if extra in L:
        r = L[extra]
        print(f"{extra:>14}  {r['score']-b6:+.4f} vs Q6   (k={len(r['members'])})")
