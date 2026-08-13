#!/usr/bin/env python
"""Join the arm scores with the residual-correlation structure and answer:
does score gain track DECORRELATION, or just member QUALITY?"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from div_lib import FAM, UT4

A = json.load(open(f"{HERE}/div_add5.json"))
order = A["order"]
G = np.array(A["gram"])                 # <e_i, e_j> per pixel-channel, in [0,1] units
dec = np.array(A["decorr255"])
smse = np.array(A["solo_mse"])
i4 = [order.index(m) for m in UT4]
base_e2 = G[np.ix_(i4, i4)].mean()      # MSE of the 4-UT mean = mean of the 4x4 Gram block

try:
    S = json.load(open(f"{HERE}/div_solo.json"))["arms"]
except FileNotFoundError:
    S = {}

b = A["arms"]["base_k4UT"]["score"]
rows = []
for i, m in enumerate(order):
    if m in UT4:
        continue
    key = f"+{m}"
    if key not in A["arms"]:
        continue
    r = A["arms"][key]
    # correlation of this member's error with the 4-UT pool-mean error
    ce = G[i, i4].mean() / np.sqrt(G[i, i] * base_e2)
    # exact analytic PSNR-side prediction for the k=5 mean (no restore, no chain):
    # MSE(mean of 5) = (16*base_e2 + 8*G[i,i4].mean() + G[i,i]) / 25
    m5 = (16 * base_e2 + 8 * G[i, i4].mean() + G[i, i]) / 25.0
    rows.append(dict(name=m, fam=FAM[m], d=r["score"] - b, dec=dec[i],
                     corr=ce, solo_psnr=10 * np.log10(1 / smse[i]),
                     solo_chain=S.get(f"solo:{m}", {}).get("score", float("nan")),
                     pred_dpsnr=10 * np.log10(base_e2 / m5),
                     psnr=r["psnr"], ssim=r["ssim"], lpips=r["lpips"]))

print(f"HCM0181 n={A['n']}  base(4 UT) score={b:.4f}  lam={A['lam']} gain={A['gain']}")
print(f"{'5th member':>18} {'fam':>4} {'dSCORE':>8} {'dPSNR':>7} {'dSSIM':>8} {'dLPIPS':>8} "
      f"{'errcorr':>8} {'decorr':>7} {'soloPSNR':>9} {'soloCHAIN':>10} {'predPSNR':>9}")
bp, bs, bl = (A["arms"]["base_k4UT"][k] for k in ("psnr", "ssim", "lpips"))
for r in sorted(rows, key=lambda z: -z["d"]):
    print(f"{r['name']:>18} {r['fam']:>4} {r['d']:+8.4f} {r['psnr']-bp:+7.3f} "
          f"{100*(r['ssim']-bs):+8.4f} {100*(r['lpips']-bl):+8.4f} {r['corr']:8.4f} "
          f"{r['dec']:7.3f} {r['solo_psnr']:9.3f} {r['solo_chain']:10.4f} {r['pred_dpsnr']:+9.3f}")

for c in ("NOISECTL", "CLONECTL"):
    if c in A["arms"]:
        r = A["arms"][c]
        print(f"{c:>18} {'CTL':>4} {r['score']-b:+8.4f} {r['psnr']-bp:+7.3f} "
              f"{100*(r['ssim']-bs):+8.4f} {100*(r['lpips']-bl):+8.4f}")

d = np.array([r["d"] for r in rows])
x_dec = np.array([r["dec"] for r in rows])
x_cor = np.array([r["corr"] for r in rows])
x_q = np.array([r["solo_psnr"] for r in rows])
print(f"\ncorr(dSCORE, decorrelation |m-mean|) = {np.corrcoef(x_dec, d)[0,1]:+.3f}")
print(f"corr(dSCORE, error-correlation rho) = {np.corrcoef(x_cor, d)[0,1]:+.3f}")
print(f"corr(dSCORE, solo PSNR)             = {np.corrcoef(x_q, d)[0,1]:+.3f}")
X = np.stack([np.ones_like(d), x_cor, x_q], 1)
beta, *_ = np.linalg.lstsq(X, d, rcond=None)
pred = X @ beta
R2 = 1 - ((d - pred) ** 2).sum() / ((d - d.mean()) ** 2).sum()
print(f"OLS dSCORE ~ 1 + rho + soloPSNR: b_rho={beta[1]:+.4f}  b_q={beta[2]:+.4f}  R2={R2:.3f}")
print("\nby family (mean dSCORE):")
for f in ("AA", "FGS", "UTd"):
    sel = [r["d"] for r in rows if r["fam"] == f]
    sc = [r["corr"] for r in rows if r["fam"] == f]
    if sel:
        print(f"  {f:>4}  n={len(sel):2d}  dSCORE {np.mean(sel):+.4f} "
              f"(min {min(sel):+.4f} max {max(sel):+.4f})   mean rho {np.mean(sc):.4f}")
