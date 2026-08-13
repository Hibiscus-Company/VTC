"""What actually predicts a member's marginal ensemble value?
Compare candidate selection criteria against the MEASURED add-one-to-k4 score deltas."""
import os, json, numpy as np
from scipy.stats import spearmanr, pearsonr

OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens"
mar = json.load(open(os.path.join(OUT, "marginal.json")))["k4"]
corr = {r["name"]: r for r in json.load(open(os.path.join(OUT, "corr.json")))}
th = {r["add"]: r for r in json.load(open(os.path.join(OUT, "theory.json")))}

rows = sorted(mar, key=lambda r: -r["delta"])
print(f"{'add':20s} {'delta':>8s} | {'gap':>7s} {'corr_k4':>8s} {'rmsR':>6s} {'w*MSE':>6s} "
      f"{'analytic dScore':>16s}")
for r in rows:
    x = r["add"]
    print(f"{x:20s} {r['delta']:+8.4f} | {r['gap']:+7.3f} {corr[x]['corr_k4']:8.3f} "
          f"{corr[x]['res_rms']/0.0609:6.2f} {th[x]['wstar']:6.3f} {0.6*th[x]['dpsnr_02']:+16.4f}")

d = np.array([r["delta"] for r in rows])
crit = {
    "single-score gap (the 0.15-band rule)": np.array([r["gap"] for r in rows]),
    "residual corr with base (lower=better)": -np.array([corr[r["add"]]["corr_k4"] for r in rows]),
    "analytic MSE-optimal weight w*": np.array([th[r["add"]]["wstar"] for r in rows]),
    "analytic dScore from PSNR alone @w=0.2": np.array([0.6 * th[r["add"]]["dpsnr_02"] for r in rows]),
}
print("\n=== how well does each criterion predict the measured marginal? (n=%d) ===" % len(d))
for k, v in crit.items():
    print(f"  {k:42s} pearson r={pearsonr(v,d)[0]:+.3f}  spearman rho={spearmanr(v,d)[0]:+.3f}")

good = [i for i, r in enumerate(rows) if r["add"] not in ("sh0", "sh1", "sh2", "gsplatB6bilagrid")]
print("\n=== same, restricted to the 12 'normal-quality' members (drops the 4 crippled ones) ===")
for k, v in crit.items():
    print(f"  {k:42s} pearson r={pearsonr(v[good],d[good])[0]:+.3f}  spearman rho={spearmanr(v[good],d[good])[0]:+.3f}")

print("\n=== what the 0.15-band rule would have done ===")
best = 75.9644
adm = [r for r in rows if r["gap"] >= -0.15]
print(f"  members within 0.15 of best single: {[r['add'] for r in adm]}  -> rule admits NONE of the 16")
pos = [r for r in rows if r["delta"] > 0]
print(f"  members that ACTUALLY help: {len(pos)}/16, best is {pos[0]['add']} at gap {pos[0]['gap']:+.2f} "
      f"(delta {pos[0]['delta']:+.4f})")
print(f"  total score left on the table by the band rule (best single addition): {pos[0]['delta']:+.4f}")
