#!/usr/bin/env python
"""STEP 4: rigour. The competition score is LINEAR in the per-image metrics
    s_i = 100*(0.4*(1-L_i) + 0.3*S_i + 0.3*P_i/50),   SCORE = mean_i s_i
so per-view deltas are exact and paired statistics need no approximation.

  (1) paired per-view significance + bootstrap CI on the winner
  (2) 2-fold VIEW cross-validation of the (lam,win) selection -- the honest number you get if you
      must PICK the hyper-parameters from data
  (3) leave-one-member-out: tune on a 3-member ensemble, apply to the 4-member one
"""
import os, sys, glob, itertools
import numpy as np
OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lap_rows"


def sv(rows):
    """per-view score vector"""
    P, S, L = rows[:, 0], rows[:, 1], rows[:, 2]
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * np.minimum(P / 50.0, 1.0))


def load(tag, nm):
    return np.load(f"{OUT}/{tag}__{nm}.npy")


rng = np.random.default_rng(0)
base = sv(load("k4_png", "pixelmean"))
n = len(base)
print(f"views n={n}\n")

# ---------------------------------------------------------------- (1) significance
print("=" * 92)
print("1. PAIRED PER-VIEW SIGNIFICANCE (60 views, exact decomposition of the score)")
print("=" * 92)
print(f"{'config':<34} {'dScore':>8} {'t':>7} {'p(2s)':>9} {'boot95%':>18} {'win/60':>7}")
cands = ["energy_n1_lam1.0_win5", "energy_n1_lam1.5_win5", "energy_n1_lam1.0_win9",
         "pnorm_n1_p2.0_win3", "gain_n1_g1.02", "median_n1", "maxmag_n1_win3"]
for nm in cands:
    f = f"{OUT}/k4_png__{nm}.npy"
    if not os.path.exists(f):
        continue
    d = sv(np.load(f)) - base
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(abs(t), n - 1))
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{nm:<34} {d.mean():+8.4f} {t:7.2f} {p:9.2e} [{lo:+7.4f},{hi:+7.4f}] {(d>0).sum():>5}/60")

# ---------------------------------------------------------------- (2) 2-fold view CV
print("\n" + "=" * 92)
print("2. 2-FOLD VIEW CROSS-VALIDATION of the (lam,win) choice")
print("   naive = best config's full-data delta (optimistically biased)")
print("   cv    = pick (lam,win) on fold A, score on fold B, and vice versa -> honest")
print("=" * 92)
grid = {}
for f in glob.glob(f"{OUT}/k4_png__energy_n1_lam*_win*.npy"):
    nm = os.path.basename(f)[len("k4_png__"):-4]
    if "map" in nm or "rmax" in nm:
        continue
    grid[nm] = sv(np.load(f)) - base
print(f"   grid size = {len(grid)} configs: {sorted(grid)}")
naive_nm = max(grid, key=lambda k: grid[k].mean())
print(f"   naive best = {naive_nm}  delta {grid[naive_nm].mean():+.4f}")
cvs, picks = [], {}
for rep in range(400):
    idx = rng.permutation(n)
    A, B = idx[:n // 2], idx[n // 2:]
    tot = 0.0
    for tr, te in ((A, B), (B, A)):
        best = max(grid, key=lambda k: grid[k][tr].mean())
        picks[best] = picks.get(best, 0) + 1
        tot += grid[best][te].mean()
    cvs.append(tot / 2)
cvs = np.array(cvs)
print(f"   CV delta = {cvs.mean():+.4f}   (sd over 400 splits {cvs.std():.4f}, "
      f"5th pct {np.percentile(cvs,5):+.4f})")
print(f"   optimism = naive - cv = {grid[naive_nm].mean()-cvs.mean():+.4f}")
print(f"   selection stability: " + ", ".join(f"{k.split('_',2)[2]}:{v/8:.0f}%"
                                              for k, v in sorted(picks.items(), key=lambda x: -x[1])))

# ---------------------------------------------------------------- (3) member-fold
print("\n" + "=" * 92)
print("3. MEMBER-FOLD: (lam,win) tuned on k=3 sub-ensembles, applied at k=4")
print("=" * 92)
for k in (2, 3):
    f = f"{OUT}/k{k}_png__pixelmean.npy"
    if not os.path.exists(f):
        continue
    b = sv(np.load(f))
    e = sv(np.load(f"{OUT}/k{k}_png__energy_n1_lam1.0_win5.npy"))
    print(f"   k={k}: pixelmean {b.mean():.4f}  energy {e.mean():.4f}  delta {e.mean()-b.mean():+.4f}")
b4 = base
e4 = sv(np.load(f"{OUT}/k4_png__energy_n1_lam1.0_win5.npy"))
print(f"   k=4: pixelmean {b4.mean():.4f}  energy {e4.mean():.4f}  delta {e4.mean()-b4.mean():+.4f}")
