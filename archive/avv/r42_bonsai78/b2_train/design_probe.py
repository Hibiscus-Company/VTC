#!/usr/bin/env python
"""B2 design probe -- CPU ONLY, no GPU, no training.

Answers the four questions that decide the shape of the training-side patch:

  Q1  Is the per-frame sharpness statistic CONFOUNDED with capture position / view
      coverage on the 220 train_sub frames?  (A3 showed lapvar is a depth proxy on the
      28 holes: rho=+0.786 with median scene depth.  If it is also a coverage proxy on
      the TRAIN frames, then "sharpness weighting" is secretly "far-field weighting"
      and the arm is uninterpretable.)
  Q2  What does a HARD top-k sharpness mask do to TEMPORAL COVERAGE?  Sharpness is
      autocorrelated (r=+0.70 at lag 10), so the sharp frames come in RUNS and a naive
      top-k mask will punch multi-hundred-frame holes in the trajectory.  That is the
      dominant failure mode of (1)-hard.
  Q3  Build the coverage-MATCHED sharp/blurry/random pair-stratified subsets that make
      the decisive gate arm possible (one frame kept per consecutive pair -> identical
      coverage by construction, opposite sharpness bias).
  Q4  What per-view weights / per-view blur sigmas does each parameterisation actually
      produce (spread, effective sample size, sigma range)?

Writes the subset lists + a sidecar sharpness/sigma table the patched trainer reads.
"""
import os, sys, json, csv, math
import numpy as np

HERE = "/mnt/d/avv/r42_bonsai78/b2_train"
A2 = "/mnt/d/avv/r42_bonsai78/a2_blurpred"
ES = "/mnt/d/avv/evalsplit/bonsai"

os.makedirs(f"{HERE}/subsets", exist_ok=True)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def partial(a, b, c, deg=2):
    """partial corr(a,b | polynomial of degree deg in c)"""
    X = np.vander(np.asarray(c, float), deg + 1, increasing=True)
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return pearson(ra, rb)


# ------------------------------------------------------------------ load sharpness
rows = list(csv.DictReader(open(f"{A2}/train248_sharp.csv")))
S = {int(r["frame"]): {k: float(v) for k, v in r.items() if k != "name"} for r in rows}
EVAL = set(json.load(open(f"{ES}/split.json"))["eval_frames"])
SUB = sorted(f for f in S if f not in EVAL)          # the 220 train_sub frames
assert len(SUB) == 220, len(SUB)

fr = np.array(SUB, float)
llv = np.array([S[f]["log_lapvar"] for f in SUB])
reb = np.array([S[f]["reblur"] for f in SUB])
lhf = np.array([S[f]["log_hf025"] for f in SUB])

print("=" * 78)
print("Q1  sharpness statistics on the 220 train_sub frames")
print("=" * 78)
for nm, v in (("log_lapvar", llv), ("reblur", reb), ("log_hf025", lhf)):
    print(f"  {nm:11s} mean {v.mean():+.4f} sd {v.std():.4f}  p10 {np.percentile(v,10):+.4f} "
          f"p90 {np.percentile(v,90):+.4f}  p90/p10 spread {np.percentile(v,90)-np.percentile(v,10):+.4f}")
print(f"  corr(log_lapvar, reblur)   pearson {pearson(llv, reb):+.3f}  spearman {spearman(llv, reb):+.3f}")
print(f"  corr(log_lapvar, log_hf025) pearson {pearson(llv, lhf):+.3f}")
print(f"  corr(reblur,     log_hf025) pearson {pearson(reb, lhf):+.3f}")
print("  -- confound with capture position (frame index) --")
for nm, v in (("log_lapvar", llv), ("reblur", reb), ("log_hf025", lhf)):
    print(f"  {nm:11s} vs frame: pearson {pearson(v, fr):+.3f}  spearman {spearman(v, fr):+.3f}")

# ------------------------------------------------------------------ geometry / coverage
print()
print("  -- confound with VIEW COVERAGE (train_sub camera geometry) --")
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
imgs = read_extrinsics_binary(f"{ES}/train_sub/sparse/0/images.bin")
cen = {}
for im in imgs.values():
    f = int(os.path.splitext(im.name)[0].split("_")[-1])
    R = qvec2rotmat(im.qvec)
    cen[f] = -R.T @ im.tvec
have = [f for f in SUB if f in cen]
C = np.stack([cen[f] for f in have])
D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
np.fill_diagonal(D, np.inf)
Ds = np.sort(D, axis=1)
d_mean5 = Ds[:, :5].mean(1)
n_within_05 = (D <= 0.5).sum(1).astype(float)
llv_h = np.array([S[f]["log_lapvar"] for f in have])
reb_h = np.array([S[f]["reblur"] for f in have])
fr_h = np.array(have, float)
print(f"  n with poses {len(have)}/220;  d_mean5 med {np.median(d_mean5):.3f}")
for nm, v in (("log_lapvar", llv_h), ("reblur", reb_h)):
    print(f"  {nm:11s} vs d_mean5     spearman {spearman(v, d_mean5):+.3f}"
          f"   | frame-index-controlled pearson {partial(v, d_mean5, fr_h):+.3f}")
    print(f"  {nm:11s} vs n_within_0.5 spearman {spearman(v, n_within_05):+.3f}")
# speed proxy: distance to the temporally adjacent frame
spd = []
for i, f in enumerate(have):
    nb = [g for g in (f - 10, f + 10) if g in cen]
    spd.append(np.mean([np.linalg.norm(cen[f] - cen[g]) for g in nb]) if nb else np.nan)
spd = np.array(spd)
ok = ~np.isnan(spd)
print(f"  log_lapvar vs camera speed(+-10) spearman {spearman(llv_h[ok], spd[ok]):+.3f}")
print(f"  reblur     vs camera speed(+-10) spearman {spearman(reb_h[ok], spd[ok]):+.3f}")

# ------------------------------------------------------------------ Q2 hard mask coverage
print()
print("=" * 78)
print("Q2  temporal coverage damage from a HARD top-k sharpness mask")
print("=" * 78)


def gapstats(frames):
    f = np.sort(np.asarray(frames))
    g = np.diff(f)
    return dict(n=len(f), max_gap=int(g.max()), p95_gap=float(np.percentile(g, 95)),
                n_gap_ge30=int((g >= 30).sum()), n_gap_ge50=int((g >= 50).sum()),
                mean_gap=float(g.mean()))


print(f"  {'subset':28s} {'n':>4} {'maxgap':>7} {'p95':>6} {'>=30':>5} {'>=50':>5} {'mean_stat':>10}")
print(f"  {'ALL train_sub':28s} " + "{n:4d} {max_gap:7d} {p95_gap:6.0f} {n_gap_ge30:5d} "
      "{n_gap_ge50:5d}".format(**gapstats(SUB)) + f" {reb.mean():10.4f}")
for stat, v in (("reblur", reb), ("log_lapvar", llv)):
    for frac in (0.75, 0.5, 0.25):
        k = int(round(frac * len(SUB)))
        idx = np.argsort(-v)[:k]
        sel = [SUB[i] for i in sorted(idx)]
        gs = gapstats(sel)
        print(f"  {'top-%d%% by %s' % (frac * 100, stat):28s} "
              "{n:4d} {max_gap:7d} {p95_gap:6.0f} {n_gap_ge30:5d} {n_gap_ge50:5d}".format(**gs)
              + f" {v[idx].mean():10.4f}")

# ------------------------------------------------------------------ Q3 matched subsets
print()
print("=" * 78)
print("Q3  coverage-MATCHED pair-stratified subsets (the decisive gate arm)")
print("=" * 78)
rng = np.random.RandomState(42)
pairs = [SUB[i:i + 2] for i in range(0, len(SUB), 2)]
pairs = [p for p in pairs if len(p) == 2]
sel = {"sharp": [], "blurry": [], "rand": []}
seps = []
for a, b in pairs:
    sa, sb = S[a]["reblur"], S[b]["reblur"]
    hi, lo = (a, b) if sa >= sb else (b, a)
    sel["sharp"].append(hi)
    sel["blurry"].append(lo)
    sel["rand"].append(a if rng.rand() < 0.5 else b)
    seps.append(abs(sa - sb))
for k in sel:
    v = np.array([S[f]["reblur"] for f in sel[k]])
    l = np.array([S[f]["log_lapvar"] for f in sel[k]])
    gs = gapstats(sel[k])
    print(f"  {k:7s} n={gs['n']:3d} maxgap {gs['max_gap']:3d} meangap {gs['mean_gap']:5.1f}  "
          f"reblur mean {v.mean():.4f} sd {v.std():.4f} | log_lapvar mean {l.mean():+.4f}")
dsh = np.array([S[f]["reblur"] for f in sel["sharp"]]).mean()
dbl = np.array([S[f]["reblur"] for f in sel["blurry"]]).mean()
lsh = np.array([S[f]["log_lapvar"] for f in sel["sharp"]]).mean()
lbl = np.array([S[f]["log_lapvar"] for f in sel["blurry"]]).mean()
print(f"  SEPARATION achieved: d(reblur) {dsh - dbl:+.4f} "
      f"({(dsh - dbl) / reb.std():+.2f} sd)   d(log_lapvar) {lsh - lbl:+.4f} nats "
      f"= {lsh - lbl:+.2f} px equivalent sigma (A2 calibration 1.00 nats/px)")
print(f"  within-pair |d reblur| median {np.median(seps):.4f} (frames 10 apart)")
# camera-center coverage really is matched?
for k in sel:
    hv = [f for f in sel[k] if f in cen]
    Ck = np.stack([cen[f] for f in hv])
    Dk = np.linalg.norm(Ck[:, None] - Ck[None], axis=-1)
    np.fill_diagonal(Dk, np.inf)
    print(f"  {k:7s} coverage: d_nearest med {np.median(np.sort(Dk,1)[:,0]):.4f}  "
          f"d_mean5 med {np.median(np.sort(Dk,1)[:,:5].mean(1)):.4f}")
for k, v in sel.items():
    with open(f"{HERE}/subsets/pair_{k}.txt", "w") as fh:
        for f in sorted(v):
            fh.write(f"frame_{f:06d}.jpg\n")
    print(f"  wrote {HERE}/subsets/pair_{k}.txt")

# ------------------------------------------------------------------ Q4 weights / sigmas
print()
print("=" * 78)
print("Q4  what the weight and blur-sigma parameterisations actually produce")
print("=" * 78)


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


print("  soft power-law  w = (s/median(s))^p, renormalised to mean 1")
for stat, v, lin in (("reblur", reb, True), ("log_lapvar", llv, False)):
    s = v if lin else np.exp(v)
    for p in (0.5, 1.0, 2.0, 4.0):
        w = (s / np.median(s)) ** p
        w = w / w.mean()
        print(f"    {stat:11s} p={p:<4} min {w.min():.3f} p10 {np.percentile(w,10):.3f} "
              f"p90 {np.percentile(w,90):.3f} max {w.max():.3f}  CV {w.std():.3f}  "
              f"ESS {ess(w):.1f}/220 ({100*ess(w)/220:.0f}%)")
print("  hard top-k mask (w = 1/frac on kept, 0 else)")
for frac in (0.75, 0.5, 0.25):
    print(f"    keep {frac:.2f}: ESS = {frac*220:.0f}/220 ({100*frac:.0f}%), "
          f"and see Q2 for the coverage holes it opens")

print()
print("  per-view blur sigma init, sigma_i = clip(c*(llv_ref - llv_i), lo, hi),")
print("  ref = p95 sharpest train frame, A2 calibration c = 1.00 px per nat")
ref = np.percentile(llv, 95)
for c in (1.0,):
    for lo, hi in ((0.0, 2.5), (0.2, 2.5)):
        sg = np.clip(c * (ref - llv), lo, hi)
        print(f"    c={c} clip[{lo},{hi}]: min {sg.min():.3f} p10 {np.percentile(sg,10):.3f} "
              f"med {np.median(sg):.3f} p90 {np.percentile(sg,90):.3f} max {sg.max():.3f} "
              f"mean {sg.mean():.3f}")
sg = np.clip(1.0 * (ref - llv), 0.2, 2.5)
print(f"  -> {100*(sg>=2.49).mean():.1f}% of views pinned at the 2.5px cap, "
      f"{100*(sg<=0.21).mean():.1f}% at the floor")

# ---------------------------------------------------------------- sidecar for trainer
out = f"{HERE}/bonsai_sharp_sidecar.csv"
with open(out, "w") as fh:
    fh.write("name,frame,log_lapvar,reblur,log_hf025,sigma_init\n")
    for f in sorted(S):
        s = np.clip(1.0 * (ref - S[f]["log_lapvar"]), 0.2, 2.5)
        fh.write(f"frame_{f:06d}.jpg,{f},{S[f]['log_lapvar']:.6f},{S[f]['reblur']:.6f},"
                 f"{S[f]['log_hf025']:.6f},{s:.4f}\n")
print(f"\n  wrote {out}  (248 rows; trainer reads it by image name)")
