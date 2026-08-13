"""Cross-family weight fitting with proper 2-fold CV.

Protocol: enumerate a FIXED candidate set of weight vectors, score every one on ALL 60 views
with per-image metrics, then do all CV bookkeeping offline.  Selection on fold A, report on
fold B (and vice versa).  This is honest: the candidate set is fixed a priori, and the held-out
half never influences which weight vector is picked.
Usage: python s9_weights.py <tag> m1 m2 m3 ...
"""
import os, sys, json, time, itertools, numpy as np, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens")
import harnlm as H
from scipy.optimize import nnls
torch.backends.cudnn.benchmark = True

TAG = sys.argv[1]
MEM = sys.argv[2:]
k = len(MEM)
H.init()

FAM = {"e15ceil95": "FastGS", "e16app": "FastGS", "e17visnorm": "FastGS",
       "gsplatB1": "MCMC", "gsplatB2": "MCMC", "gsplatB3": "MCMC", "gsplatB4warm": "MCMC",
       "gsplatB5affine": "MCMC", "gsplatB6bilagrid": "MCMC", "gsplatB7ppisp2": "MCMC",
       "gsplatB8pure": "MCMC",
       "gsplatB9ut": "UT", "gsplatB10ut8M": "UT", "gsplatB11ut60k": "UT", "gsplatB12ut8Ms7": "UT",
       "m31b_nolpips": "UTmetric", "m31b_taillpips": "UTmetric",
       "sh0": "UTsh", "sh1": "UTsh", "sh2": "UTsh"}
fams = sorted(set(FAM[m] for m in MEM))
print(f"{TAG}: {k} members, families {fams}", flush=True)

# ---- NNLS-MSE weights from Gram, per fold ----
Z = np.load(os.path.join(H.OUT, "gram.npz"), allow_pickle=True)
A_all, b_all = Z["A"], Z["b"]
NA = list(Z["names"]); j = [NA.index(m) for m in MEM]


def nnls_w(imgs):
    A = A_all[np.ix_(imgs, j, j)].sum(0); b = b_all[np.ix_(imgs, j)].sum(0)
    A = A + np.eye(k) * (np.trace(A) / k) * 1e-10
    L = np.linalg.cholesky(A)
    w, _ = nnls(L.T, np.linalg.solve(L, b))
    return w / w.sum() if w.sum() > 0 else np.ones(k) / k


rng = np.random.RandomState(0)
perm = rng.permutation(60)
SPLITS = [(sorted(perm[:30].tolist()), sorted(perm[30:].tolist()))]
rng2 = np.random.RandomState(7)
p2 = rng2.permutation(60)
SPLITS.append((sorted(p2[:30].tolist()), sorted(p2[30:].tolist())))
ALL = list(range(60))

# ---------------- candidate weight vectors ----------------
cands = {}
u = np.ones(k) / k
cands["uniform"] = u
cands["nnls_all60"] = nnls_w(ALL)
for si, (a, b_) in enumerate(SPLITS):
    cands[f"nnls_split{si}A"] = nnls_w(a)
    cands[f"nnls_split{si}B"] = nnls_w(b_)

# family-weight grid (total mass per family, split evenly inside family)
fidx = {f: [i for i, m in enumerate(MEM) if FAM[m] == f] for f in fams}
if len(fams) == 2:
    grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for g in grid:
        w = np.zeros(k)
        w[fidx[fams[0]]] = g / len(fidx[fams[0]])
        w[fidx[fams[1]]] = (1 - g) / len(fidx[fams[1]])
        cands[f"fam:{fams[0]}={g:.2f}"] = w
elif len(fams) >= 3:
    # bounded family-mass grid: uniform-family baseline + structured perturbations
    nf = len(fams)
    base_mass = np.array([len(fidx[f]) for f in fams], dtype=float)
    base_mass /= base_mass.sum()                      # = member-uniform
    eq = np.ones(nf) / nf                             # = family-uniform
    masses = [base_mass, eq]
    for mult in (0.3, 0.5, 0.7, 1.5, 2.0, 3.0):       # scale ONE family, renormalise
        for fi in range(nf):
            m = base_mass.copy(); m[fi] *= mult; masses.append(m / m.sum())
            m2 = eq.copy(); m2[fi] *= mult; masses.append(m2 / m2.sum())
    r = np.random.RandomState(11)
    for _ in range(16):
        masses.append(r.dirichlet(np.ones(nf) * 4))
    for mass in masses:
        w = np.zeros(k)
        for f, mm in zip(fams, mass):
            w[fidx[f]] = mm / len(fidx[f])
        cands["fam:" + ",".join(f"{f}={m:.2f}" for f, m in zip(fams, mass))] = w

# single-member up/down weighting (detects "one member deserves 1.5x")
for i, m in enumerate(MEM):
    for mult in (0.0, 0.4, 0.7, 1.5, 2.0, 3.0):
        w = np.ones(k); w[i] = mult
        cands[f"solo:{m}x{mult}"] = w / w.sum()

# Dirichlet random samples around uniform
for alpha, n in ((20.0, 12), (6.0, 12)):
    r = np.random.RandomState(int(alpha * 100))
    for t in range(n):
        cands[f"dir{alpha:g}_{t}"] = r.dirichlet(np.ones(k) * alpha)

print(f"{len(cands)} candidate weight vectors -> ~{len(cands)*25/60:.0f} min", flush=True)

t0 = time.time()
per = {}
for ci, (name, w) in enumerate(cands.items()):
    d = H.score(MEM, weights=w, per_image=True)
    per[name] = dict(w=list(map(float, np.asarray(w) / np.sum(w))), **{kk: d[kk] for kk in ("psnr", "ssim", "lpips", "score")},
                     per=d["per"])
    print(f"[{ci+1}/{len(cands)}] {name:34s} SCORE {d['score']:8.4f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(per, open(os.path.join(H.OUT, f"weights_{TAG}.json"), "w"))
print("SCORING DONE", time.time() - t0, flush=True)

# ---------------- CV bookkeeping ----------------


def sub(name, imgs):
    return H.score_subset(per[name]["per"], imgs)["score"]


print("\n=== 2-FOLD CV: does a fitted weight vector beat uniform out of sample? ===", flush=True)
NAMES_ALL = list(cands.keys())
FAMONLY = [n for n in NAMES_ALL if n.startswith("fam:")] + ["uniform"]
NNLSONLY = [n for n in NAMES_ALL if n.startswith("nnls")] + ["uniform"]
for label, pool in [("ALL candidates", NAMES_ALL), ("family-grid only", FAMONLY),
                    ("NNLS-MSE only", NNLSONLY)]:
    gains = []
    for si, (a, b_) in enumerate(SPLITS):
        for tr, te in ((a, b_), (b_, a)):
            pick = max(pool, key=lambda n: sub(n, tr))
            g = sub(pick, te) - sub("uniform", te)
            gains.append(g)
            print(f"  {label:18s} split{si} fit->{pick:32s} heldout delta {g:+.4f}", flush=True)
    print(f"  {label:18s} MEAN CV GAIN {np.mean(gains):+.4f}  (folds {np.round(gains,4)})\n", flush=True)

best_all = max(NAMES_ALL, key=lambda n: per[n]["score"])
print(f"in-sample best-on-all-60: {best_all} = {per[best_all]['score']:.4f} "
      f"(uniform {per['uniform']['score']:.4f}, in-sample gain {per[best_all]['score']-per['uniform']['score']:+.4f})", flush=True)
print("  w =", ", ".join(f"{m}:{x:.3f}" for m, x in zip(MEM, per[best_all]["w"])), flush=True)
