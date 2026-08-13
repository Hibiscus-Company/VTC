#!/usr/bin/env python
"""Closed-form family-weighting analysis off the cached residual Gram. No GPU."""
import numpy as np, itertools, json, sys

Z = np.load("/home/bkai/.claude/jobs/1c9cf7e9/tmp/wt_gram.npz")
G, names = Z["G"], list(Z["names"])
psnr, lp = Z["psnr"], Z["lpips"]
C = G.mean(0)
K = len(names)
idx = {n: i for i, n in enumerate(names)}

# ---- duplicates -------------------------------------------------------------
d = np.sqrt(np.diag(C))
corr = C / np.outer(d, d)
print("PAIRS with residual corr > 0.995 (possible duplicate renders):")
for i in range(K):
    for j in range(i + 1, K):
        if corr[i, j] > 0.995:
            print(f"   {names[i]:>18} ~ {names[j]:<18} corr={corr[i,j]:.6f}")

UT = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
print("\nresidual corr WITHIN the UT family (the production majority family):")
for a, b in itertools.combinations(UT, 2):
    print(f"   {a:>16} ~ {b:<16} {corr[idx[a], idx[b]]:.4f}")

CAND = [n for n in names if n not in UT]
print("\ncandidate 2nd families: mean corr to the UT family, solo rmse")
rows = []
for n in CAND:
    c = np.mean([corr[idx[n], idx[u]] for u in UT])
    rows.append((c, n, np.sqrt(C[idx[n], idx[n]] / 3), psnr[:, idx[n]].mean(), lp[:, idx[n]].mean()))
for c, n, r, p, l in sorted(rows):
    print(f"   {n:>18} corr_UT={c:.4f} rmse={r:.5f} psnr={p:7.3f} lpips={l:.4f}")


# ---- weighting machinery ----------------------------------------------------
def mse(w, Cm):
    w = np.asarray(w, float)
    return float(w @ Cm @ w) / 3.0


def ls_w(Cm, ridge=0.0):
    n = Cm.shape[0]
    A = Cm + ridge * np.trace(Cm) / n * np.eye(n)
    v = np.linalg.solve(A, np.ones(n))
    return v / v.sum()


def sub(members, S=None):
    ii = [idx[m] for m in members]
    Gs = G[S] if S is not None else G
    return Gs.mean(0)[np.ix_(ii, ii)]


def report(members, fam, label, ridge=0.0):
    """fam: list of family-id per member. Sweep family weight, LS weights, 2-fold CV."""
    Cm = sub(members)
    n = len(members)
    print(f"\n{'='*88}\n{label}\n  members: " +
          ", ".join(f"{m}[{f}]" for m, f in zip(members, fam)))
    nA = fam.count(0); nB = fam.count(1)
    A = np.array([1.0 / nA if f == 0 else 0.0 for f in fam])
    B = np.array([1.0 / nB if f == 1 else 0.0 for f in fam])
    uni = nB / n
    print(f"  family sizes {nA}/{nB}; uniform-per-member gives family B weight {uni:.4f}")
    print(f"  {'wB':>7} {'MSE*1e4':>9} {'PSNR_eq':>8} {'vs uniform dPSNR':>17}")
    grid = np.linspace(0, 1, 101)
    m = np.array([mse((1 - w) * A + w * B, Cm) for w in grid])
    base = mse((1 - uni) * A + uni * B, Cm)
    for w in [0.0, 0.10, 0.1667, 0.20, 0.25, 0.3333, 0.40, 0.50, 0.60, 1.0]:
        v = mse((1 - w) * A + w * B, Cm)
        print(f"  {w:7.4f} {v*1e4:9.5f} {10*np.log10(1/v):8.4f} "
              f"{10*np.log10(base/v):+17.4f}")
    wopt = grid[m.argmin()]
    print(f"  ARGMIN wB = {wopt:.3f}  (uniform-per-member = {uni:.4f}), "
          f"dPSNR vs uniform {10*np.log10(base/m.min()):+.4f} dB")
    # analytic 2-family optimum from the 2x2 blend covariance
    caa, cbb, cab = mse(A, Cm), mse(B, Cm), float(A @ Cm @ B) / 3
    wstar = (caa - cab) / (caa + cbb - 2 * cab)
    print(f"  analytic 2-family optimum wB* = {wstar:.4f}  "
          f"(rmseA={np.sqrt(caa):.5f} rmseB={np.sqrt(cbb):.5f} corrAB={cab/np.sqrt(caa*cbb):.4f})")

    # LS-optimal per-member weights, in-sample and honest 2-fold CV
    w_ls = ls_w(Cm, ridge)
    print(f"  LS-optimal per-member weights (in-sample): " +
          " ".join(f"{m_:s}={w_:.3f}" for m_, w_ in zip(members, w_ls)))
    print(f"    in-sample dPSNR vs uniform-per-member: "
          f"{10*np.log10(base/mse(w_ls, Cm)):+.4f} dB   "
          f"vs the {uni:.3f}-family blend baseline")
    rng = np.random.RandomState(0)
    accs = {"LS": [], "FAMwopt": [], "UNI": []}
    for rep in range(20):
        perm = rng.permutation(G.shape[0])
        for f in range(2):
            te = perm[f::2]; tr = perm[1 - f::2]
            Ctr, Cte = sub(members, tr), sub(members, te)
            wl = ls_w(Ctr, ridge)
            mtr = np.array([mse((1 - w) * A + w * B, Ctr) for w in grid])
            wf = grid[mtr.argmin()]
            accs["LS"].append(mse(wl, Cte))
            accs["FAMwopt"].append(mse((1 - wf) * A + wf * B, Cte))
            accs["UNI"].append(mse((1 - uni) * A + uni * B, Cte))
    u = np.mean(accs["UNI"])
    print(f"  HONEST 2-fold CV (20 repeats), dPSNR vs uniform-per-member:")
    for k in ("LS", "FAMwopt"):
        print(f"    {k:>8}: {10*np.log10(u/np.mean(accs[k])):+.4f} dB")
    return wopt, uni, wstar


# CONTROL: same-family split -- two halves of the UT family. Optimum MUST be 0.5.
report(UT, [0, 0, 1, 1], "CONTROL A: same-family 2+2 split of the UT pool (truth: wB=0.5)")
report(UT, [0, 0, 0, 1], "CONTROL B: 3 UT + 1 UT (truth: wB=0.25)")
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/wt_corr.npy", corr)
json.dump({"names": names}, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/wt_names.json", "w"))
