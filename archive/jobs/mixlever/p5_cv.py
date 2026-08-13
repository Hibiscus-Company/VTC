"""Phase 5: honest CV of everything that has a fitted parameter, with per-view metrics
saved so any fold split can be evaluated post hoc.

Fitted things under test:
  (a) the ensemble DEPTH k in the rank-ordered curve
  (b) the greedy-selected SUBSET
  (c) the cross-family LS weights
All are selected on fold A and scored on fold B (and vice versa).
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool as P

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "p2_gram.npz"), allow_pickle=True)
GRAM, XTY = D["gram"], D["xty"]
N = P.names()[:21]
FA = list(range(0, 60, 2)); FB = list(range(1, 60, 2))


def expand(idxs, w):
    v = np.zeros(21); v[list(idxs)] = w; return v


def fit(idxs, views):
    G = GRAM[views][:, idxs][:, :, idxs].mean(0); b = XTY[views][:, idxs].mean(0); k = len(idxs)
    r = minimize(lambda w: w @ G @ w - 2 * w @ b, np.ones(k) / k,
                 jac=lambda w: 2 * (G @ w - b), method="SLSQP", bounds=[(0, 1)] * k,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                               "jac": lambda w: np.ones(k)}],
                 options={"maxiter": 800, "ftol": 1e-15})
    w = np.clip(r.x, 0, None); return w / w.sum()


def main():
    t0 = time.time()
    p3 = json.load(open(os.path.join(HERE, "p3.json")))
    order = [N.index(x) for x in p3["A_order"]]
    greedy = p3["C_greedy"]
    gsel = [N.index(c["added"]) for c in greedy]
    dist = [i for i, n in enumerate(N) if n != "sh3"]
    UT = [N.index(x) for x in ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]]

    cands = {"UT4_shipped": expand(UT, np.ones(4) / 4)}
    for k in range(2, 13):
        cands[f"rank_k{k}"] = expand(order[:k], np.ones(k) / k)
        cands[f"greedy_k{k}"] = expand(gsel[:k], np.ones(k) / k)
    # LS weights on the greedy-best subset, fitted per fold and on all views
    kbest = max(range(1, len(greedy) + 1), key=lambda k: greedy[k - 1]["score"])
    GB = gsel[:kbest]
    cands["greedyBest_unif"] = expand(GB, np.ones(len(GB)) / len(GB))
    cands["greedyBest_lsA"] = expand(GB, fit(GB, FA))
    cands["greedyBest_lsB"] = expand(GB, fit(GB, FB))
    cands["greedyBest_lsALL"] = expand(GB, fit(GB, list(range(60))))
    # LS over ALL 20 distinct members (LS does its own selection)
    cands["all20_lsA"] = expand(dist, fit(dist, FA))
    cands["all20_lsB"] = expand(dist, fit(dist, FB))
    cands["all20_lsALL"] = expand(dist, fit(dist, list(range(60))))
    meta = {k: {N[i]: round(float(v[i]), 4) for i in range(21) if v[i] > 1e-4} for k, v in cands.items()}

    pl = P.Pool()
    res = pl.score(cands)
    pl.close()
    json.dump({"perview": res, "meta": meta, "kbest": kbest,
               "greedy_order": [N[i] for i in gsel]},
              open(os.path.join(HERE, "p5.json"), "w"))

    def sc(n, v=None): return P.agg(res[n], v)[0]

    print("\n=== depth k selected on one fold, scored on the other (rank-ordered)")
    for tag, ord_ in (("rank", "rank"), ("greedy", "greedy")):
        kA = max(range(2, 13), key=lambda k: sc(f"{ord_}_k{k}", FA))
        kB = max(range(2, 13), key=lambda k: sc(f"{ord_}_k{k}", FB))
        oo = (sc(f"{ord_}_k{kA}", FB) + sc(f"{ord_}_k{kB}", FA)) / 2
        ins = max(sc(f"{ord_}_k{k}") for k in range(2, 13))
        print(f"{tag:7s} k*(A)={kA} k*(B)={kB}  out-of-fold {oo:.4f}  in-sample-best {ins:.4f}")

    print("\n=== full curve, all 60 views")
    print(f"{'k':>3s} {'rank':>9s} {'greedy':>9s}")
    for k in range(2, 13):
        print(f"{k:3d} {sc(f'rank_k{k}'):9.4f} {sc(f'greedy_k{k}'):9.4f}")

    print("\n=== LS weights vs uniform, out of fold (full competition metric)")
    for base, unif in (("greedyBest", "greedyBest_unif"), ("all20", None)):
        uA = sc(unif, FA) if unif else sc("rank_k20" if False else "greedyBest_unif", FA)
        uB = sc(unif, FB) if unif else sc("greedyBest_unif", FB)
        oA = sc(f"{base}_lsB", FA); oB = sc(f"{base}_lsA", FB)
        print(f"{base:12s} unifA {uA:.4f} fitB->A {oA:.4f} ({oA-uA:+.4f}) | "
              f"unifB {uB:.4f} fitA->B {oB:.4f} ({oB-uB:+.4f}) | "
              f"CV mean {((oA-uA)+(oB-uB))/2:+.4f} | in-sample {sc(base+'_lsALL'):.4f}")
    print("\nweights:")
    for k in ["greedyBest_unif", "greedyBest_lsALL", "all20_lsALL", "all20_lsA", "all20_lsB"]:
        print(" ", k, meta[k])
    print("elapsed %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
