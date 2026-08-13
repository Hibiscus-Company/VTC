"""Phase 3: how far does ensembling go, and does adding WEAKER members help?

A) rank-ordered k-curve: add distinct members in order of their solo score, k=1..20.
B) add-one-to-production-base: base = shipped UT4 (=k4), add each other distinct
   member at equal weight (1/5) and at down-weight 0.15. Directly tests the
   "only ensemble members within ~0.15 of the best" rule of thumb.
C) greedy forward selection by FULL SCORE, k=1..20 -> the optimal-subset ceiling.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool as P

HERE = os.path.dirname(os.path.abspath(__file__))
# sh3 is byte-identical to gsplatB11ut60k -> drop sh3 (index 20)
DUP = "sh3"


def w_of(idxs, weights=None, n=21):
    w = np.zeros(n)
    if weights is None:
        w[list(idxs)] = 1.0 / len(idxs)
    else:
        for i, x in zip(idxs, weights):
            w[i] += x
    return w


def main():
    t0 = time.time()
    N = P.names()[:21]
    p1 = json.load(open(os.path.join(HERE, "p1.json")))
    solo = {n: P.agg(p1[n])[0] for n in N}
    distinct = [i for i, n in enumerate(N) if n != DUP]
    order = sorted(distinct, key=lambda i: -solo[N[i]])
    print("solo ranking:", [(N[i], round(solo[N[i]], 4)) for i in order], flush=True)

    pl = P.Pool()
    out = {}

    # ---- A: rank-ordered k-curve
    cands = {}
    for k in range(1, len(order) + 1):
        cands[f"rank_k{k}"] = w_of(order[:k])
    resA = pl.score(cands)
    out["A_rank"] = {k: P.agg(v) for k, v in resA.items()}
    out["A_order"] = [N[i] for i in order]
    json.dump(out, open(os.path.join(HERE, "p3.json"), "w"))
    print("\n=== A: rank-ordered k-curve  (%.0fs)" % (time.time() - t0), flush=True)
    for k in range(1, len(order) + 1):
        sc, ps, ss, lp = out["A_rank"][f"rank_k{k}"]
        print(f"k={k:2d} +{N[order[k-1]]:16s} score {sc:8.4f}  PSNR {ps:7.4f} SSIM {ss:.4f} LPIPS {lp:.4f}", flush=True)

    # ---- B: add-one to the shipped production base (UT4 = k4)
    UT = [N.index(x) for x in ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]]
    cands = {"base_UT4": w_of(UT)}
    for i in distinct:
        if i in UT:
            continue
        cands[f"add5_{N[i]}"] = w_of(UT + [i])                                   # equal 1/5
        cands[f"add15_{N[i]}"] = w_of(UT + [i], [0.85 / 4] * 4 + [0.15])         # newcomer at 0.15
    resB = pl.score(cands)
    out["B_addone"] = {k: P.agg(v) for k, v in resB.items()}
    json.dump(out, open(os.path.join(HERE, "p3.json"), "w"))
    b0 = out["B_addone"]["base_UT4"][0]
    print("\n=== B: add-one to UT4 base (base=%.4f)  (%.0fs)" % (b0, time.time() - t0), flush=True)
    print(f"{'member':18s} {'solo':>8s} {'gap':>7s} {'d(w=1/5)':>9s} {'d(w=.15)':>9s}", flush=True)
    bestsolo = max(solo[N[i]] for i in distinct)
    for i in sorted([j for j in distinct if j not in UT], key=lambda j: -solo[N[j]]):
        d5 = out["B_addone"][f"add5_{N[i]}"][0] - b0
        d15 = out["B_addone"][f"add15_{N[i]}"][0] - b0
        print(f"{N[i]:18s} {solo[N[i]]:8.4f} {solo[N[i]]-bestsolo:7.4f} {d5:9.4f} {d15:9.4f}", flush=True)

    # ---- C: greedy forward selection by full score
    S, chain = [], []
    remaining = list(distinct)
    while remaining:
        cands = {f"g_{i}": w_of(S + [i]) for i in remaining}
        r = pl.score(cands)
        scored = sorted(((P.agg(r[f'g_{i}'])[0], i) for i in remaining), reverse=True)
        best, bi = scored[0]
        S.append(bi); remaining.remove(bi)
        chain.append({"k": len(S), "added": N[bi], "score": best,
                      "full": P.agg(r[f'g_{bi}']),
                      "round": [(N[i], s) for s, i in scored]})
        out["C_greedy"] = chain
        json.dump(out, open(os.path.join(HERE, "p3.json"), "w"))
        print(f"greedy k={len(S):2d} +{N[bi]:17s} {best:8.4f}   (%.0fs)" % (time.time() - t0), flush=True)
    pl.close()
    print("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
