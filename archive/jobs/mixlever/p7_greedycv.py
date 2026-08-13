"""Phase 7: cross-validate the SUBSET SELECTION itself.

Greedy forward selection is re-run independently on fold A views only and on fold B
views only; each fold's selected subset is then scored on the OTHER fold. Compares
against (a) uniform over the shipped UT4 and (b) the rank-ordered subset, both scored
on the same held-out views. This is the honest number for "greedy re-membering".
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool as P

HERE = os.path.dirname(os.path.abspath(__file__))
KMAX = 7
GPUS = tuple(int(x) for x in os.environ.get("GPUS", "0,1").split(","))


def expand(idxs):
    v = np.zeros(21); v[list(idxs)] = 1.0 / len(idxs); return v


def main():
    t0 = time.time()
    N = P.names()[:21]
    p1 = json.load(open(os.path.join(HERE, "p1.json")))
    solo = {n: P.agg(p1[n])[0] for n in N}
    dist = [i for i, n in enumerate(N) if n != "sh3"]
    FA = list(range(0, 60, 2)); FB = list(range(1, 60, 2))
    UT = [N.index(x) for x in ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]]

    pl = P.Pool(gpus=GPUS)
    out = {}
    for fname, ftr in (("A", FA), ("B", FB)):
        fte = FB if fname == "A" else FA
        S, rem, chain = [], list(dist), []
        while len(S) < KMAX:
            cands = {f"g_{i}": expand(S + [i]) for i in rem}
            r = pl.score(cands)
            best, bi = max((P.agg(r[f"g_{i}"], ftr)[0], i) for i in rem)
            S.append(bi); rem.remove(bi)
            chain.append({"k": len(S), "added": N[bi], "train_score": best,
                          "test_score": P.agg(r[f"g_{bi}"], fte)[0]})
            print(f"fold{fname} k={len(S)} +{N[bi]:16s} train {best:.4f}  test {chain[-1]['test_score']:.4f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            out[f"fold{fname}"] = chain
            json.dump(out, open(os.path.join(HERE, "p7.json"), "w"), indent=1)

        kstar = max(range(1, KMAX + 1), key=lambda k: chain[k - 1]["train_score"])
        out[f"fold{fname}_kstar"] = kstar
        # reference points on the SAME held-out views
        order = sorted(dist, key=lambda i: -solo[N[i]])
        ref = pl.score({"UT4": expand(UT), "rank7": expand(order[:7]),
                        "greedySel": expand(S[:kstar])})
        out[f"fold{fname}_ref"] = {k: P.agg(v, fte)[0] for k, v in ref.items()}
        print(f"fold{fname}: k*={kstar} set={[N[i] for i in S[:kstar]]}", flush=True)
        print(f"fold{fname} HELD-OUT: greedy {out[f'fold{fname}_ref']['greedySel']:.4f}  "
              f"rank7 {out[f'fold{fname}_ref']['rank7']:.4f}  UT4 {out[f'fold{fname}_ref']['UT4']:.4f}", flush=True)
        json.dump(out, open(os.path.join(HERE, "p7.json"), "w"), indent=1)
    pl.close()

    g = np.mean([out["foldA_ref"]["greedySel"], out["foldB_ref"]["greedySel"]])
    r7 = np.mean([out["foldA_ref"]["rank7"], out["foldB_ref"]["rank7"]])
    u = np.mean([out["foldA_ref"]["UT4"], out["foldB_ref"]["UT4"]])
    print(f"\nOUT-OF-FOLD MEAN: greedy-selected {g:.4f} | rank-ordered-7 {r7:.4f} | shipped UT4 {u:.4f}")
    print(f"  greedy vs UT4  {g-u:+.4f}\n  rank7  vs UT4  {r7-u:+.4f}\n  greedy vs rank7 {g-r7:+.4f}")
    print("elapsed %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
