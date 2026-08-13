"""Final cross-pool synthesis of the AGGREGATOR lever.
Emits (a) the replication matrix for the k=7 candidates, (b) paired-bootstrap CIs.
"""
import json, sys
import numpy as np

T = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/agg_%s.json"
RNG = np.random.default_rng(0)


def cpv(a):
    return 100 * (0.4 * (1 - a[:, 2]) + 0.3 * a[:, 1] + 0.3 * np.minimum(a[:, 0] / 50.0, 1.0))


def load(p):
    d = json.load(open(T % p))
    return {k: np.asarray(v) for k, v in d["per_view"].items()}, d["variants"]


def ci(d, B=20000):
    n = len(d)
    idx = RNG.integers(0, n, size=(B, n))
    bs = d[idx].mean(1)
    return np.percentile(bs, [2.5, 97.5])


def main():
    pools = sys.argv[1:]
    # ---- (a) replication matrix ----
    cands = ["winsor1", "trim1", "winsor2", "blend(a=0.75)", "blend(a=0.5)",
             "huber(c=4.0)", "median", "wex(t=1.5)", "wex(t=2.0)"]
    data = {}
    for p in pools:
        pv, var = load(p)
        data[p] = (pv, var)
    print(f"{'aggregator':<16}" + "".join(f"{p:>12}" for p in pools))
    print(f"{'  [k]':<16}" + "".join(f"{len(data[p][1]):>12}" for p in pools))
    print(f"{'  mean baseline':<16}" + "".join(
        f"{cpv(data[p][0]['mean']).mean():>12.4f}" for p in pools))
    print("-" * (16 + 12 * len(pools)))
    for c in cands:
        if not any(c in data[p][0] for p in pools):
            continue
        cells = []
        for p in pools:
            pv = data[p][0]
            if c not in pv:
                cells.append(f"{'--':>12}")
                continue
            d = cpv(pv[c]) - cpv(pv["mean"])
            lo, hi = ci(d)
            mark = "*" if lo > 0 else ("x" if hi < 0 else " ")
            cells.append(f"{d.mean():>+11.4f}{mark}")
        print(f"{c:<16}" + "".join(cells))
    print("\n* = 95% paired-bootstrap CI strictly above 0 ; x = strictly below 0")

    # ---- (b) pooled test across the CLEAN k=7 pools ----
    clean = [p for p in pools if "OUT" not in p]
    if len(clean) > 1:
        print(f"\n--- REPLICATION over clean pools {clean} ---")
        print("CAVEAT: these pools share the SAME 60 views (same scene, same poses), so the")
        print("per-view deltas are correlated ACROSS pools. Concatenating views therefore")
        print("OVERSTATES precision. The honest unit of replication is the POOL, so the")
        print("column that matters is the per-pool sign, not the pooled CI.")
        for c in cands:
            ds = [cpv(data[p][0][c]) - cpv(data[p][0]["mean"])
                  for p in clean if c in data[p][0]]
            if len(ds) != len(clean):
                continue
            d = np.concatenate(ds)
            lo, hi = ci(d)
            means = [x.mean() for x in ds]
            npos = sum(m > 0 for m in means)
            # paired across pools: does the SAME view gain in every pool?
            V = np.stack(ds, 0)                      # (pools, views)
            allpos = int((V > 0).all(0).sum())
            print(f"  {c:<16}avg{np.mean(means):+8.4f}  [pooled CI {lo:+.4f},{hi:+.4f}]  "
                  f"sign {npos}/{len(means)}  views+in-all {allpos}/{V.shape[1]}  "
                  f"per-pool: " + " ".join(f"{m:+.4f}" for m in means))


if __name__ == "__main__":
    main()
