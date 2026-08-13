"""Phase 4: cross-family non-negative weight fitting with 2-fold CV.

Weights are fitted on fold A views (pixel least squares to real test GT, w>=0,
sum w = 1), then the resulting mix is SCORED with the full competition metric on
fold B views, and vice versa. Uniform over the same member set is the control.
The honest number is the out-of-fold score minus the out-of-fold uniform score.
"""
import os, sys, json, time, itertools
import numpy as np
from scipy.optimize import nnls, minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool as P

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
D = np.load(os.path.join(HERE, "p2_gram.npz"), allow_pickle=True)
GRAM, XTY = D["gram"], D["xty"]
N = P.names()[:21]

FAMILY = {
    "e15ceil95": "fastgs_e", "e16app": "fastgs_e", "e17visnorm": "fastgs_e",
    "gsplatB1": "gsplat_base", "gsplatB2": "gsplat_base", "gsplatB3": "gsplat_base",
    "gsplatB4warm": "gsplat_base", "gsplatB5affine": "gsplat_base",
    "gsplatB6bilagrid": "gsplat_base", "gsplatB7ppisp2": "gsplat_base",
    "gsplatB8pure": "gsplat_base",
    "gsplatB9ut": "gsplat_ut", "gsplatB10ut8M": "gsplat_ut",
    "gsplatB11ut60k": "gsplat_ut", "gsplatB12ut8Ms7": "gsplat_ut",
    "m31b_nolpips": "m31b", "m31b_taillpips": "m31b",
    "sh0": "sh", "sh1": "sh", "sh2": "sh", "sh3": "DUP_of_gsplatB11ut60k",
}


def fit(idxs, views, simplex=True):
    """min_w sum_{v in views} ||X_v w - g_v||^2  s.t. w>=0 (and sum w = 1)."""
    G = GRAM[views][:, idxs][:, :, idxs].mean(0)
    b = XTY[views][:, idxs].mean(0)
    k = len(idxs)
    if not simplex:
        L = np.linalg.cholesky(G + 1e-12 * np.eye(k))
        w, _ = nnls(L.T, np.linalg.solve(L, b))
        return w
    f = lambda w: w @ G @ w - 2 * w @ b
    jac = lambda w: 2 * (G @ w - b)
    r = minimize(f, np.ones(k) / k, jac=jac, method="SLSQP",
                 bounds=[(0, 1)] * k,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                               "jac": lambda w: np.ones(k)}],
                 options={"maxiter": 500, "ftol": 1e-14})
    return np.clip(r.x, 0, None) / max(np.clip(r.x, 0, None).sum(), 1e-12)


def expand(idxs, w):
    v = np.zeros(21); v[list(idxs)] = w; return v


def main():
    t0 = time.time()
    p1 = json.load(open(os.path.join(HERE, "p1.json")))
    solo = {n: P.agg(p1[n])[0] for n in N}
    distinct = [i for i, n in enumerate(N) if n != "sh3"]

    # one representative (best solo) per family
    reps = {}
    for i in distinct:
        f = FAMILY[N[i]]
        if f not in reps or solo[N[i]] > solo[N[reps[f]]]:
            reps[f] = i
    REP = sorted(reps.values())
    print("family reps:", [(FAMILY[N[i]], N[i], round(solo[N[i]], 3)) for i in REP], flush=True)

    UT = [N.index(x) for x in ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]]
    # production analogue: 6 members + 1 mip-ish outsider -> use the 7 best distinct
    TOP7 = sorted(distinct, key=lambda i: -solo[N[i]])[:7]

    SETS = {"REP5": REP, "UT4": UT, "TOP7": TOP7, "ALL20": distinct}

    # 2-fold split of views: interleaved (A = even, B = odd) so both folds cover
    # the whole trajectory; a contiguous split would confound with scene content.
    fa = list(range(0, 60, 2)); fb = list(range(1, 60, 2))
    folds = {"A": fa, "B": fb}

    cands = {}
    meta = {}
    for sname, idxs in SETS.items():
        cands[f"{sname}_unif"] = expand(idxs, np.ones(len(idxs)) / len(idxs))
        meta[f"{sname}_unif"] = {"set": sname, "kind": "unif"}
        for fname, fv in folds.items():
            w = fit(idxs, fv, simplex=True)
            other = "B" if fname == "A" else "A"
            cands[f"{sname}_fit{fname}"] = expand(idxs, w)
            meta[f"{sname}_fit{fname}"] = {"set": sname, "kind": "fit", "trained_on": fname,
                                           "eval_on": other, "w": dict(zip([N[i] for i in idxs],
                                                                           [round(float(x), 4) for x in w]))}
        # in-sample (all views) fit -- the OPTIMISTIC number, for contrast
        w = fit(idxs, list(range(60)), simplex=True)
        cands[f"{sname}_fitALL"] = expand(idxs, w)
        meta[f"{sname}_fitALL"] = {"set": sname, "kind": "fit_insample",
                                   "w": dict(zip([N[i] for i in idxs], [round(float(x), 4) for x in w]))}

    for k, v in meta.items():
        if "w" in v:
            print(k, v["w"], flush=True)

    pl = P.Pool()
    res = pl.score(cands)
    pl.close()
    agg = {k: P.agg(v) for k, v in res.items()}
    aggA = {k: P.agg(v, fa) for k, v in res.items()}
    aggB = {k: P.agg(v, fb) for k, v in res.items()}
    json.dump({"meta": meta, "all": agg, "foldA": aggA, "foldB": aggB,
               "sets": {k: [N[i] for i in v] for k, v in SETS.items()}},
              open(os.path.join(HERE, "p4.json"), "w"), indent=1)

    print("\n=== cross-family weight fitting, 2-fold CV (score on HELD-OUT views)")
    print(f"{'set':7s} {'k':>2s} {'unifA':>8s} {'fitB->A':>8s} {'unifB':>8s} {'fitA->B':>8s} "
          f"{'CV delta':>9s} {'insample':>9s} {'insam d':>8s}")
    for sname, idxs in SETS.items():
        uA = aggA[f"{sname}_unif"][0]; uB = aggB[f"{sname}_unif"][0]
        oA = aggA[f"{sname}_fitB"][0]                # trained on B, eval on A
        oB = aggB[f"{sname}_fitA"][0]                # trained on A, eval on B
        cv = ((oA - uA) + (oB - uB)) / 2
        ins = agg[f"{sname}_fitALL"][0]; insd = ins - agg[f"{sname}_unif"][0]
        print(f"{sname:7s} {len(idxs):2d} {uA:8.4f} {oA:8.4f} {uB:8.4f} {oB:8.4f} "
              f"{cv:+9.4f} {ins:9.4f} {insd:+8.4f}")
    print("elapsed %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
