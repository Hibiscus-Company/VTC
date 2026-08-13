"""AGGREGATOR STUDY on the calibrated production harness (HCM0181, 60 real test poses).

usage: python agg_run.py <pool>            pool in {p4, p7}
Stores PER-VIEW metrics so any fold/CV analysis afterwards is exact and free
(the project composite is linear in mean(PSNR), mean(SSIM), mean(LPIPS)).
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_lib as A
import agg_core as C

POOLS = {
    # the sanctioned, CALIBRATED k=4 ut family (prod_harness.py MEMBERS)
    "p4": ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"],
    # production-shaped depth: 7 members (prod towers ship 6 + 1 mip3d)
    "p7": ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
           "m31b_taillpips", "e17visnorm", "gsplatB8pure"],
    # --- replication pools: disjoint k=4 ensembles from other HCM0181 generations ---
    "pB": ["gsplatB1", "gsplatB2", "gsplatB3", "gsplatB4warm"],
    "pC": ["e15ceil95", "e16app", "e17visnorm", "m31b_taillpips"],
    "pD": ["gsplatB5affine", "gsplatB7ppisp2", "gsplatB8pure", "m31b_nolpips"],
    # --- stress pool: 3 good ut members + one BROKEN member (bilagrid, PSNR 17.4).
    #     quantifies the insurance value of a robust aggregator when a member fails. ---
    "pOUT": ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB6bilagrid"],
    # --- REPLICATION of the only positive result (winsor1 @ k=7). Members FULLY
    #     DISJOINT from p7, so this is an independent test of the same rule. ---
    "p7B": ["gsplatB1", "gsplatB2", "gsplatB3", "gsplatB4warm", "gsplatB5affine",
            "gsplatB7ppisp2", "m31b_nolpips"],
    # --- third k=7 pool, mixed generations (closest in spirit to a shipped tower) ---
    "p7C": ["gsplatB9ut", "gsplatB12ut8Ms7", "e15ceil95", "e16app", "sh3",
            "gsplatB4warm", "m31b_nolpips"],
    # --- production-DEPTH outlier stress: p7 with one member swapped for the BROKEN
    #     bilagrid run. Quantifies the insurance value of robustness at k=7. ---
    "p7OUT": ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
              "m31b_taillpips", "e17visnorm", "gsplatB6bilagrid"],
}


def registry(k):
    R = {
        "mean": C.agg_mean,
        "median": C.agg_median,
        "trim1": lambda X: C.agg_trim(X, 1),
        "winsor1": lambda X: C.agg_winsor(X, 1),
        "vecmed": C.agg_vecmed,
        "closest2med": C.agg_closest2med,
        "patchsel64": lambda X: C.agg_patchsel(X, 64),
        "geomean(p=0)": lambda X: C.agg_pmean(X, 0.0),
        "pmean(p=2)": lambda X: C.agg_pmean(X, 2.0),
        "pmean(p=0.5)": lambda X: C.agg_pmean(X, 0.5),
        "mean_linearRGB": C.agg_mean_linear,
        "lumguided(med)": lambda X: C.agg_lumguided(X, "median"),
        "chromarobust": C.agg_chromarobust,
        "invvar(r=8)": lambda X: C.agg_invvar(X, 8),
        "invvar(r=24)": lambda X: C.agg_invvar(X, 24),
    }
    if k >= 6:
        R["trim2"] = lambda X: C.agg_trim(X, 2)
        R["winsor2"] = lambda X: C.agg_winsor(X, 2)
    for a in (-0.5, -0.25, 0.25, 0.5, 0.75):
        R[f"blend(a={a})"] = (lambda a_: (lambda X: C.agg_blend(X, a_)))(a)
    for t in (0.5, 1.5, 2.0, 3.0, 4.0):
        R[f"wex(t={t})"] = (lambda t_: (lambda X: C.agg_wex(X, t_)))(t)
    for r in (2, 8, 32):
        R[f"hybridfreq(r={r})"] = (lambda r_: (lambda X: C.agg_hybrid_freq(X, r_)))(r)
    for c in (0.5, 1.0, 1.345, 2.0, 4.0):
        R[f"huber(c={c})"] = (lambda c_: (lambda X: C.agg_huber(X, c_)))(c)
    for c in (2.0, 3.0, 4.685, 8.0):
        R[f"tukey(c={c})"] = (lambda c_: (lambda X: C.agg_tukey(X, c_)))(c)
    for b in (0.5, 1.0, 2.0):
        R[f"softmax(b={b})"] = (lambda b_: (lambda X: C.agg_softmax(X, b_)))(b)
    R["huber-lum(c=1.345)"] = lambda X: C.agg_huber(X, 1.345, lum=True)
    R["tukey-lum(c=4.685)"] = lambda X: C.agg_tukey(X, 4.685, lum=True)
    sub = os.environ.get("AGG_SUBSET")
    if sub:
        keep = [s for s in sub.split(",") if s]
        R = {k: v for k, v in R.items() if k in keep}
        missing = [k for k in keep if k not in R]
        if missing:
            raise SystemExit(f"unknown aggregator(s) in AGG_SUBSET: {missing}")
    return R


def main():
    pool = sys.argv[1] if len(sys.argv) > 1 else "p4"
    variants = POOLS[pool]
    dev = A.init()
    stems, gt_by = A.stems_for(variants)
    print(f"pool={pool}  k={len(variants)}  views={len(stems)}")
    R = registry(len(variants))
    names = list(R)
    per = {n: [] for n in names}
    t0 = time.time()
    for i, s in enumerate(stems):
        X = np.stack([A.load(A.mdir(v), s) for v in variants], 0)
        g = A.gt_tensor(gt_by[s], dev)
        if i == 0:
            med, tr, wi = C.agg_median(X), C.agg_trim(X, 1), C.agg_winsor(X, 1)
            print(f"  [degeneracy check k={len(variants)}] "
                  f"max|median-trim1|={np.abs(med-tr).max():.2e}  "
                  f"max|median-winsor1|={np.abs(med-wi).max():.2e}")
        for n in names:
            img = R[n](X)
            p, ss, l = A.metrics(img, g, dev)
            per[n].append((p, ss, l))
        del X, g
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(stems)}  ({time.time()-t0:.0f}s)", flush=True)
    out = {n: np.asarray(v, dtype=np.float64).tolist() for n, v in per.items()}
    tag = os.environ.get("AGG_TAG", "")
    p = f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/agg_{pool}{tag}.json"
    json.dump({"pool": pool, "variants": variants, "stems": stems, "per_view": out},
              open(p, "w"))
    print("wrote", p)

    def comp(a):
        P, S, L = a.mean(0)
        return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    base = comp(np.asarray(per["mean"]))
    rows = []
    for n in names:
        a = np.asarray(per[n]); P, S, L = a.mean(0)
        rows.append((n, comp(a), comp(a) - base, P, S, L))
    rows.sort(key=lambda r: -r[1])
    print(f"\n{'aggregator':<22}{'SCORE':>9}{'d(mean)':>9}{'PSNR':>9}{'SSIM':>8}{'LPIPS':>9}")
    for n, sc, d, P, S, L in rows:
        print(f"{n:<22}{sc:9.4f}{d:+9.4f}{P:9.4f}{S:8.4f}{L:9.4f}")


if __name__ == "__main__":
    main()
