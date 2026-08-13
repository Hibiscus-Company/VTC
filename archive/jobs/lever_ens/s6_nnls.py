"""MSE-optimal (NNLS) weights from Gram matrices, with 2-fold CV over views.
Reports fitted weights and the PSNR they buy vs uniform, in-sample and out-of-sample."""
import os, sys, json, numpy as np
from scipy.optimize import nnls, minimize

OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens"
Z = np.load(os.path.join(OUT, "gram.npz"), allow_pickle=True)
A_all, b_all, c_all, n_all = Z["A"], Z["b"], Z["c"], Z["n"]
NAMES = list(Z["names"])
IDX = {n: i for i, n in enumerate(NAMES)}
singles = json.load(open(os.path.join(OUT, "singles.json")))


def agg(members, imgs):
    j = [IDX[m] for m in members]
    A = A_all[np.ix_(imgs, j, j)].sum(0)
    b = b_all[np.ix_(imgs, j)].sum(0)
    c = c_all[imgs].sum()
    n = n_all[imgs].sum()
    return A, b, c, n


def mse_of(w, A, b, c, n):
    return (w @ A @ w - 2 * b @ w + c) / n


def psnr_of(w, A, b, c, n):
    """per-image-mean PSNR is not linear in aggregate MSE; use per-image."""
    return None


def per_image_psnr(members, w, imgs):
    j = [IDX[m] for m in members]
    ps = []
    for gi in imgs:
        A = A_all[gi][np.ix_(j, j)]; b = b_all[gi][j]; c = c_all[gi]; n = n_all[gi]
        mse = max((w @ A @ w - 2 * b @ w + c) / n, 1e-12)
        ps.append(10 * np.log10(1.0 / mse))
    return float(np.mean(ps))


def fit_nnls(members, imgs, simplex=False):
    A, b, c, n = agg(members, imgs)
    A = A + np.eye(len(members)) * (np.trace(A) / len(members)) * 1e-10
    if not simplex:
        L = np.linalg.cholesky(A)
        w, _ = nnls(L.T, np.linalg.solve(L, b))
        return w
    k = len(members)
    fun = lambda w: w @ A @ w - 2 * b @ w
    jac = lambda w: 2 * (A @ w - b)
    r = minimize(fun, np.ones(k) / k, jac=jac, method="SLSQP",
                 bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                                    "jac": lambda w: np.ones(k)}],
                 options=dict(maxiter=500, ftol=1e-14))
    return np.clip(r.x, 0, None)


FAM = {
    "e15ceil95": "FastGS", "e16app": "FastGS", "e17visnorm": "FastGS",
    "gsplatB1": "MCMC", "gsplatB2": "MCMC", "gsplatB3": "MCMC", "gsplatB4warm": "MCMC",
    "gsplatB5affine": "MCMC", "gsplatB6bilagrid": "MCMC", "gsplatB7ppisp2": "MCMC",
    "gsplatB8pure": "MCMC",
    "gsplatB9ut": "UT", "gsplatB10ut8M": "UT", "gsplatB11ut60k": "UT", "gsplatB12ut8Ms7": "UT",
    "m31b_nolpips": "UTmetric", "m31b_taillpips": "UTmetric",
    "sh0": "UTsh", "sh1": "UTsh", "sh2": "UTsh", "sh3": "UTdup", "k4": "ENS",
}

SETS = {
    # one representative per family (the production-relevant question)
    "xfam5": ["e17visnorm", "gsplatB8pure", "gsplatB11ut60k", "gsplatB12ut8Ms7", "m31b_taillpips"],
    # the 4-member UT ensemble that IS the calibrated harness k4
    "k4": ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"],
    # production-analogue: 6 diverse + representatives
    "xfam7": ["e15ceil95", "e17visnorm", "gsplatB8pure", "gsplatB9ut", "gsplatB10ut8M",
              "gsplatB11ut60k", "gsplatB12ut8Ms7"],
    # all 20 unique models
    "all20": [n for n in NAMES if n not in ("sh3", "k4")],
}

rng = np.random.RandomState(0)
perm = rng.permutation(60)
FOLDS = [(sorted(perm[:30]), sorted(perm[30:])), (sorted(perm[30:]), sorted(perm[:30]))]
ALL = list(range(60))

print("=== MSE-optimal (NNLS) weights, PSNR only ===")
out = {}
for sname, mem in SETS.items():
    for simplex in (True, False):
        tag = f"{sname}{'/simplex' if simplex else '/free'}"
        w_in = fit_nnls(mem, ALL, simplex)
        u = np.ones(len(mem)) / len(mem)
        p_u = per_image_psnr(mem, u, ALL)
        p_f = per_image_psnr(mem, w_in, ALL)
        cvs = []
        for tr, te in FOLDS:
            w = fit_nnls(mem, tr, simplex)
            cvs.append(per_image_psnr(mem, w, te) - per_image_psnr(mem, u, te))
        print(f"\n{tag}  n={len(mem)}")
        print("  uniform PSNR %.4f   in-sample fit %.4f (+%.4f dB)   CV-fold gains %s dB (mean %+.4f)"
              % (p_u, p_f, p_f - p_u, np.round(cvs, 4), np.mean(cvs)))
        print("  w =", ", ".join(f"{m}:{x:.3f}" for m, x in zip(mem, w_in)))
        out[tag] = dict(members=mem, w=list(map(float, w_in)), psnr_uniform=p_u,
                        psnr_fit=p_f, cv_gain_db=list(map(float, cvs)))
json.dump(out, open(os.path.join(OUT, "nnls.json"), "w"), indent=1)
