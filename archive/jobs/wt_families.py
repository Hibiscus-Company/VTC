#!/usr/bin/env python
"""Family-weight law from the cached Gram: wB* for every plausible A/B pairing, plus the
stakes between the shipped 0.333 and the uniform-per-member share. No GPU."""
import numpy as np, itertools

Z = np.load("/home/bkai/.claude/jobs/1c9cf7e9/tmp/wt_gram.npz")
G, names = Z["G"], list(Z["names"])
psnr, lp = Z["psnr"], Z["lpips"]
idx = {n: i for i, n in enumerate(names)}
C = G.mean(0)


def sm(members, S=None):
    ii = [idx[m] for m in members]
    return (G[S] if S is not None else G).mean(0)[np.ix_(ii, ii)]


def mse(w, Cm):
    return float(np.asarray(w, float) @ Cm @ np.asarray(w, float)) / 3.0


def two_family(A, B, S=None):
    M = A + B
    Cm = sm(M, S)
    wa = np.array([1.0 / len(A)] * len(A) + [0.0] * len(B))
    wb = np.array([0.0] * len(A) + [1.0 / len(B)] * len(B))
    caa, cbb = mse(wa, Cm), mse(wb, Cm)
    cab = float(wa @ Cm @ wb) / 3.0
    w = (caa - cab) / (caa + cbb - 2 * cab)
    return w, caa, cbb, cab, Cm, wa, wb


def dpsnr(w1, w0, Cm, wa, wb):
    return 10 * np.log10(mse((1 - w0) * wa + w0 * wb, Cm) / mse((1 - w1) * wa + w1 * wb, Cm))


UT4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
A6 = UT4 + ["gsplatB1", "gsplatB2"]                 # 6-member main pool, the r22 analogue
FAMS = {
    "m31b(2)":       ["m31b_nolpips", "m31b_taillpips"],
    "fid(3)":        ["fid_ctrl", "fid_l2off", "fid_l2reg"],
    "e1x(3)":        ["e15ceil95", "e16app", "e17visnorm"],
    "sh(3)":         ["sh0", "sh1", "sh2"],
    "appearance(3)": ["gsplatB5affine", "gsplatB6bilagrid", "gsplatB7ppisp2"],
    "pure/warm(2)":  ["gsplatB8pure", "gsplatB4warm"],
    "fid_ctrl(1)":   ["fid_ctrl"],
    "e17(1)":        ["e17visnorm"],
    "B8pure(1)":     ["gsplatB8pure"],
}

print("A = 6-member main pool", A6)
print(f"\n{'family B':>15} {'nB':>3} {'uni':>6} {'wB*':>7} {'corrAB':>7} {'rmseA':>7} {'rmseB':>7}"
      f" {'dP(wB*)':>8} {'dP(.333)':>9} {'dP(.20)':>8} {'CVwB*':>7} {'dP_cv':>8}")
rng = np.random.RandomState(0)
for fn, B in FAMS.items():
    A = [m for m in A6 if m not in B]
    w, caa, cbb, cab, Cm, wa, wb = two_family(A, B)
    uni = len(B) / (len(A) + len(B))
    r = cab / np.sqrt(caa * cbb)
    # honest CV: pick wB on half the stems, score on the other half
    cvw, cvd = [], []
    for rep in range(30):
        perm = rng.permutation(G.shape[0])
        for f in range(2):
            te, tr = perm[f::2], perm[1 - f::2]
            wt = two_family(A, B, tr)[0]
            Ct = sm(A + B, te)
            cvw.append(wt)
            cvd.append(10 * np.log10(mse((1 - uni) * wa + uni * wb, Ct) /
                                     mse((1 - wt) * wa + wt * wb, Ct)))
    print(f"{fn:>15} {len(B):3d} {uni:6.3f} {w:7.3f} {r:7.4f} {np.sqrt(caa):7.5f} "
          f"{np.sqrt(cbb):7.5f} {dpsnr(w,uni,Cm,wa,wb):+8.4f} {dpsnr(0.3333,uni,Cm,wa,wb):+9.4f} "
          f"{dpsnr(0.20,uni,Cm,wa,wb):+8.4f} {np.mean(cvw):7.3f} {np.mean(cvd):+8.4f}")

print("\n\nDOES A NO-GT ESTIMATOR RECOVER wB*?  (production has no test GT)")
print("  proxy: replace the GT residual second moments by DISAGREEMENT about the pooled mean")
print("  i.e. treat each family mean's deviation from the all-member mean as its error proxy.")
print(f"{'family B':>15} {'wB*(GT)':>9} {'wB(proxy)':>10} {'dP@proxy':>9}")
for fn, B in FAMS.items():
    A = [m for m in A6 if m not in B]
    w, caa, cbb, cab, Cm, wa, wb = two_family(A, B)
    # proxy uses only member-vs-member differences (available without GT):
    # d = mean over members of E[(x_i - xbar)^2]; here xbar = grand mean of A+B members
    M = A + B
    Cf = sm(M)
    n = len(M)
    u = np.ones(n) / n
    # E[(x_i - xbar)^2] = C_ii - 2 C_i.u + u C u  -- GT cancels, so this is GT-free
    dv = np.array([Cf[i, i] - 2 * Cf[i] @ u + u @ Cf @ u for i in range(n)])
    ca = np.mean(dv[:len(A)]); cb = np.mean(dv[len(A):])
    # cross term proxy: covariance of family-mean deviations
    da = wa - u; db = wb - u
    pa, pb = da @ Cf @ da, db @ Cf @ db
    pab = da @ Cf @ db
    wp = (ca - 0) / (ca + cb) if (ca + cb) > 0 else 0.5
    # inverse-variance-of-disagreement weighting, the only GT-free rule available
    wp = (1 / cb) / (1 / ca * len(A) / len(A) + 1 / cb) if cb > 0 else 0.5
    wp = (len(B) / cb) / (len(A) / ca + len(B) / cb)
    print(f"{fn:>15} {w:9.3f} {wp:10.3f} {dpsnr(wp,w,Cm,wa,wb):+9.4f}")
