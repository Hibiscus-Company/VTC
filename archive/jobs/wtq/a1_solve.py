#!/usr/bin/env python
"""Solve the weighting problem exactly (in MSE) from the cached Gram, for any subset / partition.

P*MSE(w) = w' Gd w - 2 w' bd + s          for any w with sum(w) = 1
"""
import json, sys
import numpy as np

Z = np.load("/home/bkai/.claude/jobs/1c9cf7e9/tmp/wtq/gram.npz", allow_pickle=True)
Gd, bd, ss, P = Z["Gd"], Z["bd"], float(Z["ss"][0]) * 0 + 0.0, float(Z["P"])
SS = Z["ss"]
NAMES = list(Z["names"])
FAM = list(Z["fam"])
NIM = Gd.shape[0]


def mse_per_image(w, idx, sel=None):
    """w over the members listed in idx; returns per-image MSE array."""
    w = np.asarray(w, dtype=np.float64)
    G = Gd[:, idx][:, :, idx]
    b = bd[:, idx]
    q = np.einsum("i,nij,j->n", w, G, w) - 2 * (b @ w) + SS
    return (q if sel is None else q[sel]) / P


def psnr(w, idx, sel=None):
    m = mse_per_image(w, idx, sel)
    return float(np.mean(10 * np.log10(1.0 / np.maximum(m, 1e-12))))


def ls_weights(idx, sel=None, ridge=0.0, nonneg=False):
    """simplex(sum=1) LS weights fit on images `sel`."""
    G = Gd[:, idx][:, :, idx]
    b = bd[:, idx]
    if sel is not None:
        G, b = G[sel], b[sel]
    G = G.sum(0)
    b = b.sum(0)
    k = len(idx)
    r = 0                                   # reference member index inside the subset
    fr = [i for i in range(k) if i != r]
    Gp = G[np.ix_(fr, fr)] - G[np.ix_(fr, [r] * (k - 1))] \
         - G[np.ix_([r] * (k - 1), fr)] + G[r, r]
    bp = b[fr] - b[r] - G[fr, r] + G[r, r]
    Gp = Gp + ridge * np.trace(Gp) / (k - 1) * np.eye(k - 1)
    wf = np.linalg.solve(Gp, bp)
    w = np.empty(k)
    w[fr] = wf
    w[r] = 1.0 - wf.sum()
    if nonneg:                              # projected gradient on the simplex
        w = np.full(k, 1.0 / k)
        L = np.linalg.eigvalsh(G - np.outer(G[:, r], np.ones(k)) * 0)[-1]
        step = 1.0 / max(L, 1e-9)
        for _ in range(4000):
            g = 2 * (G @ w - b)
            w = simplex_proj(w - step * g)
    return w


def simplex_proj(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1.0), 0)


def main():
    print(f"n images = {NIM},  {len(NAMES)} members\n")
    print(f"{'member':>18} {'fam':>5} {'solo PSNR':>10}")
    solo = {}
    for i, nm in enumerate(NAMES):
        w = np.zeros(len(NAMES)); w[i] = 1.0
        solo[nm] = psnr(w, list(range(len(NAMES))))
        print(f"{nm:>18} {FAM[i]:>5} {solo[nm]:10.4f}")

    odd = np.arange(NIM) % 2 == 1
    ev = ~odd

    def report(tag, idx):
        k = len(idx)
        names = [NAMES[i] for i in idx]
        uni = np.full(k, 1.0 / k)
        pu = psnr(uni, idx)
        wls = ls_weights(idx)
        pls = psnr(wls, idx)
        w_e = ls_weights(idx, sel=ev)
        w_o = ls_weights(idx, sel=odd)
        p_cv = 0.5 * (psnr(w_e, idx, sel=odd) + psnr(w_o, idx, sel=ev))
        p_cv_u = 0.5 * (psnr(uni, idx, sel=odd) + psnr(uni, idx, sel=ev))
        wnn = ls_weights(idx, nonneg=True)
        print(f"\n--- {tag}  k={k}: {', '.join(names)}")
        print(f"  uniform          PSNR {pu:8.4f}")
        print(f"  LS oracle        PSNR {pls:8.4f}  ({pls-pu:+.4f} dB) w = "
              f"{np.array2string(wls, precision=3, suppress_small=True)}")
        print(f"  LS nonneg oracle PSNR {psnr(wnn, idx):8.4f}  "
              f"({psnr(wnn,idx)-pu:+.4f} dB) w = {np.array2string(wnn, precision=3)}")
        print(f"  LS 2-fold CV     PSNR {p_cv:8.4f}  ({p_cv-p_cv_u:+.4f} dB honest)")
        print(f"     fold weights even: {np.array2string(w_e, precision=3, suppress_small=True)}")
        print(f"     fold weights odd : {np.array2string(w_o, precision=3, suppress_small=True)}")
        return wls, w_e, w_o

    ALL = list(range(len(NAMES)))
    ut = [i for i in ALL if FAM[i] == "UT"]
    report("UT family only", ut)
    report("ALL 12", ALL)
    return solo, ut, ALL


def family_blend(idxA, idxB, tag):
    """uniform inside each family, sweep the total weight on family B; exact MSE."""
    kA, kB = len(idxA), len(idxB)
    idx = idxA + idxB
    print(f"\n=== 2-family blend {tag}: A(k={kA})={[NAMES[i] for i in idxA]} | "
          f"B(k={kB})={[NAMES[i] for i in idxB]}")
    best = (None, -1e9)
    rows = []
    for wB in np.arange(0.0, 1.001, 0.025):
        w = np.concatenate([np.full(kA, (1 - wB) / kA), np.full(kB, wB / kB)])
        p = psnr(w, idx)
        rows.append((wB, p))
        if p > best[1]:
            best = (wB, p)
    uni = kB / (kA + kB)
    pu = psnr(np.full(kA + kB, 1.0 / (kA + kB)), idx)
    for wB, p in rows:
        if abs(wB - round(wB, 2)) < 1e-9 and (abs(wB * 40 % 4) < 1e-9):
            print(f"   wB={wB:5.3f}  PSNR {p:8.4f}  ({p-pu:+.4f} vs uniform)")
    print(f"   uniform wB={uni:.3f} PSNR {pu:.4f} | BEST wB={best[0]:.3f} PSNR {best[1]:.4f} "
          f"({best[1]-pu:+.4f} dB)")
    # per-member weight ratio implied
    if best[0] not in (0.0, 1.0):
        ratio = (best[0] / kB) / ((1 - best[0]) / kA)
        print(f"   => a family-B member deserves {ratio:.2f}x a family-A member "
              f"(production assumes 1.50x)")
    return best


if __name__ == "__main__":
    solo, ut, ALL = main()
    ordered = sorted(NAMES, key=lambda n: -solo[n])
    print("\nsolo ranking:", [(n, round(solo[n], 3)) for n in ordered])
