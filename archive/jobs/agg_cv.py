"""Honest analysis of the aggregator sweep: paired bootstrap CIs vs the plain MEAN,
plus proper cross-validated selection inside every parametric family.

The project composite is LINEAR in mean(PSNR), mean(SSIM), mean(LPIPS) (PSNR<50 here),
so per-view composites average exactly to the reported score -> fold analysis is exact.
"""
import json, sys, re
import numpy as np

RNG = np.random.default_rng(0)


def comp_pv(a):
    """per-view composite, (n,3)->(n,)"""
    P, S, L = a[:, 0], a[:, 1], a[:, 2]
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * np.minimum(P / 50.0, 1.0))


def load(p):
    d = json.load(open(p))
    return {k: np.asarray(v) for k, v in d["per_view"].items()}, d


FAMILIES = {
    "blend(a=*)": r"^blend\(a=(-?[\d.]+)\)$",
    "huber(c=*)": r"^huber\(c=([\d.]+)\)$",
    "tukey(c=*)": r"^tukey\(c=([\d.]+)\)$",
    "softmax(b=*)": r"^softmax\(b=([\d.]+)\)$",
    "hybridfreq(r=*)": r"^hybridfreq\(r=([\d.]+)\)$",
    "invvar(r=*)": r"^invvar\(r=([\d.]+)\)$",
}


def main():
    pv, meta = load(sys.argv[1])
    n = len(next(iter(pv.values())))
    C = {k: comp_pv(v) for k, v in pv.items()}
    base = C["mean"]
    print(f"pool={meta['pool']} k={len(meta['variants'])} views={n}")
    print(f"MEAN baseline score = {base.mean():.4f}\n")

    print(f"{'aggregator':<22}{'d(mean)':>9}{'95% CI':>18}{'win/60':>8}{'dPSNR':>8}{'dSSIM':>9}{'dLPIPS':>9}")
    rows = []
    B = 20000
    idx = RNG.integers(0, n, size=(B, n))
    for k, c in C.items():
        if k == "mean":
            continue
        d = c - base
        bs = d[idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        dm = pv[k].mean(0) - pv["mean"].mean(0)
        rows.append((k, d.mean(), lo, hi, int((d > 0).sum()), dm))
    rows.sort(key=lambda r: -r[1])
    for k, d, lo, hi, w, dm in rows:
        star = "  *" if lo > 0 else ("  x" if hi < 0 else "")
        print(f"{k:<22}{d:+9.4f}  [{lo:+7.4f},{hi:+7.4f}]{w:8d}"
              f"{dm[0]:+8.4f}{dm[1]:+9.5f}{dm[2]:+9.5f}{star}")

    print("\n--- cross-validated selection inside each parametric family "
          "(repeated 2-fold, 400 splits; and leave-one-view-out) ---")
    print(f"{'family':<18}{'best(full)':>26}{'CV 2-fold d':>13}{'LOO d':>10}")
    for fam, pat in FAMILIES.items():
        mem = [(float(re.match(pat, k).group(1)), k) for k in C if re.match(pat, k)]
        if not mem:
            continue
        mem.sort()
        keys = [k for _, k in mem]
        M = np.stack([C[k] for k in keys], 0)          # (m, n)
        full = M.mean(1)
        bi = int(np.argmax(full))
        # repeated 2-fold
        acc = []
        for _ in range(400):
            perm = RNG.permutation(n)
            a, b = perm[: n // 2], perm[n // 2:]
            for tr, te in ((a, b), (b, a)):
                j = int(np.argmax(M[:, tr].mean(1)))
                acc.append(M[j, te].mean() - base[te].mean())
        cv = float(np.mean(acc))
        # leave-one-view-out
        loo = []
        for i in range(n):
            m = np.ones(n, bool); m[i] = False
            j = int(np.argmax(M[:, m].mean(1)))
            loo.append(M[j, i] - base[i])
        print(f"{fam:<18}{keys[bi]:>26}{cv:+13.4f}{np.mean(loo):+10.4f}")


if __name__ == "__main__":
    main()
