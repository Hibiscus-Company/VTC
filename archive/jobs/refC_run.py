#!/usr/bin/env python
"""REFUTATION STEP C: test the EXTRAPOLATION and the DEVIATION-TIER assumption directly.

The claim's number is a power-law extrapolation from rungs k<=6 out to k=8->9, and it assumes the
private candidate (s2gates champA, deviation ratio 2.23-2.26 x the in-pool UT spread) behaves like
HCM0181's e17visnorm/e15ceil95 (deviation ratio 1.59-1.60). Two things to measure:

  A. one rung DEEPER than anything the claim measured (base7, all non-FGS), at exactly the
     production weight share f = 1/7, to see whether the fitted power law survives out-of-sample;
  B. the same add with e16app -- the FGS-family member that actually sits in the private
     candidate's DEVIATION TIER (2.79 vs champA's 2.23) instead of e17's 1.60.

Everything runs through the FULL shipped chain and keeps per-view scores for paired SEs.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from div_lib import UT4, RD, GTD, ld, prep, combine, chain, load_field, stems
from lapfuse import _K
from fieldlib import warp

LAM, GAIN, F = 1.0, 1.30, 1.0 / 7.0
BASE6 = UT4 + ["m31b_taillpips", "m31b_nolpips"]      # == the claim's Q6
BASE7 = BASE6 + ["sh3"]                                # +1 same-family near-duplicate rung


def w_for(n, f):
    return [1.0] * n + [n * f / (1.0 - f)]


ARMS = {
    "base6 (claim Q6)":          BASE6,
    "base7 (=Q6+sh3)":           BASE7,
    "b6 +FGS(e17) f=1/7":        (BASE6 + ["e17visnorm"], w_for(6, F)),
    "b7 +FGS(e17) f=1/7":        (BASE7 + ["e17visnorm"], w_for(7, F)),
    "b7 +FGS(e15) f=1/7":        (BASE7 + ["e15ceil95"], w_for(7, F)),
    "b7 +FGS(e16app) f=1/7":     (BASE7 + ["e16app"], w_for(7, F)),
    "b7 +CLONE(ut60k) f=1/7":    "clone7",
}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K = _K.to(dev)
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    ss, gt_by = stems()
    lens = load_field(GAIN)
    need = sorted({m for v in ARMS.values() if not isinstance(v, str)
                   for m in (v[0] if isinstance(v, tuple) else v)})
    names = list(ARMS)
    per = {a: [] for a in names}
    acc = {a: np.zeros(3) for a in names}
    t0 = time.time()
    for c, s in enumerate(ss):
        X = {m: ld(os.path.join(RD(m), s + ".png"), dev) for m in need}
        P = {m: prep(X[m], K) for m in need}
        L0 = {m: P[m][0] for m in need}
        Ae = {m: P[m][1] for m in need}
        X["__clone__"], L0["__clone__"], Ae["__clone__"] = (X["gsplatB11ut60k"],
                                                            L0["gsplatB11ut60k"],
                                                            Ae["gsplatB11ut60k"])
        g = ld(os.path.join(GTD, gt_by[s]), dev)
        for a in names:
            v = ARMS[a]
            if v == "clone7":
                idx, w = BASE7 + ["__clone__"], w_for(7, F)
            elif isinstance(v, tuple):
                idx, w = v
            else:
                idx, w = v, None
            out = combine(X, L0, Ae, idx, LAM, w)
            j, _ = chain(out, lens, warp)
            r = torch.from_numpy(j).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                Pv = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                Sv = float(repo_ssim(r, g))
                Lv = float(vgg(r * 2 - 1, g * 2 - 1).item())
            acc[a] += (Pv, Sv, Lv)
            per[a].append(100 * (0.4 * (1 - Lv) + 0.3 * Sv + 0.3 * min(Pv / 50.0, 1.0)))
        if c % 10 == 0:
            print(f"  {c}/{len(ss)} {time.time()-t0:.0f}s", flush=True)

    N = len(ss)
    res = {a: dict(zip(("psnr", "ssim", "lpips"), acc[a] / N)) for a in names}
    for a in names:
        r = res[a]
        r["score"] = 100 * (0.4 * (1 - r["lpips"]) + 0.3 * r["ssim"]
                            + 0.3 * min(r["psnr"] / 50.0, 1.0))
        r["per"] = per[a]
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in res.items()},
              open(f"{HERE}/refC.json", "w"), indent=1)

    print(f"\nREFUTE-C  HCM0181  n={N}  lam={LAM}  gain={GAIN}  FULL SHIPPED CHAIN  f=1/7")
    print(f"{'arm':>26} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7}")
    for a in names:
        r = res[a]
        print(f"{a:>26} {r['score']:9.4f} {r['psnr']:8.4f} {r['ssim']:7.4f} {r['lpips']:7.4f}")

    def paired(a, b):
        d = np.array(per[a]) - np.array(per[b])
        se = d.std(ddof=1) / np.sqrt(N)
        return res[a]["score"] - res[b]["score"], se, d.mean() / se, int((d > 0).sum())

    print(f"\n{'contrast':>44} {'delta':>9} {'se':>7} {'t':>7} {'wins':>7}")
    for a, b in [("b6 +FGS(e17) f=1/7", "base6 (claim Q6)"),
                 ("b7 +FGS(e17) f=1/7", "base7 (=Q6+sh3)"),
                 ("b7 +FGS(e15) f=1/7", "base7 (=Q6+sh3)"),
                 ("b7 +FGS(e16app) f=1/7", "base7 (=Q6+sh3)"),
                 ("b7 +CLONE(ut60k) f=1/7", "base7 (=Q6+sh3)"),
                 ("base7 (=Q6+sh3)", "base6 (claim Q6)")]:
        d, se, t, w = paired(a, b)
        print(f"{a+'  vs  '+b:>44} {d:+9.4f} {se:7.4f} {t:+7.2f} {w:4d}/{N}")


if __name__ == "__main__":
    main()
