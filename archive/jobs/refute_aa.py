#!/usr/bin/env python
"""ADVERSARIAL RE-MEASUREMENT of "AA member given FGS is worth +0.0447 tower".

The claim's number is the 7->8 increment  Q6+FGS+AA  -  Q6+FGS  with AA = gsplatB8pure.
Two controls the original probe never ran AT THAT STEP:
  (a) CLONE  -- duplicate an existing pool member: adds ZERO information but the SAME k bump
                and the SAME dilution of the pool's worst member. This is the true zero point.
  (b) FGS2   -- a second member of the family already present. If it beats AA, "foreign family"
                is not what is being bought.
And the recipe control the original probe never ran AT ALL:
  (c) AA_B2 / AA_B1 -- gsplat AA COLD START (--iters 30000 --cap_max 5e6 --noise_stop 25000,
                no --ut). THIS is the recipe the claim proposes to ship. B8pure is a warm-start
                finetune off a FastGS champion .ply, i.e. a FastGS/gsplat hybrid.

Per-view scores are retained so the increments get an honest paired standard error.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from div_lib import UT4, RD, GTD, ld, prep, combine, chain, load_field
from lapfuse import _K
from fieldlib import warp
from PIL import Image

U1, U2, U3, U4 = UT4
Q6 = UT4 + ["m31b_taillpips", "m31b_nolpips"]
FGS = "e17visnorm"
BASE7 = Q6 + [FGS]

ARMS = {
    "Q6":                 Q6,
    "Q6+FGS":             BASE7,
    "Q6+FGS+CLONE(U1)":   BASE7 + ["__cloneU1__"],
    "Q6+FGS+CLONE(FGS)":  BASE7 + ["__cloneFG__"],
    "Q6+FGS+AA_B8pure":   BASE7 + ["gsplatB8pure"],
    "Q6+FGS+AA_B2(SHIP)": BASE7 + ["gsplatB2"],
    "Q6+FGS+AA_B1":       BASE7 + ["gsplatB1"],
    "Q6+FGS+FGS2_e15":    BASE7 + ["e15ceil95"],
}
NEED = sorted({m for v in ARMS.values() for m in v if not m.startswith("__")})


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K = _K.to(dev)
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    ss = sorted(s for s in gt_by
                if all(os.path.exists(os.path.join(RD(m), s + ".png")) for m in NEED))
    print(f"stems={len(ss)}  members={NEED}", flush=True)
    lens = load_field(1.30)
    names = list(ARMS)
    per = {a: [] for a in names}          # per-view [psnr, ssim, lpips]

    t0 = time.time()
    for c, s in enumerate(ss):
        X = {m: ld(os.path.join(RD(m), s + ".png"), dev) for m in NEED}
        P = {m: prep(X[m], K) for m in NEED}
        L0 = {m: P[m][0] for m in NEED}
        Ae = {m: P[m][1] for m in NEED}
        X["__cloneU1__"], L0["__cloneU1__"], Ae["__cloneU1__"] = X[U1], L0[U1], Ae[U1]
        X["__cloneFG__"], L0["__cloneFG__"], Ae["__cloneFG__"] = X[FGS], L0[FGS], Ae[FGS]
        g = ld(os.path.join(GTD, gt_by[s]), dev)

        for a in names:
            out = combine(X, L0, Ae, ARMS[a], 1.0, None)
            j, _ = chain(out, lens, warp)
            r = torch.from_numpy(j).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                p = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                sm = float(repo_ssim(r, g))
                lp = float(vgg(r * 2 - 1, g * 2 - 1).item())
            per[a].append([p, sm, lp])
        if c % 10 == 0:
            print(f"  {c}/{len(ss)}  {time.time()-t0:.0f}s", flush=True)

    A = {a: np.asarray(per[a]) for a in names}
    # per-view score, same formula as the grader
    sc = {a: 100 * (0.4 * (1 - A[a][:, 2]) + 0.3 * A[a][:, 1] + 0.3 * np.minimum(A[a][:, 0] / 50, 1))
          for a in names}
    json.dump({a: A[a].tolist() for a in names}, open(f"{HERE}/refute_aa.json", "w"))

    print(f"\nHCM0181  n={len(ss)}  lam=1.0  gain=1.30  FULL SHIPPED CHAIN (field->JPEG q100/ss2)")
    print(f"{'arm':>22} {'k':>2} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7} {'vs Q6+FGS':>10}")
    b = sc["Q6+FGS"].mean()
    for a in names:
        m = A[a].mean(0)
        print(f"{a:>22} {len(ARMS[a]):>2} {sc[a].mean():9.4f} {m[0]:8.4f} {m[1]:7.4f} "
              f"{m[2]:7.4f} {sc[a].mean()-b:+10.4f}")

    n = len(ss)
    print("\nPAIRED INCREMENTS over Q6+FGS (k=7 -> k=8), with paired SE and t")
    print(f"{'8th member':>22} {'delta':>9} {'SE':>7} {'t':>7} {'vs CLONE(U1)':>13} {'SE':>7} {'t':>7}")
    ctl = sc["Q6+FGS+CLONE(U1)"]
    for a in names:
        if len(ARMS[a]) != 8:
            continue
        d = sc[a] - sc["Q6+FGS"]
        se = d.std(ddof=1) / np.sqrt(n)
        e = sc[a] - ctl
        se2 = e.std(ddof=1) / np.sqrt(n)
        t2 = e.mean() / se2 if se2 > 0 else float("nan")
        print(f"{a:>22} {d.mean():+9.4f} {se:7.4f} {d.mean()/se:7.2f} "
              f"{e.mean():+13.4f} {se2:7.4f} {t2:7.2f}")

    print("\nSPLIT-HALF STABILITY of each 8th-member delta (odd views / even views)")
    print(f"{'8th member':>22} {'odd':>9} {'even':>9}")
    for a in names:
        if len(ARMS[a]) != 8:
            continue
        d = sc[a] - sc["Q6+FGS"]
        print(f"{a:>22} {d[0::2].mean():+9.4f} {d[1::2].mean():+9.4f}")


if __name__ == "__main__":
    main()
