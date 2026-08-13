#!/usr/bin/env python
"""DIVERSITY PROBE -- generic arm runner on the production harness, FULL SHIPPED CHAIN.

chain per arm: uniform member mean -> energy restore (lam, k = true member count)
               -> median lens field x gain, INTER_LANCZOS4 -> JPEG q100/ss2 progressive
               -> PSNR/SSIM(repo)/LPIPS(vgg) vs REAL test GT, 60 views of HCM0181.

Every arm is a member SUBSET; all Laplacian/energy work is hoisted per stem (see div_lib).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from div_lib import UT4, ALL, FAM, RD, GTD, ld, prep, combine, chain, load_field, stems
from lapfuse import _K
from fieldlib import warp
from PIL import Image


def build_arms(which):
    U1, U2, U3, U4 = UT4
    P3 = [U1, U2, U3]
    others = [m for m in ALL if m not in UT4]
    A = {}
    if which == "solo":
        A["base_k4UT"] = UT4
        for m in ALL:
            A[f"solo:{m}"] = [m]
    elif which == "add5":
        A["base_k4UT"] = UT4
        for m in others:
            A[f"+{m}"] = UT4 + [m]
        A["NOISECTL"] = "noise"          # synthetic decorrelated-but-signal-free 5th member
        A["CLONECTL"] = "clone"          # exact duplicate of U3 as a 5th member (zero diversity)
    elif which == "fixk":
        # COUNT-CONTROLLED CONTROL: hold k=4 fixed and vary only the FAMILY of the 4th member.
        # Isolates family from pool size. Junk members (sh0/sh1/sh2, bilagrid) excluded --
        # they measure damage, not family.
        A["base_k3UT"] = P3
        A["k4_sameFAM(+B12ut8Ms7)"] = UT4
        for m in ["m31b_taillpips", "m31b_nolpips", "gsplatB8pure", "gsplatB4warm",
                  "gsplatB7ppisp2", "gsplatB1", "gsplatB2", "e17visnorm", "e15ceil95",
                  "e16app"]:
            A[f"k4_+{m}"] = P3 + [m]
    elif which == "wsweep":
        # THE OPERATIONALLY DECISIVE AXIS. Round-13 shipped 3 FastGS-family members on the set2
        # towers at 0.40 total weight and graded -0.5961 on the LB. So "add a foreign family"
        # is only a live idea if the optimum weight is well below 0.4. Sweep the foreign
        # member's SHARE f of the ensemble: UT4 carry weight 1 each, foreign carries 4f/(1-f).
        A["base_k4UT"] = UT4
        for nm, m in (("AA(B8pure)", "gsplatB8pure"), ("FGS(e17)", "e17visnorm")):
            for f in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
                A[f"{nm} f={f:.2f}"] = (UT4 + [m], [1, 1, 1, 1, 4 * f / (1 - f)])
        # both foreign families together, each at f/2 -- the r30 candidate composition
        for f in (0.10, 0.20, 0.30):
            w = 4 * (f / 2) / (1 - f)
            A[f"AA+FGS f={f:.2f}"] = (UT4 + ["gsplatB8pure", "e17visnorm"], [1, 1, 1, 1, w, w])
    elif which == "ladder":
        # THE OPERATIONAL EXPERIMENT: marginal value of the NEXT member vs current pool size k,
        # for a SAME-FAMILY add and for a FOREIGN-FAMILY add. Production towers sit at k=8, all
        # UT, so the decision for r30 is the k=8 -> k=9 marginal.
        # Q_k is grown by same-family (UT) adds only. NOTE: HCM0181 only has 4 genuinely
        # independent UT trainings; Q5..Q8 are grown with UT DERIVATIVES (8k refits of B11ut60k
        # and SH-clamped re-renders of it), which are the MOST correlated members obtainable.
        # The depth ladder above k=4 is therefore a LOWER bound on same-family marginal value,
        # and the k<=4 rungs are the honest same-family numbers.
        # HCM0181 has exactly SIX usable UT-family members (4 independent UT trainings +
        # 2 metric-loss refits of B11ut60k). The SH-clamped re-renders sh0/sh1/sh2 are NOT
        # usable pool members -- measured solo through the chain they score 68.80/70.21/73.75
        # against 76.8-77.4 for real members -- so growing the ladder with them would measure
        # member DAMAGE, not depth. Q therefore stops at 6 and k=8->9 is extrapolated.
        Q = {2: [U1, U2], 3: [U1, U2, U3], 4: UT4}
        Q[5] = Q[4] + ["m31b_taillpips"]
        Q[6] = Q[5] + ["m31b_nolpips"]
        for k in range(2, 7):
            A[f"Q{k}_UTonly"] = Q[k]
        for k in (2, 3, 4, 5, 6):
            A[f"Q{k}+AA(B8pure)"] = Q[k] + ["gsplatB8pure"]
        for k in (3, 4, 6):
            A[f"Q{k}+FGS(e17)"] = Q[k] + ["e17visnorm"]
        A["Q6+AA+FGS"] = Q[6] + ["gsplatB8pure", "e17visnorm"]
        A["Q6+4foreign"] = Q[6] + ["gsplatB8pure", "gsplatB4warm", "e17visnorm", "e15ceil95"]
    elif which == "grow":
        A["base_k4UT"] = UT4
        A["div6 +AA+FGS"] = UT4 + ["gsplatB8pure", "e17visnorm"]
        A["div7 +2AA+FGS"] = UT4 + ["gsplatB8pure", "gsplatB4warm", "e17visnorm"]
        A["div7 +AA+2FGS"] = UT4 + ["gsplatB8pure", "e17visnorm", "e15ceil95"]
        A["div8 +2AA+2FGS"] = UT4 + ["gsplatB8pure", "gsplatB4warm", "e17visnorm", "e15ceil95"]
        A["div6 +AAworst+FGS"] = UT4 + ["gsplatB2", "e17visnorm"]
        A["w: UT4 x1 + AA x2 (dbl)"] = (UT4 + ["gsplatB8pure", "e17visnorm"],
                                        [1, 1, 1, 1, 2, 2])
    else:
        raise SystemExit(which)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=1.30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", action="store_true", help="also dump residual Gram matrix")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K = _K.to(dev)
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    ss, gt_by = stems()
    lens = load_field(args.gain)
    arms = build_arms(args.arms)
    names = list(arms)
    acc = {a: np.zeros(3) for a in names}
    nb = {a: 0 for a in names}
    def _mem(v):
        if isinstance(v, tuple):
            return v[0]
        return v if isinstance(v, list) else []
    n_used = sorted({m for v in arms.values() for m in _mem(v)})
    need = sorted(set(n_used) | set(UT4) | (set(ALL) if args.stats else set()))
    G = np.zeros((len(ALL), len(ALL)))     # residual Gram <e_i,e_j>
    dec = np.zeros(len(ALL))               # mean |m - UT4mean| in 0..255
    solo_mse = np.zeros(len(ALL))
    rng = np.random.RandomState(0)

    t0 = time.time()
    for c, s in enumerate(ss):
        X = {m: ld(os.path.join(RD(m), s + ".png"), dev) for m in need}
        P = {m: prep(X[m], K) for m in need}
        L0 = {m: P[m][0] for m in need}
        Ae = {m: P[m][1] for m in need}
        g = ld(os.path.join(GTD, gt_by[s]), dev)

        base = torch.stack([X[m] for m in UT4]).mean(0)
        if args.stats:
            E = torch.stack([(X[m] - g)[0] for m in ALL]).reshape(len(ALL), -1)
            G += (E @ E.T).cpu().numpy() / E.shape[1]
            for i, m in enumerate(ALL):
                dec[i] += float((X[m] - base).abs().mean()) * 255.0
                solo_mse[i] += float(((X[m] - g) ** 2).mean())

        # synthetic controls, built inside the stem loop so they share the same chain
        if "NOISECTL" in arms:
            devi = torch.stack([X[m] - base for m in UT4]).abs().mean()
            nz = torch.from_numpy(rng.randn(*base.shape).astype(np.float32)).to(dev)
            fake = (base + nz * float(devi) * 1.2533).clamp(0, 1)   # match mean|dev| of a member
            X["__noise__"] = fake
            L0["__noise__"], Ae["__noise__"] = prep(fake, K)
        if "CLONECTL" in arms:
            X["__clone__"] = X[UT4[2]]
            L0["__clone__"], Ae["__clone__"] = L0[UT4[2]], Ae[UT4[2]]

        for a in names:
            idx, w = arms[a], None
            if idx == "noise":
                idx = UT4 + ["__noise__"]
            elif idx == "clone":
                idx = UT4 + ["__clone__"]
            elif isinstance(idx, tuple):
                idx, w = idx
            out = combine(X, L0, Ae, idx, args.lam, w)
            j, bts = chain(out, lens, warp)
            nb[a] += bts
            r = torch.from_numpy(j).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                acc[a][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[a][1] += float(repo_ssim(r, g))
                acc[a][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
        if c % 10 == 0:
            print(f"  {c}/{len(ss)}  {time.time()-t0:.0f}s", flush=True)

    N = len(ss)
    res = {}
    for a in names:
        P_, S_, L_ = acc[a] / N
        res[a] = dict(score=100 * (0.4 * (1 - L_) + 0.3 * S_ + 0.3 * min(P_ / 50.0, 1.0)),
                      psnr=P_, ssim=S_, lpips=L_, mb=nb[a] / 1e6,
                      members=(_mem(arms[a]) or [str(arms[a])]))
    payload = dict(n=N, lam=args.lam, gain=args.gain, arms=res)
    if args.stats:
        payload["gram"] = (G / N).tolist()
        payload["order"] = ALL
        payload["decorr255"] = (dec / N).tolist()
        payload["solo_mse"] = (solo_mse / N).tolist()
    json.dump(payload, open(args.out, "w"), indent=1)

    base_name = "base_k4UT" if "base_k4UT" in res else names[0]
    b = res[base_name]["score"]
    print(f"\n{args.arms}  n={N}  lam={args.lam}  fieldgain={args.gain}  "
          f"FULL SHIPPED CHAIN")
    print(f"{'arm':>26} {'fam':>4} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7} "
          f"{'vs base':>9} {'MB/60':>7}")
    for a in sorted(names, key=lambda z: -res[z]["score"]):
        r = res[a]
        mm = r["members"][-1]
        f = FAM.get(mm, "-")
        print(f"{a:>26} {f:>4} {r['score']:9.4f} {r['psnr']:8.4f} {r['ssim']:7.4f} "
              f"{r['lpips']:7.4f} {r['score']-b:+9.4f} {r['mb']:7.1f}")


if __name__ == "__main__":
    main()
