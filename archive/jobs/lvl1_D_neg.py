#!/usr/bin/env python
"""PHASE D -- the level-1 term with the OPPOSITE sign, plus the analytic reason.

Phase B: a level-1 energy BOOST is monotonically negative (lam1 0.25/0.5/1.0 -> -0.0042/-0.0098/
-0.0241 at k=8), and the shuffled-map control is much worse (-0.1027), so the disagreement map at
level 1 is genuinely informative -- it is the SIGN that is wrong.  Reading that trend backwards,
lam1 < 0 SHRINKS band 1 exactly where the members disagree, which is the Wiener/Bayes-correct move
for an unreliable estimate.  Untested: the known dead attenuation result was flat-region mu*f on
LEVEL 0, a different operator on a different band.

Also computes, in the REGISTERED (post-field) domain, the exact MSE geometry of each band's
increment d1_l = (r_l - 1) * L_l(mean):
    lam*_MSE = -<d1,err>/||d1||^2 ,  coherence = -<d1,err>/(||d1||*||err||) ,  err = mean - gt
which says how much of each band's energy deficit is COHERENT with what is actually missing.

Everything shares ONE pass of member pyramids.  CPU except LPIPS.
"""
import io, json, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
D = lambda m: f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png"
POOL8 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
         "gsplatB8pure", "e17visnorm", "m31b_taillpips", "gsplatB6bilagrid"]
NLEV, WIN, CLAMP = 5, 3, 4.0
# (name, lam0, lam1, map1)
ARMS = [("base(shipped)",     1.0,  0.00, "raw"),
        ("l1_-0.25",          1.0, -0.25, "raw"),
        ("l1_-0.50",          1.0, -0.50, "raw"),
        ("l1_-1.00",          1.0, -1.00, "raw"),
        ("CTRL_shuf_-0.50",   1.0, -0.50, "shuffle"),
        ("no_restore(lam0=0)",0.0,  0.00, "raw")]


def ld(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def rmap(laps_i, Lm, k):
    Eb = boxf((Lm ** 2).sum(1, keepdim=True), WIN)
    V = 0.0
    for Li in laps_i:
        V = V + boxf(((Li - Lm) ** 2).sum(1, keepdim=True), WIN)
    V = V / len(laps_i) * (k / (k - 1.0))
    return torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=CLAMP)


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(D(m), s + ".png")) for m in POOL8))
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in cache["HW"]]
    lens = upsample(LooPool(cache["s8"]).pooled("median"), H, W, "cubic")
    wp = lambda t: np.clip(warp(t.clamp(0, 1)[0].permute(1, 2, 0).numpy(), lens, "lanczos"), 0, 1)

    names = [a[0] for a in ARMS]
    acc = {n: [0.0, 0.0, 0.0] for n in names}
    geo = {l: np.zeros(4) for l in (0, 1)}
    t0 = time.time()
    for c, s in enumerate(stems):
        gtn = np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                         dtype=np.float32) / 255.0
        g = torch.from_numpy(gtn).permute(2, 0, 1).unsqueeze(0).to(dev)
        pyr = [lap_pyr(ld(os.path.join(D(m), s + ".png")), NLEV, _K) for m in POOL8]  # HOISTED
        k = len(pyr)
        laps = [torch.stack([p[0][l] for p in pyr]).mean(0) for l in range(NLEV)]
        res = torch.stack([p[1] for p in pyr]).mean(0)
        sizes = pyr[0][2]
        r0 = rmap([p[0][0] for p in pyr], laps[0], k)
        r1 = rmap([p[0][1] for p in pyr], laps[1], k)
        f1 = r1.reshape(-1)
        gen = torch.Generator().manual_seed(1234 + c)
        R1 = {"raw": r1, "shuffle": f1[torch.randperm(f1.numel(), generator=gen)].reshape(r1.shape)}

        # ---- MSE geometry (no jpeg/lpips, reuses the same bands) ----
        plain = wp(lap_recon(laps, res, sizes, _K))
        err = plain - gtn
        for l, rr in ((0, r0), (1, r1)):
            o = list(laps); o[l] = o[l] * rr
            d = wp(lap_recon(o, res, sizes, _K)) - plain
            geo[l] += np.array([float((d * err).sum()), float((d * d).sum()),
                                float((err * err).sum()), d.size])

        for name, lam0, lam1, m1 in ARMS:
            o = list(laps)
            if lam0 != 0.0:
                o[0] = o[0] * (1.0 + lam0 * (r0 - 1.0))
            if lam1 != 0.0:
                o[1] = o[1] * (1.0 + lam1 * (R1[m1] - 1.0))
            x = wp(lap_recon(o, res, sizes, _K))
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                acc[name][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[name][1] += float(repo_ssim(r, g))
                acc[name][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
        if c % 10 == 0:
            print(f"  {c}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    print(f"\nNEGATIVE lam1 (band-1 SHRINKAGE), {TAG}, n={N}, k=8, FULL shipped chain")
    print(f"{'arm':>20} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'vs base':>9}")
    out = {}
    base = None
    for name, *_ in ARMS:
        P, S, L = (x / N for x in acc[name])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        if name == "base(shipped)":
            base = sc
        out[name] = dict(score=sc, psnr=P, ssim=S, lpips=L, delta=sc - base)
        print(f"{name:>20} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {sc-base:+9.4f}")

    print(f"\nMSE GEOMETRY of each band's increment (registered domain, k=8)")
    print(f"{'lev':>4} {'lam*_MSE':>9} {'dPSNR@lam1':>11} {'coherence':>10} {'||d||/||err||':>14}")
    for l in (0, 1):
        de, dd, ee, n = geo[l]
        mse0 = ee / n
        dmse = (2 * de + dd) / n
        rec = dict(lam_opt=-de / dd, dpsnr=10 * np.log10(mse0 / (mse0 + dmse)),
                   coh=-de / np.sqrt(dd * ee), dnorm=float(np.sqrt(dd / ee)))
        out[f"geo_L{l}"] = rec
        print(f"{l:>4} {rec['lam_opt']:9.4f} {rec['dpsnr']:+11.4f} {rec['coh']:+10.4f} "
              f"{rec['dnorm']:14.4f}")
    json.dump(out, open(f"{HERE}/lvl1_D_neg.json", "w"), indent=1)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
