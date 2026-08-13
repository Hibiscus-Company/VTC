#!/usr/bin/env python
"""PHASE C -- why does a level-l term help or hurt?  The analytic MSE answer, per level.

The operator adds an increment  d_l = lam * (r_l - 1) * L_l(mean)  to band l.  In the REGISTERED
domain (after the lens-field warp, which is where the score is taken) the exact MSE change is

    dMSE(lam) = ( 2*lam*<d1_l, mean - gt> + lam^2*||d1_l||^2 ) / N      d1_l = increment at lam=1

so the MSE-optimal lambda for that band is  lam*_l = -<d1_l, mean-gt> / ||d1_l||^2  and the
alignment  a_l = -<d1_l, gt-mean-direction> tells us how much of the band's deficit is COHERENT.
Level 0 is known to be mostly incoherent (96% of missing HF does not correlate with GT); the open
question is whether level 1, being a lower-frequency band, is more coherent -- which is exactly the
condition under which a level-1 term could pay where level-0 sharpening cannot.

d1 is measured through the real warp (warp(restored) - warp(mean)), not assumed linear.
CPU only; one pass; member pyramids hoisted.
"""
import json, os, sys, time
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
TAG = "HCM0181"
D = lambda m: f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png"
UT4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
DIV4 = ["gsplatB8pure", "e17visnorm", "m31b_taillpips", "gsplatB6bilagrid"]
POOL8 = UT4 + DIV4
NLEV, WIN, CLAMP = 5, 3, 4.0


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
    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(D(m), s + ".png")) for m in POOL8))
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in cache["HW"]]
    lens = upsample(LooPool(cache["s8"]).pooled("median"), H, W, "cubic")
    wp = lambda t: np.clip(warp(t.clamp(0, 1)[0].permute(1, 2, 0).numpy(), lens, "lanczos"), 0, 1)

    stat = {p: {l: np.zeros(4) for l in (0, 1)} for p in ("p8", "p4")}   # <d,e>, ||d||^2, ||e||^2, N
    t0 = time.time()
    for c, s in enumerate(stems):
        gt = np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                        dtype=np.float32) / 255.0
        mem = [ld(os.path.join(D(m), s + ".png")) for m in POOL8]
        pyr = [lap_pyr(m, NLEV, _K) for m in mem]                     # HOISTED
        for tag, idx in (("p8", list(range(8))), ("p4", list(range(4)))):
            k = len(idx)
            laps = [torch.stack([pyr[i][0][l] for i in idx]).mean(0) for l in range(NLEV)]
            res = torch.stack([pyr[i][1] for i in idx]).mean(0)
            sizes = pyr[0][2]
            base_w = wp(lap_recon(laps, res, sizes, _K))
            err = base_w - gt                                          # mean - gt, registered
            for l in (0, 1):
                r = rmap([pyr[i][0][l] for i in idx], laps[l], k)
                out = list(laps)
                out[l] = out[l] * r                                    # lam = 1
                d = wp(lap_recon(out, res, sizes, _K)) - base_w        # d1 through the REAL warp
                stat[tag][l] += np.array([float((d * err).sum()), float((d * d).sum()),
                                          float((err * err).sum()), d.size])
        if c % 10 == 0:
            print(f"  {c}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    out = {}
    print(f"\nPER-LEVEL MSE GEOMETRY of the energy increment, {TAG}, n={len(stems)}, "
          f"registered (post-field) domain")
    print(f"{'pool':>5} {'lev':>4} {'lam*_MSE':>9} {'dMSE@lam1':>11} {'dPSNR@lam1':>11} "
          f"{'coherence':>10} {'||d||/||e||':>12}")
    for tag in ("p8", "p4"):
        de, dd, ee, N = None, None, None, None
        for l in (0, 1):
            de, dd, ee, N = stat[tag][l]
            lam_opt = -de / dd
            mse0 = ee / N
            dmse = (2 * de + dd) / N
            dpsnr = 10 * np.log10(mse0 / (mse0 + dmse))
            coh = -de / np.sqrt(dd * ee)
            print(f"{tag:>5} {l:>4} {lam_opt:9.4f} {dmse:+11.3e} {dpsnr:+11.4f} "
                  f"{coh:+10.4f} {np.sqrt(dd/ee):12.4f}")
            out[f"{tag}_L{l}"] = dict(lam_opt=lam_opt, dmse=dmse, dpsnr=dpsnr, coh=coh,
                                      dnorm_over_enorm=float(np.sqrt(dd / ee)))
    json.dump(out, open(f"{HERE}/lvl1_C_optlam.json", "w"), indent=1)
    print("\nlam*_MSE>0 means the band increment points TOWARD gt; coherence is the cosine "
          "between the increment and the residual it is trying to fill.")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
