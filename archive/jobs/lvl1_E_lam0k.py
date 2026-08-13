#!/usr/bin/env python
"""PHASE E -- lam0 was swept at k=4.  Production ships k=7-8.  Does the optimum move?

Phase A: the level-0 energy deficit DEEPENS with pool size (E(L0_mean)/E(L0_gt) 0.765 at k=1 ->
0.653 at k=8) and even after a FULL lam0=1.0 correction the restored band still sits at 0.729 of
GT energy -- the correction is under-powered, and more so the bigger the pool.  Phase D confirmed
the payoff scales: turning the level-0 term off costs -0.2940 at k=8 against the -0.1883 recorded
at k=4.  The shipped lam0=1.0 was chosen on the k=4 curve (0.75 +0.1756, 1.0 +0.1883, 1.25 +0.1820,
1.5 +0.1570).  If the optimum has drifted up with k, lam0 is a one-character change to build_r29.

Both pools in one pass so the k=4 arms reproduce the known curve as a sanity check.
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
ARMS = [("p8", l) for l in (0.0, 0.75, 1.0, 1.25, 1.5)] + \
       [("p4", l) for l in (1.0, 1.25)]


def ld(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


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

    names = [f"{p}_lam{l}" for p, l in ARMS]
    acc = {n: [0.0, 0.0, 0.0] for n in names}
    nb = {n: 0 for n in names}
    t0 = time.time()
    for c, s in enumerate(stems):
        g = ld(os.path.join(gtd, gt_by[s])).to(dev)
        pyr = [lap_pyr(ld(os.path.join(D(m), s + ".png")), NLEV, _K) for m in POOL8]   # HOISTED
        prep = {}
        for tag, idx in (("p8", range(8)), ("p4", range(4))):
            idx = list(idx); k = len(idx)
            laps = [torch.stack([pyr[i][0][l] for i in idx]).mean(0) for l in range(NLEV)]
            res = torch.stack([pyr[i][1] for i in idx]).mean(0)
            Lm = laps[0]
            Eb = boxf((Lm ** 2).sum(1, keepdim=True), WIN)
            V = 0.0
            for i in idx:
                V = V + boxf(((pyr[i][0][0] - Lm) ** 2).sum(1, keepdim=True), WIN)
            V = V / k * (k / (k - 1.0))
            prep[tag] = (laps, res, pyr[0][2],
                         torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=CLAMP))
        for (tag, lam), name in zip(ARMS, names):
            laps, res, sizes, r0 = prep[tag]
            o = list(laps)
            if lam != 0.0:
                o[0] = o[0] * (1.0 + lam * (r0 - 1.0))
            x = wp(lap_recon(o, res, sizes, _K))
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            nb[name] += len(b.getvalue())
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
    print(f"\nlam0 RE-SWEEP AT PRODUCTION k, {TAG}, n={N}, FULL shipped chain")
    print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'vs lam1.0':>10} {'MB/60':>7}")
    out = {}
    b = {"p8": None, "p4": None}
    for (tag, lam), name in zip(ARMS, names):
        P, S, L = (x / N for x in acc[name])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        if lam == 1.0:
            b[tag] = sc
        out[name] = dict(score=sc, psnr=P, ssim=S, lpips=L, mb=nb[name] / 1e6)
    for (tag, lam), name in zip(ARMS, names):
        d = out[name]["score"] - b[tag]
        out[name]["delta_vs_lam1"] = d
        print(f"{name:>12} {out[name]['score']:9.4f} {out[name]['psnr']:8.4f} "
              f"{out[name]['ssim']:7.4f} {out[name]['lpips']:8.4f} {d:+10.4f} "
              f"{out[name]['mb']:7.2f}")
    json.dump(out, open(f"{HERE}/lvl1_E_lam0k.json", "w"), indent=1)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
