#!/usr/bin/env python
"""DEPTH OR DIVERSITY? Which 5th member is worth more -- same family, or a different one?

Every tower member we ship is UT-family (ut7/13/42/77, seed101, and BOTH mip3d members train with
--ut). r30 was going to be a 3rd mip3d, i.e. more DEPTH in the family we already have. But the
largest unclaimed number in the campaign is a control result: adding TWO DIFFERENT-FAMILY members
to a 4-member UT pool moved the harness +0.2608, against the +0.02..0.04 a same-family add is
worth. Pixel-mean ensembling pays in proportion to how DECORRELATED the members' errors are, and
same-family members share their errors.

So: on the production harness, through the FULL shipped chain (mean -> energy restore lam=1.0 ->
median field lanczos4 -> JPEG q100/ss2), take the 4-member UT pool and add each candidate 5th
member in turn. gsplatB8pure is the non-UT one -- the family the private towers do not have.

Reported alongside each score: the candidate's DECORRELATION (mean |member - pool_mean|), because
the thesis predicts score gain should track decorrelation, not solo quality. If it does, r30 should
be a non-UT tower member. If gsplatB8pure lands mid-pack, a 3rd mip3d is the safer r30 and this
idea dies cheaply.
"""
import io, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from energy_restore import restore
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
D = lambda m: f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png"
POOL = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]   # the UT family
CANDS = ["gsplatB8pure",      # NON-UT -- the family the towers lack
         "e17visnorm",        # different densification rule
         "m31b_taillpips",    # different loss schedule
         "gsplatB6bilagrid",  # bilateral grid appearance
         "e15ceil95",         # different cap policy
         "gsplatB7ppisp2"]    # per-pixel ISP


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
    dirs = [D(m) for m in POOL] + [D(c) for c in CANDS]
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in cache["HW"]]
    lens = upsample(LooPool(cache["s8"]).pooled("median"), H, W, "cubic")

    arms = ["k4_UT_only"] + [f"+{c}" for c in CANDS]
    acc = {a: [0.0, 0.0, 0.0] for a in arms}
    dec = {c: 0.0 for c in CANDS}
    t0 = time.time()
    for n, s in enumerate(stems):
        ut = [ld(os.path.join(D(m), s + ".png")) for m in POOL]
        base = torch.stack(ut).mean(0)
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                                        dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        outs = {"k4_UT_only": (base, ut)}
        for c in CANDS:
            m = ld(os.path.join(D(c), s + ".png"))
            dec[c] += float((m - base).abs().mean()) * 255.0
            mem = ut + [m]
            outs[f"+{c}"] = (torch.stack(mem).mean(0), mem)
        for a in arms:
            ens, mem = outs[a]
            o = restore(ens, mem, 1.0, len(mem)).clamp(0, 1)[0].permute(1, 2, 0).numpy()
            x = np.clip(warp(o, lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                acc[a][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[a][1] += float(repo_ssim(r, g))
                acc[a][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
        if n % 10 == 0:
            print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    print(f"\n5th-MEMBER CHOICE, {TAG}, n={N}, full shipped chain (restore lam=1.0 -> field -> JPEG)")
    print(f"{'arm':>20} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'vs k4':>8} {'decorr/255':>11}")
    base_sc = None
    rows = []
    for a in arms:
        P, S, L = (x / N for x in acc[a])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        if a == "k4_UT_only":
            base_sc = sc
        d = dec.get(a[1:], float("nan")) / N
        rows.append((a, sc, P, S, L, sc - base_sc, d))
    for a, sc, P, S, L, dl, d in rows:
        ds = f"{d:11.4f}" if d == d else f"{'--':>11}"
        print(f"{a:>20} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {dl:+8.4f} {ds}")
    good = [(dl, d, a) for a, sc, P, S, L, dl, d in rows if d == d]
    if len(good) > 2:
        x = np.array([g[1] for g in good]); y = np.array([g[0] for g in good])
        print(f"\ncorr(decorrelation, score gain) = {np.corrcoef(x, y)[0,1]:+.3f}"
              f"   -- the diversity thesis predicts this is POSITIVE")


if __name__ == "__main__":
    main()
