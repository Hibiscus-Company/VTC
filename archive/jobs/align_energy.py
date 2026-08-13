#!/usr/bin/env python
"""DO ALIGN-THEN-MERGE AND ENERGY RESTORATION COMPOSE, AND WHICH IS WORTH MORE PER BYTE?

Both recover the SAME deficit -- the finest-band energy that pixel-mean ensembling destroys -- by
opposite routes:
  align   warps every member into the consensus geometry before averaging, so the true detail is
          not smeared in the first place. Recovers REAL structure.
  energy  leaves the mean alone and rescales its finest Laplacian band by the disagreement map.
          SYNTHESISES the missing amplitude.
So they are plausibly substitutes, not complements: aligning first shrinks the disagreement map
that energy restoration feeds on. That has to be measured, not assumed.

And they compete for the same scarce resource. r28 ships at ~349 MB against a hard 350 MB cap, so
the question is not "which scores more" but "which scores more PER BYTE", and whether the pair
together beats either alone at equal cost. Every arm is therefore scored through the full shipped
chain AND weighed.
"""
import io, os, sys, time
import numpy as np
import cv2
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
MEM = [f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png"
       for m in ("gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7")]


def mkdis():
    d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    d.setFinestScale(0); d.setPatchSize(8); d.setPatchStride(3)
    d.setUseMeanNormalization(True); d.setUseSpatialPropagation(True)
    d.setVariationalRefinementIterations(5)
    return d


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    DIS = mkdis()

    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in cache["HW"]]
    lens = upsample(LooPool(cache["s8"]).pooled("median"), H, W, "cubic")
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    g8 = lambda x: cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    def align_merge(X, ref, clamp=1.0):
        gm = g8(ref); acc = np.zeros_like(ref)
        for i in range(X.shape[0]):
            F = DIS.calc(gm, g8(X[i]), None)
            m = np.sqrt((F ** 2).sum(-1, keepdims=True))
            F = F * np.minimum(1.0, clamp / np.maximum(m, 1e-6))
            acc += cv2.remap(X[i], gx + F[..., 0], gy + F[..., 1],
                             cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        return acc / X.shape[0]

    def er(mean_np, mem_np, lam):
        if lam <= 0:
            return mean_np
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0)
        o = restore(t(mean_np), [t(m) for m in mem_np], lam, len(mem_np))
        return o.clamp(0, 1)[0].permute(1, 2, 0).numpy()

    LAMS = [0.75, 1.0, 1.25, 1.5, 2.0]
    arms = ["base"] + [f"e{l}" for l in LAMS] + ["align"] + [f"align+e{l}" for l in (0.5, 1.0, 1.5)]
    acc = {a: [0.0, 0.0, 0.0] for a in arms}
    nb = {a: 0 for a in arms}
    t0 = time.time()
    for c, s in enumerate(stems):
        X = np.stack([np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                                 dtype=np.float32) / 255.0 for d in MEM])
        M = X.mean(0)
        A = align_merge(X, M)
        Xa = list(X)
        outs = {"base": M, "align": A}
        for l in LAMS:
            outs[f"e{l}"] = er(M, Xa, l)
        for l in (0.5, 1.0, 1.5):
            outs[f"align+e{l}"] = er(A, Xa, l)
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                                        dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        for a in arms:
            x = np.clip(warp(np.clip(outs[a], 0, 1), lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            nb[a] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                acc[a][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[a][1] += float(repo_ssim(r, g))
                acc[a][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
        if c % 10 == 0:
            print(f"  {c}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    n = len(stems)
    print(f"\nFULL SHIPPED CHAIN, {TAG}, n={n}   (k=4; production ships 7, and the gain grows with k)")
    print(f"{'arm':>14} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'d(base)':>9} "
          f"{'MB':>7} {'dMB':>7} {'score/MB':>9}")
    base_sc = base_mb = None
    for a in arms:
        P, S, L = (x / n for x in acc[a])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        mb = nb[a] / 1e6
        if a == "base":
            base_sc, base_mb = sc, mb
        d, dmb = sc - base_sc, mb - base_mb
        eff = f"{d/dmb:9.4f}" if dmb > 1e-6 else f"{'--':>9}"
        print(f"{a:>14} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {d:+9.4f} {mb:7.2f} {dmb:+7.3f} {eff}")


if __name__ == "__main__":
    main()
