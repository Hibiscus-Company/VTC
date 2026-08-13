#!/usr/bin/env python
"""WHY DID r23 LOSE? Hypothesis: the JPEG "tax" inverts with ensemble depth.

I measured the tax on a SINGLE-member bonsai render and found keep_rgb worth +1.01. On the real
leaderboard the same change made LPIPS WORSE. My calibration checked that encode DAMAGE (self-LPIPS,
png vs jpeg round-trip) was independent of ensemble depth -- it was -- but self-damage is NOT score
impact. Score impact is |encoded - GT| vs |png - GT|, which depends on where the render sits
RELATIVE TO GT, not on how far the encode moves it.

Mechanism I now suspect: a deep pixel-mean ensemble is SMOOTHER than GT (averaging destroys the
per-view stochastic texture LPIPS-vgg keys on). JPEG's q100-ss2 artifacts inject high-frequency
structure. On a smooth ensemble that pushes texture statistics TOWARD real GT (LPIPS improves);
on a noisy 1-member render it overshoots (LPIPS worsens). If so, our "lossless" keep_rgb encode
strips a free perceptual crutch -- and the deeper the ensemble, the more it costs.

TEST: chair, where I have 4 members. Score k=1,2,4-member means under each encode vs real GT.
If (keep_rgb - shipped) shrinks or flips as k grows, the hypothesis is confirmed and r23 is explained.
"""
import io, os, sys, itertools
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None

MEMBERS = [
    "/mnt/d/avv/tw_test/chair_ema099/eval_png",
    "/mnt/d/avv/tw_test/chair_ema999/eval_png",
    "/mnt/d/avv/tw_test/chair_capmax2M/eval_png",
    "/mnt/d/avv/tw_test/chair_aniso01/eval_png",
]
GT = "/mnt/d/avv/evalsplit/chair/eval_gt"
ENCODES = [
    ("png (no encode)", None),
    ("q100 ss2 [SHIPPED]", dict(quality=100, subsampling=2, optimize=True, progressive=True)),
    ("q98 ss0 keep_rgb", dict(quality=98, subsampling=0, optimize=True, progressive=True, keep_rgb=True)),
]


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    gt_by = {os.path.splitext(f)[0]: os.path.join(GT, f) for f in os.listdir(GT)}
    stems = sorted(gt_by)
    print(f"chair: {len(stems)} poses, {len(MEMBERS)} members available\n")
    print(f"{'k':>2} {'encode':22s} {'SCORE':>9} {'LPIPS':>8} {'SSIM':>7} {'PSNR':>8}   delta_vs_png")

    for k in (1, 2, 4):
        dirs = MEMBERS[:k]
        base = {}
        for name, kw in ENCODES:
            P = S = L = 0.0
            for st in stems:
                acc = None
                for d in dirs:
                    a = np.asarray(Image.open(os.path.join(d, st + ".png")).convert("RGB"),
                                   dtype=np.float32)
                    acc = a if acc is None else acc + a
                mean = acc / len(dirs)
                im = Image.fromarray(np.clip(mean + 0.5, 0, 255).astype(np.uint8))
                if kw is not None:
                    b = io.BytesIO(); im.save(b, "JPEG", **kw); b.seek(0)
                    im = Image.open(b).convert("RGB")
                r = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
                g = torch.from_numpy(np.asarray(Image.open(gt_by[st]).convert("RGB"), np.float32) / 255.0
                                     ).permute(2, 0, 1)[None].to(dev)
                with torch.no_grad():
                    P += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                    S += float(repo_ssim(r, g))
                    L += float(vgg(r * 2 - 1, g * 2 - 1).item())
            n = len(stems)
            P, S, L = P / n, S / n, L / n
            sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
            base[name] = sc
            d0 = sc - base["png (no encode)"]
            print(f"{k:>2} {name:22s} {sc:9.4f} {L:8.4f} {S:7.4f} {P:8.4f}   {d0:+8.4f}")
        print(f"   -> keep_rgb minus shipped at k={k}: "
              f"{base['q98 ss0 keep_rgb'] - base['q100 ss2 [SHIPPED]']:+.4f}\n")


if __name__ == "__main__":
    main()
