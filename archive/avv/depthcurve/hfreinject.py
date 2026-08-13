#!/usr/bin/env python
"""High-frequency RE-INJECTION combiner for the video scenes.

Motivation (from the r23 post-mortem + the depth curves measured today): a deep pixel-mean
ensemble gains PSNR/SSIM but LOSES LPIPS, because averaging cancels the per-view stochastic
texture that LPIPS-vgg keys on. Plain unsharp (amplifying the MEAN's own high frequencies) was
already measured flat-to-negative. This operator is different: it takes the LOW frequencies
from the deep mean (where the fidelity gain lives) and adds back the HIGH-frequency part of a
single member's deviation from the mean (real render texture, not synthetic grain):

    out = mean_k + alpha * HP( member_1 - mean_k ),   HP(x) = x - gaussian_blur(x, sigma)

alpha=0 is exactly the shipped pixel-mean, so the operator degrades gracefully.
"""
import argparse, os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def gblur(t, sigma):
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="comma list; mean of ALL = base, FIRST = detail donor")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--alphas", default="0,0.3,0.6,1.0")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    mem = args.members.split(",")
    alphas = [float(a) for a in args.alphas.split(",")]
    lp = lpips_pkg.LPIPS(net="vgg").eval()

    gt_by = {os.path.splitext(f)[0]: os.path.join(args.gt, f) for f in os.listdir(args.gt)}
    stems = sorted(gt_by)
    acc = {a: [0.0, 0.0, 0.0] for a in alphas}
    t0 = time.time()
    for n, st in enumerate(stems):
        g = torch.from_numpy(load(gt_by[st])).permute(2, 0, 1).unsqueeze(0)
        ims = []
        for m in mem:
            f = [x for x in os.listdir(m) if os.path.splitext(x)[0] == st][0]
            ims.append(torch.from_numpy(load(os.path.join(m, f))).permute(2, 0, 1).unsqueeze(0))
        mean = torch.stack(ims, 0).mean(0)
        dev = ims[0] - mean
        hp = dev - gblur(dev, args.sigma)
        for a in alphas:
            # quantise to uint8 exactly as we would ship it
            r = torch.clamp(mean + a * hp, 0, 1)
            r = torch.round(r * 255.0) / 255.0
            with torch.no_grad():
                acc[a][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[a][1] += float(repo_ssim(r, g))
                acc[a][2] += float(lp(r * 2 - 1, g * 2 - 1).item())
        print(f"[{n+1}/{len(stems)}] {st} {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    res = {}
    for a in alphas:
        P, S, L = [v / N for v in acc[a]]
        res[a] = dict(psnr=P, ssim=S, lpips=L,
                      score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1)))
    json.dump({"n": N, "members": mem, "sigma": args.sigma, "res": res}, open(args.out, "w"), indent=1)
    base = res[alphas[0]]["score"]
    print(f"\n=== HF REINJECTION, k={len(mem)} mean, sigma={args.sigma}, n={N} ===")
    for a in alphas:
        v = res[a]
        print(f"  alpha={a:<5} PSNR {v['psnr']:.4f} SSIM {v['ssim']:.4f} LPIPS {v['lpips']:.4f} "
              f"SCORE {v['score']:.4f}  delta {v['score']-base:+.4f}")


if __name__ == "__main__":
    main()
