#!/usr/bin/env python
"""P1: TEXTURE INJECTION -- deliberately replace the structured high-frequency content that
pixel-mean ensembling destroys.

RATIONALE (the one production-side gradient we own): r23 is an LB-GRADED measurement, at production
depth, that our shipped renders are TOO SMOOTH. Removing JPEG's structured high-frequency artifacts
cost us 0.026 ON THE LEADERBOARD. JPEG was accidentally doing texture injection for us. Do it
deliberately and better.

CONSTRUCTION: out = mean_k + lambda * HP(member_j - mean_k),  HP(x) = x - gaussian_blur(x, sigma)
The low frequencies stay at the ensemble mean (where averaging genuinely helps: it cancels
reconstruction noise). Only the HIGH frequencies get one member's texture put back. Crucially this
texture is IMAGE-CORRELATED and structured -- unlike the white/luma grain injection that measured
catastrophically negative (-3.7 to -22).

THE SIGNATURE TEST (this is the point of the experiment, not the raw number):
  cleanup-class interventions DECAY with ensemble depth k  (encode fix: +0.259/+0.194/+0.115)
  texture-injection should GROW with k                     (deeper mean = smoother = more missing)
If the gain grows with k, the proxy UNDER-states the production value and proxy-optimal lambda is a
safe LOWER bound. If it decays, it is just another cleanup trick and we drop it.
"""
import argparse, os, sys, itertools
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None


def gauss_blur(t, sigma):
    """t: [1,3,H,W] -> gaussian blur, separable, reflect-padded"""
    r = max(1, int(round(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=t.device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    t = F.pad(t, (r, r, 0, 0), mode="reflect")
    t = F.conv2d(t, k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
    t = F.pad(t, (0, 0, r, r), mode="reflect")
    return F.conv2d(t, k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="tex")
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = args.device
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    gt_by = {os.path.splitext(f)[0]: os.path.join(args.gt_dir, f) for f in os.listdir(args.gt_dir)}
    stems = sorted(gt_by)
    print(f"{args.tag}: {len(stems)} poses, {len(args.members)} members available")

    def load(d, s):
        return torch.from_numpy(
            np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                       dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    print(f"{'k':>2} {'sig':>4} {'lam':>5} {'SCORE':>9} {'LPIPS':>8} {'SSIM':>7} {'PSNR':>8}  {'vs lam=0':>9}")
    results = {}
    for k in args.ks:
        dirs = args.members[:k]
        for sigma in args.sigmas:
            base_sc = None
            for lam in args.lambdas:
                P = S = L = 0.0
                for s in stems:
                    ms = [load(d, s).to(dev) for d in dirs]
                    mean = torch.stack(ms).mean(0)
                    if lam > 0:
                        dev_j = ms[0] - mean                  # deviation of one member
                        hp = dev_j - gauss_blur(dev_j, sigma)  # keep only its HIGH frequencies
                        img = (mean + lam * hp).clamp(0, 1)
                    else:
                        img = mean.clamp(0, 1)
                    g = load(os.path.dirname(gt_by[s]) or ".", s) if False else torch.from_numpy(
                        np.asarray(Image.open(gt_by[s]).convert("RGB"), dtype=np.float32) / 255.0
                    ).permute(2, 0, 1).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        P += 10 * np.log10(1.0 / max(((img - g) ** 2).mean().item(), 1e-12))
                        S += float(repo_ssim(img, g))
                        L += float(vgg(img * 2 - 1, g * 2 - 1).item())
                n = len(stems)
                P, S, L = P / n, S / n, L / n
                sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
                if lam == 0.0:
                    base_sc = sc
                d0 = sc - base_sc
                results[(k, sigma, lam)] = sc
                print(f"{k:>2} {sigma:>4.1f} {lam:>5.2f} {sc:9.4f} {L:8.4f} {S:7.4f} {P:8.4f}  {d0:+9.4f}")
            print()

    print("=== SIGNATURE (best lambda gain at each depth) ===")
    for sigma in args.sigmas:
        for k in args.ks:
            b = max(args.lambdas, key=lambda l: results[(k, sigma, l)])
            g = results[(k, sigma, b)] - results[(k, sigma, 0.0)]
            print(f"  sigma {sigma}: k={k}  best lambda {b:.2f}  gain {g:+.4f}")
    print("GROWS with k -> texture-injection class, proxy UNDER-states it, proxy-lambda is a safe")
    print("lower bound. DECAYS with k -> just another cleanup trick, drop it.")


if __name__ == "__main__":
    main()
