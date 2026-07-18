#!/usr/bin/env python
"""Score a render dir vs an eval-GT dir with the EXACT competition metric.

Score = 100 * [ 0.4*(1 - LPIPS_vgg) + 0.3*SSIM + 0.3*clamp(PSNR/50,0,1) ]

Matches file by stem (render .png/.jpg vs eval_gt .jpg). Used to SELECT recipes on the
held-out train holes -- eval GT is TRAIN photos, never test GT.
"""
import argparse, os, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
# audit r14 bug 1: score_submission.py uses the repo SSIM (zero-pad conv), fused_ssim differs
# by up to ~0.003 SSIM (~0.09 pts) — eval numbers must be comparable to the calibrated proxy
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg


def load(p):
    return torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--psnr_max", type=float, default=50.0)
    args = ap.parse_args()
    dev = "cuda"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir)}
    P = S = L = 0.0
    n = 0
    with torch.no_grad():
        for f in sorted(os.listdir(args.render_dir)):
            s = os.path.splitext(f)[0]
            if s not in gt_by:
                continue
            r = load(os.path.join(args.render_dir, f)).to(dev)
            g = load(os.path.join(args.gt_dir, gt_by[s])).to(dev)
            if r.shape != g.shape:
                continue
            mse = ((r - g) ** 2).mean().item()
            P += 10 * np.log10(1.0 / max(mse, 1e-12))
            S += float(repo_ssim(r, g))
            L += float(vgg(r * 2 - 1, g * 2 - 1).item())
            n += 1
    P, S, L = P / n, S / n, L / n
    score = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / args.psnr_max, 1.0))
    print(f"EVAL {args.tag:16s} n={n:3d}  PSNR {P:7.4f}  SSIM {S:.4f}  "
          f"LPIPSvgg {L:.4f}  SCORE {score:.4f}")


if __name__ == "__main__":
    main()
