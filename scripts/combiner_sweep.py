#!/usr/bin/env python
"""Strategy-audit rank-3: do combiners beyond pixel-MEAN add anything? Tests per-pixel MEDIAN
and AGREEMENT-WEIGHTED mean (weight inversely prop. to distance from the per-pixel median)
against the plain mean of the same members, scored on eval GT. Members = same-recipe config
variants (the 'within-band jitter' class, eligible spares per exp15) -- a proxy for the
production seed-members whose eval renders we don't have."""
import argparse, os
import numpy as np
from PIL import Image
import torch

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()
    dev = "cuda"
    import sys
    sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    names = sorted(f for f in os.listdir(args.dirs[0]) if f.lower().endswith((".png", ".jpg")))
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir)}
    scores = {m: [0.0, 0.0, 0.0, 0] for m in ("mean", "median", "agree")}
    with torch.no_grad():
        for f in names:
            s = os.path.splitext(f)[0]
            assert s in gt_by, f"no GT for {f}"
            imgs = []
            for d in args.dirs:
                p = os.path.join(d, f)
                if not os.path.isfile(p):
                    p = os.path.join(d, s + ".png")
                imgs.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32))
            stack = np.stack(imgs)  # [N,H,W,3]
            outs = {"mean": stack.mean(0), "median": np.median(stack, axis=0)}
            med = outs["median"]
            w = 1.0 / (np.abs(stack - med[None]).mean(axis=3, keepdims=True) + 3.0)  # [N,H,W,1]
            outs["agree"] = (stack * w).sum(0) / w.sum(0)
            g = torch.from_numpy(np.asarray(
                Image.open(os.path.join(args.gt_dir, gt_by[s])).convert("RGB"),
                dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            for m, arr in outs.items():
                r = torch.from_numpy(arr / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
                mse = ((r - g) ** 2).mean().item()
                scores[m][0] += 10 * np.log10(1.0 / max(mse, 1e-12))
                scores[m][1] += float(repo_ssim(r, g))
                scores[m][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
                scores[m][3] += 1
    for m, (P, S, L, n) in scores.items():
        P, S, L = P / n, S / n, L / n
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1.0))
        print(f"COMBINE {args.tag:12s} {m:7s} n={n:3d} PSNR {P:7.4f} SSIM {S:.4f} "
              f"LPIPS {L:.4f} SCORE {sc:.4f}")


if __name__ == "__main__":
    main()
