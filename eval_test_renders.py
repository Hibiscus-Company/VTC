#
# Evaluate rendered test images against ground truth with the competition metrics:
#   Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*clamp(PSNR/psnr_max, 0, 1)
#
# Usage:
#   python eval_test_renders.py --renders renders/<scene> --gt <scene>/test/images [--psnr_max 50]
#

import os
import json
import argparse
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--psnr_max", type=float, default=50.0)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    gt_by_stem = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt)}
    render_files = sorted(os.listdir(args.renders))

    per_image = {}
    ssims, psnrs, lpipss = [], [], []
    for fname in tqdm(render_files, desc="Evaluating"):
        stem = os.path.splitext(fname)[0]
        if stem not in gt_by_stem:
            print(f"WARNING: no GT for {fname}, skipping")
            continue
        render = tf.to_tensor(Image.open(os.path.join(args.renders, fname))).unsqueeze(0)[:, :3].cuda()
        gt = tf.to_tensor(Image.open(os.path.join(args.gt, gt_by_stem[stem]))).unsqueeze(0)[:, :3].cuda()
        assert render.shape == gt.shape, f"size mismatch {fname}: {render.shape} vs {gt.shape}"
        with torch.no_grad():
            s = ssim(render, gt).item()
            p = psnr(render, gt).mean().item()
            l = lpips(render, gt, net_type="vgg").item()
        ssims.append(s); psnrs.append(p); lpipss.append(l)
        per_image[fname] = {"ssim": s, "psnr": p, "lpips": l}

    n = len(ssims)
    if n == 0:
        print("No image pairs evaluated!")
        return
    mssim = sum(ssims) / n
    mpsnr = sum(psnrs) / n
    mlpips = sum(lpipss) / n
    psnr_norm = min(max(mpsnr / args.psnr_max, 0.0), 1.0)
    score = 0.4 * (1.0 - mlpips) + 0.3 * mssim + 0.3 * psnr_norm

    print(f"\n  images : {n}")
    print(f"  PSNR   : {mpsnr:.4f}")
    print(f"  SSIM   : {mssim:.4f}")
    print(f"  LPIPS  : {mlpips:.4f}")
    print(f"  Score  : {score:.4f}  (psnr_max={args.psnr_max})")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"mean": {"psnr": mpsnr, "ssim": mssim, "lpips": mlpips, "score": score},
                       "per_image": per_image}, f, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
