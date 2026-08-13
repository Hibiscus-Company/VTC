#!/usr/bin/env python
"""CPU-only clone of scripts/eval_score.py (metric definitions copied verbatim)."""
import argparse, os, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
torch.set_num_threads(int(os.environ.get("NT","3")))
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
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir)}
    P = S = L = 0.0
    n = 0
    renders = [f for f in sorted(os.listdir(args.render_dir))
               if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")]
    if not args.limit:
        assert {os.path.splitext(f)[0] for f in renders} == set(gt_by), "render/GT stem sets differ"
    else:
        renders = renders[:args.limit]
    with torch.no_grad():
        for f in renders:
            s = os.path.splitext(f)[0]
            assert s in gt_by, f"render {f} has no GT match"
            r = load(os.path.join(args.render_dir, f)).to(dev)
            g = load(os.path.join(args.gt_dir, gt_by[s])).to(dev)
            assert r.shape == g.shape, f"shape mismatch {f}"
            mse = ((r - g) ** 2).mean().item()
            P += 10 * np.log10(1.0 / max(mse, 1e-12))
            S += float(repo_ssim(r, g))
            L += float(vgg(r * 2 - 1, g * 2 - 1).item())
            n += 1
    P, S, L = P / n, S / n, L / n
    score = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / args.psnr_max, 1.0))
    print(f"EVAL {args.tag:16s} n={n:3d}  PSNR {P:7.4f}  SSIM {S:.4f}  "
          f"LPIPSvgg {L:.4f}  SCORE {score:.4f}", flush=True)


if __name__ == "__main__":
    main()
