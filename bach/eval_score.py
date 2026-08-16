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
import torch
# audit r14 bug 1: score_submission.py uses the repo SSIM (zero-pad conv), fused_ssim differs
# by up to ~0.003 SSIM (~0.09 pts) — eval numbers must be comparable to the calibrated proxy
from metrics_utils import ssim as repo_ssim
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
    renders = [f for f in sorted(os.listdir(args.render_dir))
               if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")]
    # audit 26/07: the 23/07 hardening asserted every RENDER has a GT, but not the converse.
    # An arm that renders 57/58 poses and one that renders 58/58 both passed, and were then
    # averaged over DIFFERENT subsets -- the same A/B poisoning, entered from the other side.
    assert {os.path.splitext(f)[0] for f in renders} == set(gt_by), (
        f"render/GT stem sets differ: {len(renders)} renders vs {len(gt_by)} GT "
        f"(missing renders: {sorted(set(gt_by) - {os.path.splitext(f)[0] for f in renders})[:5]})")
    with torch.no_grad():
        for f in renders:
            s = os.path.splitext(f)[0]
            # audit 23/07: silent pair-dropping can poison a paired A/B (arms averaged over
            # different subsets while the printout looks normal) -- hard-error instead.
            assert s in gt_by, f"render {f} has no GT match in {args.gt_dir}"
            r = load(os.path.join(args.render_dir, f)).to(dev)
            g = load(os.path.join(args.gt_dir, gt_by[s])).to(dev)
            assert r.shape == g.shape, f"shape mismatch {f}: render {tuple(r.shape)} vs GT {tuple(g.shape)}"
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
