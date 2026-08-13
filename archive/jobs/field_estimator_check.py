#!/usr/bin/env python
"""INDEPENDENT verification that a median-pooled lens field beats the shipped mean-pooled one.

The agents measured this with a PARTIAL score (0.6*PSNR + 30*SSIM), omitting LPIPS. The field's
LB signature was PSNR +0.556 / SSIM +1.31 / LPIPS -0.0009, so LPIPS should be ~flat -- but "should
be" is exactly the reasoning that cost us r23, so measure it with the project's real scorer.

PROTOCOL: leave-one-view-out. For each held-out view i, the field is pooled over all views EXCEPT
i, then applied to render i and scored against photo i. So no view ever contributes to the field
that corrects it -- this is genuine held-out validation, not an in-sample fit.
Rule 10: train photos only (the same train renders/photos fit_field already uses).
"""
import argparse, os, sys
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track")
from fit_field import fit_field, apply_field
Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--n_eval", type=int, default=30, help="held-out views to score")
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    print(f"[{args.tag}] computing flow stack ...", flush=True)
    _, stack, stems = fit_field(args.render_dir, args.gt_dir, ds=args.ds,
                               verbose=False, return_stack=True)
    N = len(stems)
    print(f"[{args.tag}] {N} train pairs, stack {stack.shape}", flush=True)

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir)}
    rd_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.render_dir)}
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)[:args.n_eval]

    def score(img, g):
        r = torch.from_numpy(np.clip(img, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(dev)
        t = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            p = 10 * np.log10(1.0 / max(((r - t) ** 2).mean().item(), 1e-12))
            s = float(repo_ssim(r, t))
            l = float(vgg(r * 2 - 1, t * 2 - 1).item())
        return p, s, l

    res = {k: [0.0, 0.0, 0.0] for k in ("none", "mean", "median")}
    for c, i in enumerate(idx):
        stem = stems[i]
        img = np.asarray(Image.open(os.path.join(args.render_dir, rd_by[stem])).convert("RGB"),
                         dtype=np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(args.gt_dir, gt_by[stem])).convert("RGB"),
                       dtype=np.float32) / 255.0
        others = np.delete(stack, i, axis=0)          # <-- the held-out part
        fields = {"none": None,
                  "mean": others.mean(axis=0),
                  "median": np.median(others, axis=0)}
        for k, f in fields.items():
            im = img if f is None else np.clip(apply_field(img, f), 0, 1)
            p, s, l = score(im, g)
            res[k][0] += p; res[k][1] += s; res[k][2] += l
        if c % 10 == 0:
            print(f"  {c}/{len(idx)}", flush=True)

    n = len(idx)
    print(f"\n[{args.tag}] leave-one-view-out over n={n} held-out train views")
    print(f"{'estimator':>10} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8}   {'vs mean':>9}")
    base = None
    out = {}
    for k in ("none", "mean", "median"):
        P, S, L = (x / n for x in res[k])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        out[k] = sc
        if k == "mean":
            base = sc
    for k in ("none", "mean", "median"):
        P, S, L = (x / n for x in res[k])
        sc = out[k]
        print(f"{k:>10} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f}   {sc - base:+9.4f}")
    d = out["median"] - out["mean"]
    print(f"\n[{args.tag}] MEDIAN vs SHIPPED MEAN: {d:+.4f}   "
          f"({'SHIP' if d > 0.02 else 'not worth it'})")


if __name__ == "__main__":
    main()
