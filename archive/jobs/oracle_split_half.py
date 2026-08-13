#!/usr/bin/env python
"""IS THE PER-VIEW FIELD ORACLE (+0.66..+0.80) REAL SIGNAL, OR IS IT FITTING NOISE?

This gates a multi-hour 3D-lift build, so it must be answered first and cheaply.

THE PROBLEM WITH THE ORACLE NUMBER: it fits a displacement field for view i using DIS flow between
render_i and photo_i, then applies that field to view i. It is fitted ON the thing it is scored on.
Some of its +0.7 is therefore guaranteed to be fitting view i's own DIS-flow noise, which is
unreachable by ANY legal predictor -- no 3D lift, no interpolation, nothing.

THE TEST -- spatial hold-out within a view. Fit the per-view field using ONLY the TOP half of the
image, then apply it to the BOTTOM half and score the bottom half only. Compare against the global
(scene-wide) field on that same bottom half.
  per-view BEATS global on held-out image rows  => there is real, spatially-coherent per-view signal
                                                   => a 3D lift can plausibly reach it. BUILD IT.
  per-view LOSES to global on held-out rows     => the oracle is noise-fitting, the +0.7 is a mirage,
                                                   and the whole 3D-lift branch is dead for free.
Rule 10: train photos only, never test GT.
"""
import os, sys
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track")
from fit_field import fit_field, apply_field
Image.MAX_IMAGE_PIXELS = None


def flow_of(render, gt, ds=8, clip=6.0):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    rg = (cv2.cvtColor(render, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
    gg = (cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
    fl = np.clip(dis.calc(gg, rg, None), -clip, clip)
    H, W, _ = render.shape
    return cv2.resize(fl, (W // ds, H // ds), interpolation=cv2.INTER_AREA)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[{args.tag}] pooling global field ...", flush=True)
    glob_mean, stack, stems = fit_field(args.render_dir, args.gt_dir, ds=args.ds,
                                        verbose=False, return_stack=True)
    glob_med = np.median(stack, axis=0)
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir)}
    rd_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.render_dir)}
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(stems))[:args.n]

    def sc_bottom(img, g):
        """score ONLY the bottom half -- the held-out rows"""
        H = img.shape[0]; h0 = H // 2
        r = torch.from_numpy(np.clip(img[h0:], 0, 1)).permute(2, 0, 1).unsqueeze(0).to(dev)
        t = torch.from_numpy(g[h0:]).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            P = 10 * np.log10(1.0 / max(((r - t) ** 2).mean().item(), 1e-12))
            S = float(repo_ssim(r, t))
        return P, S

    acc = {k: [0.0, 0.0] for k in ("none", "global_med", "perview_tophalf", "perview_offset", "perview_oracle")}
    for c, i in enumerate(idx):
        stem = stems[i]
        img = np.asarray(Image.open(os.path.join(args.render_dir, rd_by[stem])).convert("RGB"),
                         dtype=np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(args.gt_dir, gt_by[stem])).convert("RGB"),
                       dtype=np.float32) / 255.0
        fh = stack.shape[1]                      # field rows
        own = stack[i]                           # this view's own flow == the ORACLE field
        # per-view field fitted on the TOP half only, then extended to the whole frame:
        # top rows keep the per-view estimate, bottom rows get the top half's mean offset
        # (i.e. we must EXTRAPOLATE into the held-out region -- that is the honest test)
        top = own[: fh // 2]
        pv_top = np.repeat(top.mean(axis=0, keepdims=True), fh, axis=0)
        # CORRECTED variant: keep the global field's spatial structure and add only the per-view
        # OFFSET estimated from the top half. This isolates "is the per-view DEVIATION predictable"
        # from "does discarding the global field's spatial structure hurt" (it does, a lot).
        dev_top = (own - glob_med)[: fh // 2].mean(axis=0, keepdims=True)
        pv_offset = glob_med + np.repeat(dev_top, fh, axis=0)
        variants = {"none": None, "global_med": glob_med,
                    "perview_tophalf": pv_top, "perview_offset": pv_offset,
                    "perview_oracle": own}
        for k, f in variants.items():
            im = img if f is None else np.clip(apply_field(img, f), 0, 1)
            P, S = sc_bottom(im, g)
            acc[k][0] += P; acc[k][1] += S
        if c % 10 == 0:
            print(f"  {c}/{len(idx)}", flush=True)

    n = len(idx)
    print(f"\n[{args.tag}] scored on HELD-OUT BOTTOM HALF only, n={n}")
    print(f"{'variant':>18} {'PSNR':>8} {'SSIM':>7} {'partial':>9}  {'vs global':>10}")
    base = None
    for k in ("none", "global_med", "perview_tophalf", "perview_offset", "perview_oracle"):
        P, S = (x / n for x in acc[k])
        part = 0.6 * P + 30 * S
        if k == "global_med":
            base = part
        print(f"{k:>18} {P:8.4f} {S:7.4f} {part:9.4f}  {part - (base if base else part):+10.4f}")
    P, S = (x / n for x in acc["perview_offset"]); pv = 0.6 * P + 30 * S
    P, S = (x / n for x in acc["perview_oracle"]); orc = 0.6 * P + 30 * S
    print(f"\n[{args.tag}] VERDICT")
    print(f"  oracle (fitted ON the scored rows, upper bound incl. noise-fitting): {orc-base:+.4f}")
    print(f"  per-view fitted on HELD-OUT rows (what a real predictor could reach): {pv-base:+.4f}")
    print("  -> if the held-out number is <= 0, the oracle is noise-fitting and the 3D-lift is DEAD.")


if __name__ == "__main__":
    main()
