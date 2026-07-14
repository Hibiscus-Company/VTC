#
# Local competition scorer: score a submission (folder of scene folders, or a .zip)
# against public-set test GT with the exact competition formula:
#   Score = 0.4*(1 - LPIPS) + 0.3*SSIM + 0.3*clamp(PSNR/psnr_max, 0, 1)
# Score = mean over scenes (each scene = mean over its images).
#
# Computes LPIPS with BOTH backbones (alex, vgg) from the `lpips` pip package
# so the backbone can be calibrated against the known leaderboard score.
#
# Usage:
#   python score_submission.py --sub ~/subs_eval/round1_g2 \
#       --gt_root ~/data/phase1/public_set [--psnr_max 50] [--out_json eval.json]
#
import os
import sys
import json
import zipfile
import tempfile
import argparse

import torch
import torchvision.transforms.functional as tf
from PIL import Image

from utils.loss_utils import ssim as fastgs_ssim
from utils.image_utils import psnr as fastgs_psnr


def score_scene(render_dir, gt_dir, lpips_nets, device):
    gt_by_stem = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)}
    files = sorted(os.listdir(render_dir))
    acc = {"psnr": 0.0, "ssim": 0.0, "lpips_alex": 0.0, "lpips_vgg": 0.0}
    n = 0
    missing = []
    for fname in files:
        stem = os.path.splitext(fname)[0]
        if stem not in gt_by_stem:
            missing.append(fname)
            continue
        render = tf.to_tensor(Image.open(os.path.join(render_dir, fname))).unsqueeze(0)[:, :3].to(device)
        gt = tf.to_tensor(Image.open(os.path.join(gt_dir, gt_by_stem[stem]))).unsqueeze(0)[:, :3].to(device)
        assert render.shape == gt.shape, f"size mismatch {fname}: {render.shape} vs {gt.shape}"
        with torch.no_grad():
            acc["psnr"] += fastgs_psnr(render, gt).mean().item()
            acc["ssim"] += fastgs_ssim(render, gt).item()
            # lpips package expects [-1, 1]
            r2, g2 = render * 2 - 1, gt * 2 - 1
            acc["lpips_alex"] += lpips_nets["alex"](r2, g2).item()
            acc["lpips_vgg"] += lpips_nets["vgg"](r2, g2).item()
        n += 1
    if missing:
        print(f"  WARNING: {len(missing)} renders had no GT match: {missing[:3]}...")
    return {k: v / n for k, v in acc.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True, help="submission folder or .zip")
    ap.add_argument("--gt_root", default=os.path.expanduser("~/data/phase1/public_set"))
    ap.add_argument("--psnr_max", type=float, default=50.0)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    args = ap.parse_args()

    sub = os.path.expanduser(args.sub)
    tmpdir = None
    if sub.endswith(".zip"):
        tmpdir = tempfile.mkdtemp(prefix="scoresub_")
        gt_scenes = set(os.listdir(args.gt_root))
        with zipfile.ZipFile(sub) as z:
            members = [m for m in z.namelist() if m.split("/")[0] in gt_scenes]
            z.extractall(tmpdir, members)
        sub = tmpdir

    device = torch.device(args.device)
    import lpips as lpips_pkg
    lpips_nets = {
        "alex": lpips_pkg.LPIPS(net="alex", verbose=False).to(device),
        "vgg": lpips_pkg.LPIPS(net="vgg", verbose=False).to(device),
    }

    gt_root = os.path.expanduser(args.gt_root)
    scenes = sorted(s for s in os.listdir(sub)
                    if os.path.isdir(os.path.join(sub, s)) and os.path.isdir(os.path.join(gt_root, s)))
    if not scenes:
        sys.exit(f"No scenes in {sub} match GT scenes in {gt_root}")

    results = {}
    for scene in scenes:
        m, n = score_scene(os.path.join(sub, scene),
                           os.path.join(gt_root, scene, "test", "images"),
                           lpips_nets, device)
        m["n"] = n
        for net in ("alex", "vgg"):
            m[f"score_{net}"] = (0.4 * (1 - m[f"lpips_{net}"]) + 0.3 * m["ssim"]
                                 + 0.3 * min(max(m["psnr"] / args.psnr_max, 0.0), 1.0))
        results[scene] = m
        print(f"{scene:10s} n={n:3d}  PSNR {m['psnr']:7.4f}  SSIM {m['ssim']:.4f}  "
              f"LPIPS(alex) {m['lpips_alex']:.4f} LPIPS(vgg) {m['lpips_vgg']:.4f}  "
              f"Score(alex) {m['score_alex']:.4f}  Score(vgg) {m['score_vgg']:.4f}")

    K = len(scenes)
    mean = {k: sum(r[k] for r in results.values()) / K
            for k in ("psnr", "ssim", "lpips_alex", "lpips_vgg", "score_alex", "score_vgg")}
    psnr_norm = min(max(mean["psnr"] / args.psnr_max, 0.0), 1.0)
    print("-" * 100)
    print(f"{'MEAN':10s} n={K:3d}  PSNR {mean['psnr']:7.4f}  SSIM {mean['ssim']:.4f}  "
          f"LPIPS(alex) {mean['lpips_alex']:.4f} LPIPS(vgg) {mean['lpips_vgg']:.4f}")
    for net in ("alex", "vgg"):
        print(f"  Score({net})  = {100*mean[f'score_{net}']:.4f}   "
              f"[0.4*(1-LPIPS)={0.4*(1-mean[f'lpips_{net}']):.4f}  0.3*SSIM={0.3*mean['ssim']:.4f}  "
              f"0.3*PSNRn={0.3*psnr_norm:.4f}]  (psnr_max={args.psnr_max})")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"scenes": results, "mean": mean, "psnr_max": args.psnr_max}, f, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
