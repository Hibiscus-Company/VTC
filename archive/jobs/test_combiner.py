#!/usr/bin/env python
"""B3 test: does a detail-preserving combine beat naive pixel-mean on real train GT?
LF = gaussian-blur of the naive mean (denoised, shared exposure/color already averaged).
HF = per-pixel, take the member's HF residual with the LARGEST local luma magnitude
     (winner-take-all in a small neighborhood, not raw per-pixel, to avoid speckle).
final = LF + winner_HF, clipped.
Never touches test data -- compares against TRAIN photos only (Rule 10 clean, same
legality class as fit_field.py).
"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter
import glob, os, sys

SIGMA = 2.0
NBHD = 5  # winner-take-all neighborhood (px) to avoid single-pixel speckle picks

def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)

def combine_mean(members):
    return np.mean(members, axis=0)

def combine_detail(members):
    mean = np.mean(members, axis=0)
    lf = gaussian_filter(mean, sigma=(SIGMA, SIGMA, 0))
    hfs = [m - gaussian_filter(m, sigma=(SIGMA, SIGMA, 0)) for m in members]
    lumas = [np.abs(hf).mean(axis=2) for hf in hfs]  # per-member local HF magnitude (luma)
    lumas_smooth = [maximum_filter(l, size=NBHD) for l in lumas]  # dilate winner regions
    stacked = np.stack(lumas_smooth, axis=0)  # [N,H,W]
    winner = np.argmax(stacked, axis=0)  # [H,W]
    hf_stack = np.stack(hfs, axis=0)  # [N,H,W,3]
    combined_hf = np.take_along_axis(hf_stack, winner[None, :, :, None], axis=0)[0]
    return lf + combined_hf

def score_dir(pred_dir, gt_dir, tag):
    import subprocess
    r = subprocess.run(
        ["python", "scripts/eval_score.py", "--render_dir", pred_dir, "--gt_dir", gt_dir, "--tag", tag],
        cwd="/mnt/c/Users/BKAI/an_plaza2/FastGS", capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)

def run_scene(name, seed_dirs, gt_dir, out_root):
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(seed_dirs[0]))
    mean_dir = os.path.join(out_root, name, "mean")
    det_dir = os.path.join(out_root, name, "detail")
    os.makedirs(mean_dir, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)
    for stem in stems:
        files = [os.path.join(d, stem + ".png") for d in seed_dirs]
        if not all(os.path.exists(f) for f in files):
            continue
        members = [load(f) for f in files]
        m = combine_mean(members)
        d = combine_detail(members)
        Image.fromarray(np.clip(m, 0, 255).astype(np.uint8)).save(os.path.join(mean_dir, stem + ".png"))
        Image.fromarray(np.clip(d, 0, 255).astype(np.uint8)).save(os.path.join(det_dir, stem + ".png"))
    print(f"--- {name}: {len(stems)} views combined ---")
    score_dir(mean_dir, gt_dir, f"{name}_naive-mean")
    score_dir(det_dir, gt_dir, f"{name}_detail-combine")
    # also single-member reference (first seed) for context
    score_dir(seed_dirs[0], gt_dir, f"{name}_single-member")

if __name__ == "__main__":
    OUT = "/mnt/d/avv/b3test"
    run_scene("chair",
               ["/mnt/d/avv/r14/chair_aa42/train_png_sample",
                "/mnt/d/avv/r14/chair_aa7/train_png_sample",
                "/mnt/d/avv/r14/chair_aa13/train_png_sample"],
               "/mnt/d/avv/data/phase1/private_set2/chair/train/images", OUT)
    run_scene("bonsai",
               ["/mnt/d/avv/r14/bonsai_aa42/train_png_sample",
                "/mnt/d/avv/r14/bonsai_aa7/train_png_sample",
                "/mnt/d/avv/r14/bonsai_aa13/train_png_sample"],
               "/mnt/d/avv/data/phase1/private_set2/bonsai/train/images", OUT)
