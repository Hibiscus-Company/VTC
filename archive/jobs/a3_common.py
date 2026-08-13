"""Shared CPU metric definitions -- copied verbatim from scripts/eval_score.py.
NEVER touches the GPU."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
import sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
torch.set_num_threads(6)
from utils.loss_utils import ssim as repo_ssim  # noqa: E402
import lpips as lpips_pkg  # noqa: E402

DEV = "cpu"


def load(p):
    return torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)


def score_from(P, S, L, psnr_max=50.0):
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / psnr_max, 1.0))


def pair_list(render_dir, gt_dir):
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)}
    renders = [f for f in sorted(os.listdir(render_dir))
               if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")]
    assert {os.path.splitext(f)[0] for f in renders} == set(gt_by), (
        f"stem sets differ: {len(renders)} vs {len(gt_by)}")
    return [(s, os.path.join(render_dir, f), os.path.join(gt_dir, gt_by[s]))
            for f in renders for s in [os.path.splitext(f)[0]]]


def gray(x):  # x: 1,3,H,W torch -> HxW numpy float32 in [0,1] (Rec.601)
    a = x[0].numpy()
    return (0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]).astype(np.float32)


def lapvar(g):
    import cv2
    return float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var())
