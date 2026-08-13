"""Harness lib for the TRAJECTORY / PHOTOMETRIC lever (agent tj).

Uses the prebuilt uint8 cache (renders_u8.npy 22x60xHxWx3, gt_u8.npy 60xHxWx3).
Score = 100*[0.4*(1-LPIPS_vgg) + 0.3*SSIM + 0.3*PSNR/50]  with the PROJECT scorer.
"""
import os, sys
import numpy as np
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
DEV = "cuda"
_vgg = None
_ssim = None


def init(dev=DEV):
    global _vgg, _ssim
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    _ssim = repo_ssim
    _vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()


def names():
    return open(f"{TMP}/names.txt").read().split("\n")


def renders():
    return np.load(f"{TMP}/renders_u8.npy", mmap_mode="r")


def gt():
    return np.load(f"{TMP}/gt_u8.npy", mmap_mode="r")


def score(pred, gtu8, tag="", dev=DEV, quiet=False, per_view=False):
    """pred: (N,H,W,3) float in [0,1] (or uint8).  gtu8: (N,H,W,3) uint8."""
    P = S = L = 0.0
    n = len(pred)
    pv = []
    with torch.no_grad():
        for i in range(n):
            a = pred[i]
            if a.dtype == np.uint8:
                a = a.astype(np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(np.clip(a, 0, 1), dtype=np.float32)
                                 ).permute(2, 0, 1).unsqueeze(0).to(dev)
            g = torch.from_numpy(np.ascontiguousarray(gtu8[i], dtype=np.float32) / 255.0
                                 ).permute(2, 0, 1).unsqueeze(0).to(dev)
            mse = ((r - g) ** 2).mean().item()
            p = 10 * np.log10(1.0 / max(mse, 1e-12))
            s = float(_ssim(r, g))
            l = float(_vgg(r * 2 - 1, g * 2 - 1).item())
            P += p; S += s; L += l
            if per_view:
                pv.append((p, s, l))
    P, S, L = P / n, S / n, L / n
    sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    if not quiet:
        print(f"{tag:34s} PSNR {P:7.4f}  SSIM {S:.4f}  LPIPS {L:.4f}  SCORE {sc:8.4f}",
              flush=True)
    d = dict(tag=tag, psnr=P, ssim=S, lpips=L, score=sc)
    if per_view:
        d["pv"] = pv
    return d


def u8(x):
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
