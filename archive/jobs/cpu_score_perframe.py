#!/usr/bin/env python
"""CPU-only PER-FRAME scorer.  Metric definitions copied verbatim from scripts/eval_score.py
(repo SSIM from utils.loss_utils, lpips.LPIPS(net='vgg'), PSNR = 10*log10(1/mse)).
Writes a CSV, one row per frame.  NEVER touches the GPU."""
import argparse, os, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
torch.set_num_threads(6)
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(a.gt_dir)}
    renders = sorted(f for f in os.listdir(a.render_dir)
                     if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg"))
    assert {os.path.splitext(f)[0] for f in renders} == set(gt_by)
    rows = []
    with torch.no_grad():
        for f in renders:
            s = os.path.splitext(f)[0]
            r = load(os.path.join(a.render_dir, f))
            g = load(os.path.join(a.gt_dir, gt_by[s]))
            assert r.shape == g.shape, f"shape {f}"
            mse = ((r - g) ** 2).mean().item()
            P = 10 * np.log10(1.0 / max(mse, 1e-12))
            S = float(repo_ssim(r, g))
            L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
            rows.append((s, P, S, L, sc))
            print(f"{s} PSNR {P:.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {sc:.4f}", flush=True)
    with open(a.out, "w") as fh:
        fh.write("name,frame,psnr,ssim,lpips,score\n")
        for s, P, S, L, sc in rows:
            fh.write(f"{s},{int(s.split('_')[-1])},{P:.6f},{S:.6f},{L:.6f},{sc:.6f}\n")
    A = np.array([[r[1], r[2], r[3], r[4]] for r in rows]).mean(0)
    print(f"MEAN n={len(rows)} PSNR {A[0]:.4f} SSIM {A[1]:.4f} LPIPS {A[2]:.4f} SCORE_of_means "
          f"{100*(0.4*(1-A[2])+0.3*A[1]+0.3*A[0]/50):.4f}")


if __name__ == "__main__":
    main()
