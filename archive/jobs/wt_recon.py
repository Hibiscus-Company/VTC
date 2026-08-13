#!/usr/bin/env python
"""RECON for family-weight re-derivation.

One pass over the 60 REAL test poses of HCM0181. For every render variant on disk:
  * solo PSNR / LPIPS against real test GT (raw PNG -- selection only, not a claim)
  * per-stem residual Gram  G_s[i,j] = <x_i - gt, x_j - gt> / npix   (RGB-summed, per pixel)

The Gram is the whole point: any linear weighting w (sum w = 1) has
    MSE(w) = w^T (mean_s G_s) w / 3
so the LS-optimal weights, the family-blend MSE curve, and honest 2-fold CV are all closed form
with ZERO extra GPU work. Only the handful of weightings that survive get the full shipped chain.
"""
import os, sys, time, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None

TAG = "HCM0181"
GTD = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
ROOT = "/mnt/d/avv/output"


def variants():
    out = {}
    for d in sorted(os.listdir(ROOT)):
        if not d.startswith(f"{TAG}_"):
            continue
        for sub in ("test_poses_renders_png", "tp_png"):
            p = os.path.join(ROOT, d, sub)
            if os.path.isdir(p) and len(os.listdir(p)) == 60:
                out[d[len(TAG) + 1:]] = p
                break
    return out


def main():
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    V = variants()
    names = sorted(V)
    gt_by = {os.path.splitext(f)[0]: os.path.join(GTD, f) for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(V[n], s + ".png"))
                                           for n in names))
    print(f"variants={len(names)} stems={len(stems)}", flush=True)
    print(names, flush=True)

    K = len(names)
    G = np.zeros((len(stems), K, K), dtype=np.float64)
    psnr = np.zeros((len(stems), K))
    lp = np.zeros((len(stems), K))
    t0 = time.time()
    for si, s in enumerate(stems):
        g = torch.from_numpy(np.asarray(Image.open(gt_by[s]).convert("RGB"), dtype=np.float32)
                             / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        R = []
        for ki, n in enumerate(names):
            x = torch.from_numpy(np.asarray(Image.open(os.path.join(V[n], s + ".png")).convert("RGB"),
                                            dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                psnr[si, ki] = 10 * np.log10(1.0 / max(((x - g) ** 2).mean().item(), 1e-12))
                lp[si, ki] = float(vgg(x * 2 - 1, g * 2 - 1).item())
            R.append((x - g).reshape(-1))
        Rm = torch.stack(R)                      # [K, 3HW]
        G[si] = (Rm @ Rm.T).double().cpu().numpy() / Rm.shape[1]
        del R, Rm
        if si % 10 == 0:
            print(f"  {si}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    np.savez("/home/bkai/.claude/jobs/1c9cf7e9/tmp/wt_gram.npz",
             G=G, psnr=psnr, lpips=lp, names=np.array(names), stems=np.array(stems))
    order = np.argsort(-psnr.mean(0))
    print(f"\n{'variant':>18} {'PSNR':>8} {'LPIPS':>8} {'rmse':>9}")
    for i in order:
        print(f"{names[i]:>18} {psnr[:,i].mean():8.4f} {lp[:,i].mean():8.4f} "
              f"{np.sqrt(G[:,i,i].mean()/3):9.5f}")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
