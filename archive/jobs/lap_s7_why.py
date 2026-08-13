#!/usr/bin/env python
"""STEP 7: WHY is the cross-scene number smaller? Measure the level-0 energy budget on BOTH scenes
and sweep lambda on the cross-scene one. The rule can only pay where the ensemble mean actually
sits BELOW the GT in finest-scale energy; if a scene has no deficit there is nothing to restore.
"""
import os, sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lapfuse import fuse_image, lap_pyr, _K
Image.MAX_IMAGE_PIXELS = None

SETS = {
    "HCM0181 PRODUCTION k=4": (
        "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images",
        [f"/mnt/d/avv/output/HCM0181_{t}/test_poses_renders_png" for t in
         ("gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7")]),
    "HCM0181 PRODUCTION k=2": (
        "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images",
        [f"/mnt/d/avv/output/HCM0181_{t}/test_poses_renders_png" for t in
         ("gsplatB9ut", "gsplatB10ut8M")]),
    "HCM0421 evalsplit k=2": (
        "/mnt/d/avv/evalsplit/HCM0421/eval_gt",
        ["/mnt/d/avv/evalgen/HCM0421/eval_png", "/mnt/d/avv/mip3d/HCM0421_mip0.2/eval_png"]),
}
dev = "cuda"
k = _K.to(dev)
print(f"{'set':<26} {'E_mem':>8} {'E_mean':>8} {'shrink':>7} {'E_gt':>8} {'gap':>7} {'rho':>6}")
for nm, (GT, MEM) in SETS.items():
    gtf = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
    stems = sorted(s for s in gtf if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    a = np.zeros(4)
    for s in stems:
        st = torch.stack([torch.from_numpy(np.asarray(
            Image.open(os.path.join(d, s + ".png")).convert("RGB"), dtype=np.float32) / 255.
        ).permute(2, 0, 1) for d in MEM]).to(dev)
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(GT, gtf[s])).convert("RGB"),
                                        dtype=np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(dev)
        L = lap_pyr(st, 1, k)[0][0]
        Lg = lap_pyr(g, 1, k)[0][0]
        f = L.reshape(L.shape[0], -1); f = f / (f.pow(2).mean(1, keepdim=True).sqrt() + 1e-12)
        C = (f @ f.T) / f.shape[1]; kk = L.shape[0]
        a += [L.pow(2).mean(dim=(1, 2, 3)).sqrt().mean().item(),
              L.mean(0).pow(2).mean().sqrt().item(), Lg.pow(2).mean().sqrt().item(),
              (C.sum() - C.diag().sum()).item() / (kk * (kk - 1))]
    a /= len(stems)
    print(f"{nm:<26} {a[0]:8.5f} {a[1]:8.5f} {a[1]/a[0]:7.4f} {a[2]:8.5f} {a[1]/a[2]:7.4f} {a[3]:6.3f}")

# ---- lambda sweep on the cross-scene set
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
GT, MEM = SETS["HCM0421 evalsplit k=2"]
gtf = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
stems = sorted(s for s in gtf if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
print(f"\nHCM0421 lambda sweep ({len(stems)} views)")
print(f"{'lam':>5} {'SCORE':>9} {'delta':>8} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8}")
base = None
for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    R = []
    for s in stems:
        st = torch.stack([torch.from_numpy(np.asarray(
            Image.open(os.path.join(d, s + ".png")).convert("RGB"), dtype=np.float32) / 255.
        ).permute(2, 0, 1) for d in MEM]).to(dev)
        img = (st.mean(0, keepdim=True) if lam == 0 else
               fuse_image(st, dict(nalt=1, rule="energy", kw=dict(lam=lam, win=5)), 5, k))
        a = (img.clamp(0, 1) * 255).round().to(torch.uint8).float() / 255.
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(GT, gtf[s])).convert("RGB"),
                                        dtype=np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            R.append((10 * np.log10(1. / max(((a - g) ** 2).mean().item(), 1e-12)),
                      float(repo_ssim(a, g)), float(vgg(a * 2 - 1, g * 2 - 1).item())))
    P, S, L = np.array(R).mean(0)
    sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1.))
    if base is None:
        base = sc
    print(f"{lam:>5.1f} {sc:9.4f} {sc-base:+8.4f} {P:8.4f} {S:7.4f} {L:8.4f}", flush=True)
