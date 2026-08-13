#!/usr/bin/env python
"""Per-scene TRAIN-view PSNR + SSIM of the current 60k/8M members.

The consult's strongest actionable hypothesis: we fix the lens at RENDER time but the
model is TRAINED with radial_coeffs = COLMAP k1, which on HNI0131/HNI0265 (k1=-0.115) is
the single-k1 lens that "cannot be represented". So during training the rasterizer projects
gaussians through a WRONG lens, the optimizer contorts them to compensate but can't fully,
and TRAIN fidelity is capped -> TEST capped.

Testable signature: the two k1=-0.115 scenes should have LOWER train PSNR than the +k scenes.
The R9 field-fit already rendered train views on the correct native/warp path, so this is
just scoring files that already exist. TRAIN photos are legal ground truth (no test GT).
"""
import os, sys, glob
import numpy as np
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None
try:
    import torch
    from fused_ssim import fused_ssim
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False

PRI = os.path.expanduser("~/data/phase1/private_set1")
SCENES = ["HCM0249", "HCM0254", "HCM0276", "HCM1439",
          "HNI0131", "HNI0265", "HNI0366", "HNI0437"]
K1 = {"HCM0249": +0.0089, "HCM0254": +0.0098, "HCM0276": +0.0081, "HCM1439": +0.0076,
      "HNI0131": -0.1148, "HNI0265": -0.1147, "HNI0366": +0.0120, "HNI0437": +0.0138}


def psnr_ssim(rd, gt_dir):
    st = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)}
    ps, ss = [], []
    for f in sorted(glob.glob(os.path.join(rd, "*.png"))):
        s = os.path.splitext(os.path.basename(f))[0]
        if s not in st:
            continue
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255
        b = np.asarray(Image.open(os.path.join(gt_dir, st[s])).convert("RGB"),
                       dtype=np.float32) / 255
        if a.shape != b.shape:
            continue
        ps.append(10 * np.log10(1 / max(float(((a - b) ** 2).mean()), 1e-12)))
        if HAVE_SSIM:
            ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).cuda()
            tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).cuda()
            ss.append(float(fused_ssim(ta, tb)))
    return (float(np.mean(ps)) if ps else float("nan"),
            float(np.mean(ss)) if ss else float("nan"), len(ps))


print(f"{'scene':9s} {'k1':>8s} {'path':6s} {'n':>4s} {'TRAIN psnr':>11s} {'ssim':>7s}")
neg, pos = [], []
for s in SCENES:
    for utr in ("native", "warp"):
        rd = f"/mnt/d/avv/output/{s}_gsplatB9ut60k/train_png_{utr}"
        if os.path.isdir(rd):
            break
    if not os.path.isdir(rd):
        print(f"{s:9s}  (no train renders)")
        continue
    p, ss, n = psnr_ssim(rd, os.path.join(PRI, s, "train", "images"))
    tag = "warp" if K1[s] < 0 else "native"
    print(f"{s:9s} {K1[s]:+8.4f} {tag:6s} {n:4d} {p:11.3f} {ss:7.4f}")
    (neg if K1[s] < 0 else pos).append(p)

print()
print(f"  +k scenes (6): mean train PSNR {np.mean(pos):.3f}")
print(f"  -k scenes (2): mean train PSNR {np.mean(neg):.3f}   "
      f"(HNI0131/HNI0265, k1=-0.115)")
print(f"  delta: {np.mean(neg)-np.mean(pos):+.3f} dB")
print("\n  => if the -k scenes are much LOWER, train-time lens misfit is real and")
print("     per-scene, and undistort-in-training is a lever. If comparable, the 27dB")
print("     train cap is uniform -> not lens-specific, look elsewhere (capacity/content).")
