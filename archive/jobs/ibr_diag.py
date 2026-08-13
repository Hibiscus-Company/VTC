# IBR failure diagnostic: is the pasted photo texture SYSTEMATICALLY shifted
# (convention bug -> one fixable offset) or STOCHASTICALLY misaligned
# (depth/parallax noise -> needs per-pixel flow correction)?
#
# Method: on the base-arm IBR renders vs GT, sweep sub-pixel/pixel shifts
# like the earlier alignment sweep. A clear off-zero peak = systematic bug.
# A flat-worse surface = stochastic misalignment. Also compare per-image:
# IBR-vs-GT PSNR against splat-vs-GT PSNR correlated with coverage.
import os
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import shift as ndshift

IBR = "/mnt/d/avv/ibr/HCM0181_base/renders_png"
SPL = "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png"
GT = os.path.expanduser("~/data/phase1/public_set/HCM0181/test/images")

names = sorted(os.listdir(IBR))[::4][:12]

def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / np.mean((a - b) ** 2))

pairs = []
for n in names:
    ib = np.asarray(Image.open(os.path.join(IBR, n)).convert("RGB"), dtype=np.float32)
    sp = np.asarray(Image.open(os.path.join(SPL, n)).convert("RGB"), dtype=np.float32)
    g = np.asarray(Image.open(os.path.join(GT, os.path.splitext(n)[0] + ".JPG")).convert("RGB"),
                   dtype=np.float32)
    pairs.append((ib, sp, g))

print("per-image PSNR (ibr vs splat):")
for n, (ib, sp, g) in zip(names, pairs):
    print(f"  {n}: ibr {psnr(ib[8:-8,8:-8], g[8:-8,8:-8]):.2f}  splat {psnr(sp[8:-8,8:-8], g[8:-8,8:-8]):.2f}")

print("shift sweep on IBR renders (mean PSNR, crop 8):")
for label, axis in (("y", 0), ("x", 1)):
    for o in (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0):
        vals = []
        for ib, sp, g in pairs:
            sv = [0.0, 0.0, 0.0]
            sv[axis] = o
            ibs = ndshift(ib, shift=tuple(sv), order=1, mode="nearest")
            vals.append(psnr(ibs[8:-8, 8:-8], g[8:-8, 8:-8]))
        print(f"  {label} {o:+.2f}  {np.mean(vals):.3f}")
