#!/usr/bin/env python
"""Classical-CV diagnostics on the HCM0181 eval split (has GT).
D1 radial power spectrum GT vs member vs mean
D2 bias/variance decomposition across members
D3 residual whiteness
"""
import os, sys, glob
import numpy as np
from PIL import Image

GT = "/mnt/d/avv/evalsplit/HCM0181/eval_gt"
MEM = [
    "/mnt/d/avv/seedbank/HCM0181_ut_s101/eval_png",
    "/mnt/d/avv/seedbank/HCM0181_ut_s202/eval_png",
    "/mnt/d/avv/mip3d/HCM0181_mf0.2/eval_png",
    "/mnt/d/avv/lpsweep/HCM0181_lp0.3/eval_png",
    "/mnt/d/avv/tw_test/HCM0181_absgrad0/eval_png",
]
N_IMG = int(os.environ.get("NIMG", "12"))

gtmap = {os.path.splitext(f)[0]: os.path.join(GT, f) for f in os.listdir(GT)}
stems = sorted(gtmap)


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def gray(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def radial_ps(img):
    """radially averaged power spectrum of a gray image, Hann-windowed"""
    H, W = img.shape
    wy = np.hanning(H)[:, None]
    wx = np.hanning(W)[None, :]
    F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * wy * wx))
    P = (F.real ** 2 + F.imag ** 2)
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    # normalized radius in cycles/pixel: use min dimension for nyquist
    fy = (yy - cy) / H
    fx = (xx - cx) / W
    r = np.sqrt(fy ** 2 + fx ** 2)
    nb = 60
    edges = np.linspace(0, 0.5, nb + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nb - 1)
    s = np.bincount(idx.ravel(), weights=P.ravel(), minlength=nb)
    c = np.bincount(idx.ravel(), minlength=nb)
    return 0.5 * (edges[:-1] + edges[1:]), s / np.maximum(c, 1)


acc = {}
sel = stems[:: max(1, len(stems) // N_IMG)][:N_IMG]
print(f"using {len(sel)} eval frames of {len(stems)}")

ps_gt = ps_m1 = ps_mean = None
mse_mean = 0.0
var_tot = 0.0
n = 0
for s in sel:
    g = load(gtmap[s])
    ms = []
    ok = True
    for d in MEM:
        p = os.path.join(d, s + ".png")
        if not os.path.exists(p):
            ok = False
            break
        ms.append(load(p))
    if not ok:
        continue
    M = np.stack(ms)                      # K,H,W,3
    mean = M.mean(0)
    if mean.shape != g.shape:
        print("shape mismatch", s, mean.shape, g.shape)
        continue
    _, a = radial_ps(gray(g))
    f, b = radial_ps(gray(ms[0]))
    _, c = radial_ps(gray(mean))
    ps_gt = a if ps_gt is None else ps_gt + a
    ps_m1 = b if ps_m1 is None else ps_m1 + b
    ps_mean = c if ps_mean is None else ps_mean + c
    mse_mean += ((mean - g) ** 2).mean()
    var_tot += M.var(0).mean()
    n += 1

ps_gt /= n; ps_m1 /= n; ps_mean /= n
print(f"\nD2 BIAS/VARIANCE  (K={len(MEM)} members, n={n} frames)")
print(f"  MSE(mean,GT)        {mse_mean/n:.6e}   PSNR {10*np.log10(1.0/(mse_mean/n)):.3f} dB")
print(f"  mean per-pixel VAR  {var_tot/n:.6e}   (spread across members)")
print(f"  var/K residual left in mean = {var_tot/n/len(MEM):.6e}"
      f"  => infinite-member ceiling PSNR "
      f"{10*np.log10(1.0/max(mse_mean/n - var_tot/n/len(MEM),1e-12)):.3f} dB")
print(f"  ratio var(K)/MSE = {(var_tot/n/len(MEM))/(mse_mean/n)*100:.2f}%  <- how much of our error composition can still remove")

print("\nD1 RADIAL POWER SPECTRUM  (freq cyc/px, ratios vs GT)")
print(f"{'f':>7} {'GT':>12} {'m1/GT':>8} {'mean/GT':>8} {'mean/m1':>8}")
for i in range(0, 60, 2):
    print(f"{f[i]:7.4f} {ps_gt[i]:12.4e} {ps_m1[i]/ps_gt[i]:8.4f} "
          f"{ps_mean[i]/ps_gt[i]:8.4f} {ps_mean[i]/ps_m1[i]:8.4f}")
