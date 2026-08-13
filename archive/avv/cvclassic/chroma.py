#!/usr/bin/env python
"""Per-channel / luma-vs-chroma spectral match.

A real camera's chroma MTF is far lower than its luma MTF (Bayer CFA has half the
R/B samples, demosaic interpolates them, then JPEG 4:2:0 halves them again).
Our renders synthesise每 channel independently from SH -> chroma has FULL bandwidth.
If GT chroma rolls off faster than ours we carry EXCESS chroma HF = pure error,
and a chroma-only low-pass is a GT-STRUCTURE match (not a cleanup).
"""
import os, numpy as np
from PIL import Image

GT = "/mnt/d/avv/evalsplit/HCM0181/eval_gt"
MEM = ["/mnt/d/avv/seedbank/HCM0181_ut_s101/eval_png",
       "/mnt/d/avv/seedbank/HCM0181_ut_s202/eval_png",
       "/mnt/d/avv/mip3d/HCM0181_mf0.2/eval_png",
       "/mnt/d/avv/lpsweep/HCM0181_lp0.3/eval_png",
       "/mnt/d/avv/tw_test/HCM0181_absgrad0/eval_png"]
NB = 32
gtmap = {os.path.splitext(f)[0]: os.path.join(GT, f) for f in os.listdir(GT)}
stems = sorted(gtmap)
sel = stems[::5][:12]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def ycc(a):
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    cb = -0.168736 * a[..., 0] - 0.331264 * a[..., 1] + 0.5 * a[..., 2]
    cr = 0.5 * a[..., 0] - 0.418688 * a[..., 1] - 0.081312 * a[..., 2]
    return y, cb, cr


def binidx(H, W):
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((yy - cy) / H) ** 2 + ((xx - cx) / W) ** 2)
    e = np.linspace(0, 0.5, NB + 1)
    return np.clip(np.digitize(r, e) - 1, 0, NB - 1).ravel(), 0.5 * (e[:-1] + e[1:])


def psd(img, idx):
    H, W = img.shape
    w = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2((img - img.mean()) * w))) ** 2
    return np.bincount(idx, weights=P.ravel(), minlength=NB)


accG = np.zeros((3, NB)); accR = np.zeros((3, NB))
idx = f = None
n = 0
for s in sel:
    g = load(gtmap[s])
    ms = [load(os.path.join(d, s + ".png")) for d in MEM]
    if any(m.shape != g.shape for m in ms):
        continue
    r = np.stack(ms).mean(0)
    if idx is None:
        idx, f = binidx(*g.shape[:2])
    for k, (cg, cr_) in enumerate(zip(ycc(g), ycc(r))):
        accG[k] += psd(cg, idx)
        accR[k] += psd(cr_, idx)
    n += 1

print(f"n={n}\n{'f':>8} {'Y  r/GT':>9} {'Cb r/GT':>9} {'Cr r/GT':>9}   "
      f"{'GT Cb/Y':>9} {'ens Cb/Y':>9}")
for i in range(NB):
    print(f"{f[i]:8.4f} {accR[0][i]/accG[0][i]:9.4f} {accR[1][i]/accG[1][i]:9.4f} "
          f"{accR[2][i]/accG[2][i]:9.4f}   {accG[1][i]/accG[0][i]:9.5f} "
          f"{accR[1][i]/accR[0][i]:9.5f}")

hi = slice(NB // 2, NB)
print(f"\nHIGH BAND (f>0.25) power ratio render/GT:  Y {accR[0][hi].sum()/accG[0][hi].sum():.4f}"
      f"  Cb {accR[1][hi].sum()/accG[1][hi].sum():.4f}"
      f"  Cr {accR[2][hi].sum()/accG[2][hi].sum():.4f}")
