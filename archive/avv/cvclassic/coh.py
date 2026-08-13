#!/usr/bin/env python
"""Cross-spectral analysis: render -> GT.
For each radial frequency band:
  gain_LS(f)   = Re<G R*> / <|R|^2>          MSE-optimal linear gain (Wiener)
  coh2(f)      = |<G R*>|^2/(<|R|^2><|G|^2>) coherence: fraction of GT energy LINEARLY predictable
  gain_match(f)= sqrt(<|G|^2>/<|R|^2>)       PSD-matching gain (perceptual / texture)
If coh2 is high and gain_LS<1 -> we have EXCESS incoherent energy (noise) -> shrink.
If coh2 is high and gain_LS>1 -> we are genuinely ATTENUATED -> a boost is MSE-justified.
"""
import os, numpy as np
from PIL import Image

GT = "/mnt/d/avv/evalsplit/HCM0181/eval_gt"
MEM = [
    "/mnt/d/avv/seedbank/HCM0181_ut_s101/eval_png",
    "/mnt/d/avv/seedbank/HCM0181_ut_s202/eval_png",
    "/mnt/d/avv/mip3d/HCM0181_mf0.2/eval_png",
    "/mnt/d/avv/lpsweep/HCM0181_lp0.3/eval_png",
    "/mnt/d/avv/tw_test/HCM0181_absgrad0/eval_png",
]
NB = 48
N_IMG = int(os.environ.get("NIMG", "16"))

gtmap = {os.path.splitext(f)[0]: os.path.join(GT, f) for f in os.listdir(GT)}
stems = sorted(gtmap)
sel = stems[:: max(1, len(stems) // N_IMG)][:N_IMG]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def gray(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def bins(H, W):
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((yy - cy) / H) ** 2 + ((xx - cx) / W) ** 2)
    edges = np.linspace(0, 0.5, NB + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, NB - 1)
    return idx.ravel(), 0.5 * (edges[:-1] + edges[1:])


def F(img):
    H, W = img.shape
    w = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    return np.fft.fftshift(np.fft.fft2((img - img.mean()) * w))


Srr = np.zeros(NB); Sgg = np.zeros(NB); Sgr_r = np.zeros(NB); Sgr_i = np.zeros(NB)
idx = f = None
n = 0
for s in sel:
    g = load(gtmap[s])
    ms = [load(os.path.join(d, s + ".png")) for d in MEM
          if os.path.exists(os.path.join(d, s + ".png"))]
    if len(ms) < len(MEM):
        continue
    r = np.stack(ms).mean(0)
    if r.shape != g.shape:
        continue
    Gg, Rr = F(gray(g)), F(gray(r))
    if idx is None:
        idx, f = bins(*gray(g).shape)
    C = Gg * np.conj(Rr)
    for arr, val in ((Srr, np.abs(Rr) ** 2), (Sgg, np.abs(Gg) ** 2),
                     (Sgr_r, C.real), (Sgr_i, C.imag)):
        arr += np.bincount(idx, weights=val.ravel(), minlength=NB)
    n += 1

gain_ls = Sgr_r / Srr
coh2 = (Sgr_r ** 2 + Sgr_i ** 2) / (Srr * Sgg)
gain_match = np.sqrt(Sgg / Srr)
# residual PSD after the optimal linear filter, relative to GT PSD
resid_frac = 1.0 - coh2

print(f"n={n} frames, ensemble of {len(MEM)} members\n")
print(f"{'f cyc/px':>9} {'PSD r/GT':>9} {'gain_LS':>8} {'coh^2':>7} {'gain_match':>10} {'unexplained':>11}")
for i in range(NB):
    print(f"{f[i]:9.4f} {Srr[i]/Sgg[i]:9.4f} {gain_ls[i]:8.4f} {coh2[i]:7.4f} "
          f"{gain_match[i]:10.4f} {resid_frac[i]:11.4f}")

# how much MSE would the optimal radial (zero-phase) filter recover?
# total GT energy and residual energy under identity vs under gain_ls
E_gt = Sgg.sum()
res_id = (Sgg - 2 * Sgr_r + Srr).sum()
res_opt = (Sgg - Sgr_r ** 2 / Srr).sum()
print(f"\nresidual energy: identity {res_id:.4e}  optimal-radial-filter {res_opt:.4e}"
      f"  -> {10*np.log10(res_id/res_opt):+.4f} dB available from a ZERO-PHASE RADIAL filter")
res_match = (Sgg - 2 * gain_match * Sgr_r + gain_match ** 2 * Srr).sum()
print(f"                 PSD-matching filter {res_match:.4e} -> {10*np.log10(res_id/res_match):+.4f} dB")
