#!/usr/bin/env python
"""Does SPATIALLY-VARYING member selection beat a global mean?  (user proposal, 31/07)

The idea: cut the mirror region from whichever member renders mirrors best, the plant region from
whichever renders plants best, and composite. Generalised here to its strongest form -- per-TILE
selection over the whole member pool -- so the answer bounds every weaker version of the idea
(including a hand-drawn 2-region mirror/plant split).

Three numbers matter and they must not be confused:
  ORACLE      per-tile argmin of error against THIS frame's GT. Illegal, and it fits noise.
  HONEST      the per-tile choice is fitted on 14 holes and APPLIED to the other 14, both ways.
              This is the only number that says whether the idea transfers.
  MEAN        the shipped rule, all members averaged.
If HONEST does not beat MEAN, spatial selection is fitting noise and the idea is dead regardless
of how the regions are drawn.

Also reports the error share of the DARK SPECULAR region (the glass table + black display panel),
segmented from GT luminance, to bound what a mirror-specialised model branch could ever win.
"""
import os, re, sys, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
POOL = {
    "sr01_s42":   "/mnt/d/avv/r36_shape/sr01/eval_png",
    "sr01_s42b":  "/mnt/d/avv/r36_shape/sr01b/eval_png",
    "sr01_s101":  "/mnt/d/avv/r38/sr01_s101/eval_png",
    "churn_s42":  "/mnt/d/avv/r43_bonsai/churn_r25n25/eval_png",
    "churn_s101": "/mnt/d/avv/r43_bonsai/churn_r25n25_s101/eval_png",
    "churn15k":   "/mnt/d/avv/r43_bonsai/churn_r25n15/eval_png",
    "sr0":        "/mnt/d/avv/r35_scalereg/sr0/eval_png",
    "sr0.03":     "/mnt/d/avv/r35_scalereg/sr0.03/eval_png",
    "minop001":   "/mnt/d/avv/r43_bonsai/minop001/eval_png",
    "appaffine":  "/mnt/d/avv/r38/appaffine/eval_png",
}
POOL = {k: v for k, v in POOL.items() if os.path.isdir(v) and len(os.listdir(v)) == 28}
print("pool:", list(POOL), flush=True)
gt = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
stems = sorted(gt); T = 60
E, DARK = {}, {}
G0 = None
for s in stems:
    g = np.asarray(Image.open(f"{GT}/{gt[s]}").convert("RGB"), dtype=np.float32) / 255.
    H, W, _ = g.shape; H = H // T * T; W = W // T * T; g = g[:H, :W]
    lum = g.mean(2)
    tv = lambda a: a.reshape(H // T, T, W // T, T).mean((1, 3))
    DARK[s] = tv((lum < 0.30).astype(np.float32))          # dark specular surface fraction per tile
    for k, d in POOL.items():
        r = np.asarray(Image.open(f"{d}/{s}.png").convert("RGB"), dtype=np.float32) / 255.
        E.setdefault(k, {})[s] = tv(((r[:H, :W] - g) ** 2).mean(2))
keys = list(POOL)
A = np.stack([[E[k][s] for s in stems] for k in keys])      # [M, 28, ty, tx]
D = np.stack([DARK[s] for s in stems])                      # [28, ty, tx]
mean_err = np.stack([np.mean([E[k][s] for k in keys], 0) for s in stems])  # true pixel-mean is not
# NOTE: averaging per-tile MSE of members != MSE of the averaged image. Compute the real mean image.
MEAN = np.zeros_like(A[0])
for i, s in enumerate(stems):
    g = np.asarray(Image.open(f"{GT}/{gt[s]}").convert("RGB"), dtype=np.float32) / 255.
    H, W, _ = g.shape; H = H // T * T; W = W // T * T; g = g[:H, :W]
    acc = np.zeros_like(g)
    for k, d in POOL.items():
        acc += np.asarray(Image.open(f"{d}/{s}.png").convert("RGB"), dtype=np.float32)[:H, :W] / 255.
    m = acc / len(POOL)
    MEAN[i] = ((m - g) ** 2).reshape(H // T, T, W // T, T).mean(2).mean((1, 2)).reshape(H // T, W // T) \
        if False else (((m - g) ** 2).mean(2)).reshape(H // T, T, W // T, T).mean((1, 3))
def psnr(e): return 10 * np.log10(1 / max(e.mean(), 1e-12))
print(f"\nsingle members (PSNR over all 28 holes, tile-MSE aggregate):")
for j, k in enumerate(keys): print(f"   {k:12s} {psnr(A[j]):.4f} dB")
best_j = int(np.argmin([A[j].mean() for j in range(len(keys))]))
print(f"   GLOBAL BEST single = {keys[best_j]}  {psnr(A[best_j]):.4f} dB")
print(f"   PIXEL MEAN of all {len(keys)} = {psnr(MEAN):.4f} dB   <- the shipped rule")
print(f"   ORACLE per-tile (illegal, fits noise) = {psnr(A.min(0)):.4f} dB")
half1 = np.arange(0, 28, 2); half2 = np.arange(1, 28, 2)
tot = []
for fit, app in ((half1, half2), (half2, half1)):
    choice = A[:, fit].mean(1).argmin(0)                    # per-tile best member on the FIT half
    sel = np.stack([A[choice[ty, tx], app, ty, tx] for ty in range(A.shape[2]) for tx in range(A.shape[3])])
    sel = sel.reshape(A.shape[2], A.shape[3], len(app)).transpose(2, 0, 1)
    tot.append(sel)
HON = np.concatenate(tot)
print(f"   HONEST per-tile (fit on 14 holes, applied to the other 14, both ways) = {psnr(HON):.4f} dB")
print(f"      vs pixel mean: {psnr(HON) - psnr(MEAN):+.4f} dB     vs global best single: {psnr(HON) - psnr(A[best_j]):+.4f} dB")
nu = len(np.unique(np.stack([A[:, h].mean(1).argmin(0) for h in (half1, half2)])))
agree = (A[:, half1].mean(1).argmin(0) == A[:, half2].mean(1).argmin(0)).mean()
print(f"      per-tile winner agrees between the two halves on {agree*100:.1f}% of tiles "
      f"(chance = {100/len(keys):.1f}%)")
dk = D.mean(0) > 0.5
print(f"\nDARK SPECULAR REGION (GT luminance < 0.30, the glass table + black panel):")
print(f"   {dk.mean()*100:.1f}% of tiles;  carries {A[best_j][:, dk].sum()/A[best_j].sum()*100:.1f}% of squared error")
print(f"   per-tile MSE there {A[best_j][:, dk].mean():.5f} vs elsewhere {A[best_j][:, ~dk].mean():.5f} "
      f"(ratio {A[best_j][:, dk].mean()/A[best_j][:, ~dk].mean():.2f}x)")
