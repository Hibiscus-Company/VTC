#!/usr/bin/env python
"""Cheap (no-LPIPS) diagnostics on the 220 TRAIN pairs (ema099 train renders vs train photos).
Fits: per-channel gain/bias, gamma, MSE-optimal unsharp alpha per sigma, sub-pixel shift.
LEGAL: train photos only.  Never touches eval GT."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
import json, sys, time
import multiprocessing as mp
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy.ndimage import gaussian_filter, shift as ndshift

RD = "/mnt/d/avv/blurbound/bonsai/train_render"
GD = "/mnt/d/avv/evalsplit/bonsai/train_sub/images"


def one(stem):
    r = np.asarray(Image.open(f"{RD}/{stem}.jpg").convert("RGB"), np.float64) / 255.
    g = np.asarray(Image.open(f"{GD}/{stem}.jpg").convert("RGB"), np.float64) / 255.
    out = {"stem": stem}
    # 1. per-channel LS gain/bias  g ~ a*r + b
    for c in range(3):
        x = r[..., c].ravel(); y = g[..., c].ravel()
        mx, my = x.mean(), y.mean()
        a = ((x - mx) * (y - my)).sum() / ((x - mx) ** 2).sum()
        out[f"gain{c}"] = a
        out[f"bias{c}"] = my - a * mx
    out["mse0"] = float(((r - g) ** 2).mean())
    # 2. MSE-optimal unsharp alpha per sigma (closed form: alpha* = <d, e>/<d,d>)
    for s in (0.6, 1.0, 1.6, 2.5):
        blur = np.stack([gaussian_filter(r[..., c], s, mode="reflect") for c in range(3)], -1)
        d = r - blur
        e = g - r
        out[f"alpha_mse_s{s}"] = float((d * e).sum() / (d * d).sum())
        out[f"mse_gain_s{s}"] = float(out["mse0"] - ((e - out[f"alpha_mse_s{s}"] * d) ** 2).mean())
    # 3. sub-pixel shift by parabola fit on integer cross-correlation of luma gradients
    ry = r @ [0.299, 0.587, 0.114]
    gy = g @ [0.299, 0.587, 0.114]
    ry = ry - ry.mean(); gy = gy - gy.mean()
    F1 = np.fft.rfft2(ry); F2 = np.fft.rfft2(gy)
    cc = np.fft.irfft2(F2 * np.conj(F1), s=ry.shape)
    cc = np.fft.fftshift(cc)
    H, W = cc.shape
    pk = np.unravel_index(np.argmax(cc), cc.shape)
    dy0, dx0 = pk[0] - H // 2, pk[1] - W // 2

    def par(c0, cm, cp):
        den = (cm - 2 * c0 + cp)
        return 0.0 if den == 0 else 0.5 * (cm - cp) / den
    sy = par(cc[pk], cc[pk[0] - 1, pk[1]], cc[pk[0] + 1, pk[1]]) if 0 < pk[0] < H - 1 else 0.
    sx = par(cc[pk], cc[pk[0], pk[1] - 1], cc[pk[0], pk[1] + 1]) if 0 < pk[1] < W - 1 else 0.
    out["shift_dx"] = float(dx0 + sx)   # how far GT sits to the right of the render
    out["shift_dy"] = float(dy0 + sy)
    # 4. sharpness
    out["lapvar_r"] = float(np.var(np.gradient(ry)[0]))
    out["lapvar_g"] = float(np.var(np.gradient(gy)[0]))
    # 5. noise proxy (Immerkaer) on GT and render
    K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    from scipy.signal import convolve2d
    for nm, im in (("r", ry), ("g", gy)):
        v = convolve2d(im, K, mode="valid")
        out[f"noise_{nm}"] = float(np.sqrt(np.pi / 2) / (6 * (im.size ** .5)) * np.abs(v).sum() * 255 / (im.size ** .5) * (im.size ** .5) / (im.size ** .5))
        out[f"noise_{nm}"] = float(np.abs(v).mean() * np.sqrt(np.pi / 2) / 6 * 255)
    return out


if __name__ == "__main__":
    stems = sorted(f[:-4] for f in os.listdir(RD) if f.endswith(".jpg"))
    t0 = time.time()
    with mp.Pool(6) as p:
        rows = p.map(one, stems, chunksize=4)
    json.dump(rows, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/b1/diag_fit.json", "w"))
    import statistics as st
    keys = [k for k in rows[0] if k != "stem"]
    print(f"n={len(rows)}  {time.time()-t0:.0f}s")
    print(f"{'key':<20s} {'mean':>12s} {'median':>12s} {'sd':>10s} {'p10':>10s} {'p90':>10s}")
    for k in keys:
        v = np.array([r[k] for r in rows])
        print(f"{k:<20s} {v.mean():12.6f} {np.median(v):12.6f} {v.std():10.6f} "
              f"{np.percentile(v,10):10.6f} {np.percentile(v,90):10.6f}")
