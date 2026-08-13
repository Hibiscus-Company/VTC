#!/usr/bin/env python
"""DIAGNOSIS: what is actually different about our FLAT regions vs GT, through the FULL shipped
chain (4-member mean -> energy restore lam=1.0 -> median lens field lanczos4 -> JPEG q100/ss2)?

Numbers only. No operator yet."""
import io, os, sys
import numpy as np
import cv2
import torch
from PIL import Image

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, TMP)
sys.path.insert(0, os.path.join(TMP, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, upsample, warp
from energy_restore import restore

cv2.setNumThreads(8)
Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = [f"/mnt/d/avv/output/HCM0181_{t}/test_poses_renders_png"
       for t in ("gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7")]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def jpeg_rt(x):
    b = io.BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.float32) / 255.


gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
cache = np.load(f"{TMP}/lens/cache/pub_HCM0181.npz")
lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
print(f"{len(stems)} stems, using first {N}", flush=True)

acc = {}


def add(k, v):
    acc.setdefault(k, []).append(v)


psd_r = psd_g = None
os.makedirs("/mnt/d/avv/geoflat/crops", exist_ok=True)
for ii, s in enumerate(stems[:N]):
    mem = [load(os.path.join(d, s + ".png")) for d in MEM]
    ens = torch.stack(mem).mean(0)
    R = restore(ens, mem, 1.0, len(mem), 3).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    R = jpeg_rt(np.clip(warp(np.ascontiguousarray(R), lens, "lanczos"), 0, 1))
    G = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"), dtype=np.float32) / 255.
    H, W, _ = G.shape

    gy = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)
    ry = cv2.cvtColor(R, cv2.COLOR_RGB2GRAY)
    gm = np.abs(cv2.Sobel(gy, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(gy, cv2.CV_32F, 0, 1, 3))
    thr = np.percentile(gm, 80)                      # top 20% gradient = "strong edge"
    edge = (gm > thr).astype(np.uint8)
    dist = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 3)
    flat = dist > 6.0                                 # far from any edge
    near = dist <= 2.0
    add("flat_frac", flat.mean())

    d = R - G
    se = (d ** 2).sum(2)
    add("mse_all", se.mean() / 3)
    add("mse_flat", se[flat].mean() / 3)
    add("mse_near", se[near].mean() / 3)
    # ---- decompose the flat-region residual into LF / MF / HF
    for sig, nm in ((8.0, "lf"), (2.0, "mf")):
        db = cv2.GaussianBlur(d, (0, 0), sig)
        add(f"flat_{nm}_pow", (db ** 2).sum(2)[flat].mean() / 3)
    add("flat_hf_pow", ((d - cv2.GaussianBlur(d, (0, 0), 2.0)) ** 2).sum(2)[flat].mean() / 3)
    # ---- local texture amplitude (std in 7x7) in flat regions: ours vs GT
    for nm, X in (("R", R), ("G", G)):
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        m = cv2.blur(x, (7, 7))
        v = np.maximum(cv2.blur(x * x, (7, 7)) - m * m, 0)
        add(f"flatstd7_{nm}", np.sqrt(v[flat]).mean() * 255)
        m3 = cv2.blur(x, (3, 3))
        v3 = np.maximum(cv2.blur(x * x, (3, 3)) - m3 * m3, 0)
        add(f"flatstd3_{nm}", np.sqrt(v3[flat]).mean() * 255)
    # ---- colour drift in flat: per-channel mean residual
    for c, cn in enumerate("rgb"):
        add(f"flatbias_{cn}", d[..., c][flat].mean() * 255)
    add("flatbias_absmeanimg", np.abs(d[flat].mean(0)).mean() * 255)
    # ---- sky specifically: bright + low-sat + upper half
    hsv = cv2.cvtColor(G, cv2.COLOR_RGB2HSV)
    sky = (hsv[..., 2] > 0.55) & (hsv[..., 1] < 0.25)
    sky[int(H * 0.6):] = False
    if sky.mean() > 0.02:
        add("sky_frac", sky.mean())
        add("sky_mse", se[sky].mean() / 3)
        add("sky_bias", d[sky].mean() * 255)
        add("sky_std_R", np.sqrt(np.maximum(cv2.blur(ry * ry, (7, 7)) - cv2.blur(ry, (7, 7)) ** 2, 0))[sky].mean() * 255)
        add("sky_std_G", np.sqrt(np.maximum(cv2.blur(gy * gy, (7, 7)) - cv2.blur(gy, (7, 7)) ** 2, 0))[sky].mean() * 255)
    # ---- radial PSD of high-pass, restricted to a big flat window
    ys, xs = np.where(flat)
    if psd_r is None:
        # pick the 128x128 window with the most flat pixels (grid search on integral image)
        ii_f = cv2.integral(flat.astype(np.float32))
        best, bp = -1, None
        for y in range(0, H - 128, 32):
            for x in range(0, W - 128, 32):
                v = ii_f[y + 128, x + 128] - ii_f[y, x + 128] - ii_f[y + 128, x] + ii_f[y, x]
                if v > best:
                    best, bp = v, (y, x)
        y, x = bp
        add("psd_win_flatfrac", best / (128 * 128))
        wnd = np.hanning(128)[:, None] * np.hanning(128)[None, :]
        rr = np.hypot(*np.mgrid[-64:64, -64:64]).astype(int)
        psd_r, psd_g = [], []
        for nm, X in (("R", R), ("G", G)):
            p = np.abs(np.fft.fftshift(np.fft.fft2(cv2.cvtColor(X[y:y + 128, x:x + 128], cv2.COLOR_RGB2GRAY) * wnd))) ** 2
            rad = np.bincount(rr.ravel(), p.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)
            (psd_r if nm == "R" else psd_g).append(rad[:64])
        np.save("/mnt/d/avv/geoflat/psd.npy", np.stack([psd_r[0], psd_g[0]]))
        cv2.imwrite("/mnt/d/avv/geoflat/crops/flat_R.png",
                    cv2.cvtColor((R[y:y + 128, x:x + 128] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite("/mnt/d/avv/geoflat/crops/flat_G.png",
                    cv2.cvtColor((G[y:y + 128, x:x + 128] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print("flat window at", bp, "flatfrac", best / 16384, flush=True)
    if ii == 0:
        np.save("/mnt/d/avv/geoflat/ex_R.npy", R.astype(np.float16))
        np.save("/mnt/d/avv/geoflat/ex_G.npy", G.astype(np.float16))
        np.save("/mnt/d/avv/geoflat/ex_flat.npy", flat)
    print(f"  {ii}", end=" ", flush=True)

print()
for k in sorted(acc):
    v = np.array(acc[k], dtype=np.float64)
    print(f"{k:>22} {v.mean():12.6f}  (per-image sd {v.std():.6f})")
p = np.load("/mnt/d/avv/geoflat/psd.npy")
print("\nradial PSD (flat 128x128 window), ratio R/GT by frequency bin:")
for lo, hi in ((1, 4), (4, 8), (8, 16), (16, 32), (32, 48), (48, 64)):
    print(f"  bins {lo:2d}-{hi:2d} (period {128/hi:5.1f}-{128/lo:5.1f}px): "
          f"R {p[0][lo:hi].mean():10.3e}  GT {p[1][lo:hi].mean():10.3e}  "
          f"ratio {p[0][lo:hi].mean()/p[1][lo:hi].mean():6.3f}")
