#!/usr/bin/env python
"""Operator test. Insertion point: after the lens field, before the JPEG encode.

varrel.py measured, inside flat regions,   GT_localstd = 1.013 * our_localstd + 0.699/255.
Slope 1 => where we have texture we have the RIGHT AMOUNT of it; the whole deficit is an ADDITIVE
floor of ~0.7/255 that no gain can ever produce (and the Wiener gain is 0.93 < 1, so scaling up
strictly loses MSE). So: inject that floor as spectrum-matched, incoherent fine texture.

All masks are derived from OUR OWN render, never from GT, so every arm is shippable as written.

CONTROLS
  const07_all   same injection everywhere -> isolates the flat geography
  hfg1.15_flat  multiplicative gain on the same band in the same mask -> the axis said to be closed
  fit_g0.5/1.0  amplitude response curve
"""
import io, os, sys, json
import numpy as np, cv2, torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
cv2.setNumThreads(8); Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
SC = sys.argv[1] if len(sys.argv) > 1 else "HCM0181"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
CD = f"/mnt/d/avv/geoflat/cache_{SC}"
C1 = 0.6989 / 255.0
dev = "cuda"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 2.0)


def lstd(x, w=9):
    return np.sqrt(np.maximum(cv2.blur(x * x, (w, w)) - cv2.blur(x, (w, w)) ** 2, 0))


def enc(x):
    b = io.BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
    n = len(b.getvalue())
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float32) / 255., n


def t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0).to(dev)


ARMS = ["base", "fit_g0.5", "fit_g1.0", "const07_flat", "const07_all", "hfg1.15_flat"]
acc = {a: np.zeros(3) for a in ARMS}
nby = {a: 0 for a in ARMS}
stems = sorted(f[:-4] for f in os.listdir(CD) if f.endswith(".npz"))[:N]
print(f"{SC} n={len(stems)}", flush=True)
mask_agree = []
for i, s in enumerate(stems):
    z = np.load(os.path.join(CD, s + ".npz"))
    B = z["B"].astype(np.float32); G = z["G"].astype(np.float32); Dg = z["D"].astype(np.float32)
    by = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
    gm = np.abs(cv2.Sobel(by, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(by, cv2.CV_32F, 0, 1, 3))
    Db = cv2.distanceTransform((cv2.GaussianBlur(gm, (0, 0), 1.0) <=
                                np.percentile(cv2.GaussianBlur(gm, (0, 0), 1.0), 80)).astype(np.uint8),
                               cv2.DIST_L2, 3)
    M = cv2.GaussianBlur((Db > 6).astype(np.float32), (0, 0), 2.0)[..., None]
    mask_agree.append(float(((Db > 6) == (Dg > 6)).mean()))
    hb = hp(by)
    sB = lstd(hb)
    rng = np.random.default_rng(abs(hash(s)) % (2 ** 31))
    n = rng.standard_normal(by.shape).astype(np.float32)
    n = hp(n); n /= (n.std() + 1e-12)                      # unit-variance, sigma-2 high-pass spectrum
    n = n[..., None]
    a_fit = np.sqrt(np.maximum(2 * sB * C1 + C1 * C1, 0))[..., None]
    outs = {
        "base": B,
        "fit_g0.5": B + M * (0.5 * a_fit) * n,
        "fit_g1.0": B + M * a_fit * n,
        "const07_flat": B + M * C1 * n,
        "const07_all": B + C1 * n,
        "hfg1.15_flat": B + M * 0.15 * hp(B),
    }
    g = t(G)
    for a in ARMS:
        j, nb = enc(outs[a])
        nby[a] += nb
        r = t(j)
        with torch.no_grad():
            acc[a][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
            acc[a][1] += float(repo_ssim(r, g))
            acc[a][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
    if i % 10 == 0:
        print(f"  {i}", flush=True)

n = len(stems)
print(f"\nrender-derived flat mask agrees with GT-derived on {np.mean(mask_agree)*100:.1f}% of pixels")
print(f"{'arm':>14} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'dSCORE':>8} {'MB/60':>7}")
base = None; res = {}
for a in ARMS:
    P, S, L = acc[a] / n
    sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    base = sc if a == "base" else base
    res[a] = dict(score=sc, psnr=P, ssim=S, lpips=L, d=sc - base, mb=nby[a] / 1e6)
    print(f"{a:>14} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {sc-base:+8.4f} {nby[a]/1e6:7.1f}")
json.dump(res, open(f"/mnt/d/avv/geoflat/arms_{SC}.json", "w"), indent=1)
