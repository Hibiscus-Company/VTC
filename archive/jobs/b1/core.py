"""B1 CPU-only metric core + render-time operator library.

Metric definitions copied verbatim from scripts/eval_score.py:
  repo SSIM (utils.loss_utils.ssim), lpips.LPIPS(net='vgg'), PSNR = 10*log10(1/mse).
NEVER touches CUDA.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import io
import sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg

_VGG = None


def vgg():
    global _VGG
    if _VGG is None:
        _VGG = lpips_pkg.LPIPS(net="vgg").eval()
        for p in _VGG.parameters():
            p.requires_grad_(False)
    return _VGG


def load(p):
    return torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)


def gt_feats(g):
    net = vgg()
    with torch.no_grad():
        outs = net.net.forward(net.scaling_layer(g * 2 - 1))
        return [lpips_pkg.normalize_tensor(o).clone() for o in outs]


def lpips_fast(feats_g, r):
    net = vgg()
    with torch.no_grad():
        outs = list(net.net.forward(net.scaling_layer(r * 2 - 1)))
        val = 0.0
        for kk in range(net.L):
            f = lpips_pkg.normalize_tensor(outs[kk])
            outs[kk] = None
            d = (f - feats_g[kk]) ** 2
            del f
            val = val + lpips_pkg.spatial_average(net.lins[kk](d), keepdim=True)
            del d
        return float(val.item())


def metrics_fast(r, g, feats_g):
    with torch.no_grad():
        mse = ((r - g) ** 2).mean().item()
        psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
        s = float(repo_ssim(r, g))
    return psnr, s, lpips_fast(feats_g, r)


def score_from(P, S, L, psnr_max=50.0):
    return 100.0 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / psnr_max, 1.0))


# ---------------- delivery ----------------
SHIP_ENCODE = dict(format="JPEG", quality=100, subsampling=2, optimize=True, progressive=True)


def to_u8(img):
    a = (img.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return a


def from_u8(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def quantize8(img):
    """Plain uint8 delivery round-trip (PNG shipping)."""
    return from_u8(to_u8(img))


def ship_encode(img):
    """Exact shipping encode: PIL JPEG q=100 subsampling=2 optimize progressive, then decode."""
    buf = io.BytesIO()
    Image.fromarray(to_u8(img)).save(buf, **SHIP_ENCODE)
    nbytes = buf.tell()
    buf.seek(0)
    out = from_u8(np.asarray(Image.open(buf).convert("RGB")))
    return out, nbytes


# ---------------- primitives ----------------
def gauss_kernel1d(sigma):
    rad = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    return torch.tensor(k, dtype=torch.float32), rad


def gaussian_blur(img, sigma):
    k, rad = gauss_kernel1d(sigma)
    C = img.shape[1]
    kh = k.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()
    kv = k.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()
    x = F.pad(img, (rad, rad, 0, 0), mode="reflect")
    x = F.conv2d(x, kh, groups=C)
    x = F.pad(x, (0, 0, rad, rad), mode="reflect")
    x = F.conv2d(x, kv, groups=C)
    return x


def box_blur(img, r):
    """Integral-image box filter, radius r (window 2r+1), reflect-free (normalised edges)."""
    C = img.shape[1]
    k = 2 * r + 1
    kh = torch.full((C, 1, 1, k), 1.0 / k)
    kv = torch.full((C, 1, k, 1), 1.0 / k)
    x = F.pad(img, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, kh, groups=C)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, kv, groups=C)
    return x


# ---------------- operators ----------------
def op_unsharp(img, sigma=1.0, a=0.0):
    """A1 family: y = x + a*(x - G_sigma x).  a<0 == blur."""
    if a == 0.0:
        return img
    return img + a * (img - gaussian_blur(img, sigma))


# BT.601 luma, matching the usual JPEG YCbCr convention
_W = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def luma(img):
    return (img * _W).sum(1, keepdim=True)


def op_luma_unsharp(img, sigma=1.0, a=0.0):
    """Apply the unsharp/blur only to the luma channel; chroma differences untouched."""
    if a == 0.0:
        return img
    y = luma(img)
    dy = a * (y - gaussian_blur(y, sigma))
    return img + dy


def op_chroma_blur(img, sigma=1.0):
    """Blur only the colour-difference channels (Cb, Cr), keep luma exact."""
    if sigma <= 0:
        return img
    y = luma(img)
    c = img - y                      # zero-luma chroma residual
    cb = gaussian_blur(c, sigma)
    # re-project so the blurred chroma still carries zero luma
    cb = cb - luma(cb)
    return y + cb


def guided_filter(img, r=8, eps=1e-3):
    """Self-guided filter (He et al.), per channel, box radius r."""
    mean_I = box_blur(img, r)
    mean_II = box_blur(img * img, r)
    var_I = mean_II - mean_I * mean_I
    a = var_I / (var_I + eps)
    b = mean_I - a * mean_I
    mean_a = box_blur(a, r)
    mean_b = box_blur(b, r)
    return mean_a * img + mean_b


def op_guided(img, r=8, eps=1e-3, w=0.0):
    """w>0 : edge-preserving smoothing (removes 'crunch', keeps edges).
       w<0 : edge-aware detail boost."""
    if w == 0.0:
        return img
    return img + w * (guided_filter(img, r, eps) - img)


def op_tone(img, gain=(1.0, 1.0, 1.0), bias=(0.0, 0.0, 0.0)):
    g = torch.tensor(gain, dtype=torch.float32).view(1, 3, 1, 1)
    b = torch.tensor(bias, dtype=torch.float32).view(1, 3, 1, 1)
    return img * g + b


def op_gamma(img, g=1.0):
    if g == 1.0:
        return img
    return img.clamp(0, 1) ** g


def op_shift(img, dx=0.0, dy=0.0):
    """Sub-pixel translation by (dx, dy) px via bicubic grid_sample. +dx moves content right."""
    if dx == 0.0 and dy == 0.0:
        return img
    _, _, H, W = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    gx = (xs - dx) / (W - 1) * 2 - 1
    gy = (ys - dy) / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    return F.grid_sample(img, grid, mode="bicubic", padding_mode="reflection",
                         align_corners=True)


def op_noise(img, sd=0.0, seed=0):
    """Deterministic seeded Gaussian grain (sd in [0,1] units)."""
    if sd <= 0:
        return img
    g = torch.Generator().manual_seed(int(seed))
    n = torch.randn(img.shape, generator=g) * sd
    return img + n


# ---------------- composition ----------------
DEFAULTS = dict(
    shift_dx=0.0, shift_dy=0.0,
    guided_r=8, guided_eps=1e-3, guided_w=0.0,
    us_sigma=1.0, us_a=0.0, us_luma_only=0,
    chroma_sigma=0.0,
    gain=(1.0, 1.0, 1.0), bias=(0.0, 0.0, 0.0), gamma=1.0,
    noise_sd=0.0, noise_seed=0,
)


def apply_op(img, p):
    """Deterministic, side-effect free.  Fixed order:
       shift -> guided -> unsharp/blur -> chroma blur -> tone -> gamma -> noise -> clip."""
    q = dict(DEFAULTS)
    q.update(p or {})
    x = img
    x = op_shift(x, q["shift_dx"], q["shift_dy"])
    x = op_guided(x, int(q["guided_r"]), float(q["guided_eps"]), float(q["guided_w"]))
    if q["us_luma_only"]:
        x = op_luma_unsharp(x, float(q["us_sigma"]), float(q["us_a"]))
    else:
        x = op_unsharp(x, float(q["us_sigma"]), float(q["us_a"]))
    x = op_chroma_blur(x, float(q["chroma_sigma"]))
    x = op_tone(x, tuple(q["gain"]), tuple(q["bias"]))
    x = op_gamma(x, float(q["gamma"]))
    x = op_noise(x, float(q["noise_sd"]), int(q["noise_seed"]))
    return x.clamp(0, 1)
