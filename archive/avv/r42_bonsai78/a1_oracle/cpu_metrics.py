"""CPU-only metric core for A1. Metric definitions copied verbatim in spirit from
scripts/eval_score.py: repo SSIM (utils.loss_utils.ssim), lpips.LPIPS(net='vgg'),
PSNR = 10*log10(1/mse). NEVER touches CUDA."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
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
    """float32 CHW tensor in [0,1], shape (1,3,H,W) -- identical to eval_score.load."""
    return torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)


def metrics(r, g):
    """r,g: (1,3,H,W) float tensors in [0,1]. Returns (psnr, ssim, lpips_vgg)."""
    with torch.no_grad():
        mse = ((r - g) ** 2).mean().item()
        psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
        s = float(repo_ssim(r, g))
        l = float(vgg()(r * 2 - 1, g * 2 - 1).item())
    return psnr, s, l


# --- memory/time-optimised LPIPS: cache the GT's normalised VGG features so the
# reference image is only pushed through VGG once per frame.  Verified bit-close
# to lpips_pkg.LPIPS(net='vgg')(r*2-1, g*2-1) (see verify_fast()). ---
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


# ---------------- operators ----------------
def gauss_kernel1d(sigma):
    rad = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    return torch.tensor(k, dtype=torch.float32), rad


def gaussian_blur(img, sigma):
    """img (1,3,H,W). Separable Gaussian, reflect padding."""
    k, rad = gauss_kernel1d(sigma)
    C = img.shape[1]
    kh = k.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()
    kv = k.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()
    x = F.pad(img, (rad, rad, 0, 0), mode="reflect")
    x = F.conv2d(x, kh, groups=C)
    x = F.pad(x, (0, 0, rad, rad), mode="reflect")
    x = F.conv2d(x, kv, groups=C)
    return x


def unsharp(img, sigma, a):
    if a == 0.0:
        return img.clamp(0, 1)
    return (img + a * (img - gaussian_blur(img, sigma))).clamp(0, 1)


def quantize8(img):
    """uint8 delivery round-trip (what actually ships)."""
    return (img.clamp(0, 1) * 255.0).round() / 255.0


def lapvar(img):
    """Variance of the 4-neighbour Laplacian of the [0,1] luma image."""
    with torch.no_grad():
        y = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).unsqueeze(1)
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        lap = F.conv2d(y, k)
        return float(lap.var().item())


# ---------------- radial spectral match ----------------
def _radial_setup(H, W, nbins=720):
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    rmax = r.max()
    idx = np.minimum((r / rmax * nbins).astype(np.int64), nbins - 1)
    cnt = np.bincount(idx.ravel(), minlength=nbins).astype(np.float64)
    return idx, cnt, nbins


def radial_gain_apply(rimg, gimg, per_channel=True, nbins=720, gclip=(0.2, 6.0)):
    """rimg,gimg: (1,3,H,W) float. Returns filtered render (clipped to [0,1]) and gain curve(s)."""
    R = rimg[0].numpy().astype(np.float64)
    G = gimg[0].numpy().astype(np.float64)
    H, W = R.shape[1], R.shape[2]
    idx, cnt, nb = _radial_setup(H, W, nbins)
    Fr = np.fft.rfft2(R, axes=(1, 2))
    Fg = np.fft.rfft2(G, axes=(1, 2))
    Pr = np.abs(Fr) ** 2
    Pg = np.abs(Fg) ** 2
    out = np.empty_like(R)
    gains = []
    if per_channel:
        for c in range(3):
            pr = np.bincount(idx.ravel(), weights=Pr[c].ravel(), minlength=nb) / np.maximum(cnt, 1)
            pg = np.bincount(idx.ravel(), weights=Pg[c].ravel(), minlength=nb) / np.maximum(cnt, 1)
            g = np.sqrt(pg / np.maximum(pr, 1e-20))
            g = np.clip(np.nan_to_num(g, nan=1.0), gclip[0], gclip[1])
            gains.append(g)
            out[c] = np.fft.irfft2(Fr[c] * g[idx], s=(H, W))
    else:
        pr = sum(np.bincount(idx.ravel(), weights=Pr[c].ravel(), minlength=nb) for c in range(3)) / np.maximum(3 * cnt, 1)
        pg = sum(np.bincount(idx.ravel(), weights=Pg[c].ravel(), minlength=nb) for c in range(3)) / np.maximum(3 * cnt, 1)
        g = np.sqrt(pg / np.maximum(pr, 1e-20))
        g = np.clip(np.nan_to_num(g, nan=1.0), gclip[0], gclip[1])
        gains.append(g)
        gg = g[idx]
        for c in range(3):
            out[c] = np.fft.irfft2(Fr[c] * gg, s=(H, W))
    t = torch.from_numpy(out.astype(np.float32)).unsqueeze(0).clamp(0, 1)
    return t, gains


def radial_wiener_apply(rimg, gimg, per_channel=True, nbins=720, gclip=(0.0, 6.0)):
    """MSE-OPTIMAL zero-phase radial gain: g(r) = sum Re(Fgt*conj(Fren)) / sum |Fren|^2
    over each radial bin.  This is the least-squares best member of the same family as
    radial_gain_apply (which uses the power-matching gain sqrt(Pgt/Pren) -- provably not
    MSE-optimal).  Reported alongside so the family ceiling is not understated."""
    R = rimg[0].numpy().astype(np.float64)
    G = gimg[0].numpy().astype(np.float64)
    H, W = R.shape[1], R.shape[2]
    idx, cnt, nb = _radial_setup(H, W, nbins)
    Fr = np.fft.rfft2(R, axes=(1, 2))
    Fg = np.fft.rfft2(G, axes=(1, 2))
    out = np.empty_like(R)
    gains = []
    chans = range(3) if per_channel else [None]
    if per_channel:
        for c in range(3):
            num = np.real(Fg[c] * np.conj(Fr[c]))
            den = np.abs(Fr[c]) ** 2
            n_ = np.bincount(idx.ravel(), weights=num.ravel(), minlength=nb)
            d_ = np.bincount(idx.ravel(), weights=den.ravel(), minlength=nb)
            g = np.clip(np.nan_to_num(n_ / np.maximum(d_, 1e-20), nan=1.0), gclip[0], gclip[1])
            gains.append(g)
            out[c] = np.fft.irfft2(Fr[c] * g[idx], s=(H, W))
    else:
        num = sum(np.real(Fg[c] * np.conj(Fr[c])) for c in range(3))
        den = sum(np.abs(Fr[c]) ** 2 for c in range(3))
        n_ = np.bincount(idx.ravel(), weights=num.ravel(), minlength=nb)
        d_ = np.bincount(idx.ravel(), weights=den.ravel(), minlength=nb)
        g = np.clip(np.nan_to_num(n_ / np.maximum(d_, 1e-20), nan=1.0), gclip[0], gclip[1])
        gains.append(g)
        gg = g[idx]
        for c in range(3):
            out[c] = np.fft.irfft2(Fr[c] * gg, s=(H, W))
    t = torch.from_numpy(out.astype(np.float32)).unsqueeze(0).clamp(0, 1)
    return t, gains
