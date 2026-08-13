"""Shared metric utilities for the no-training metric-attack probes."""
import os, sys, io
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")

_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def _gauss(ws=11, sigma=1.5):
    g = torch.tensor([np.exp(-((x - ws // 2) ** 2) / (2 * sigma ** 2)) for x in range(ws)],
                     dtype=torch.float32)
    return g / g.sum()


def window(ws=11, ch=3, device="cpu", dtype=torch.float32):
    g = _gauss(ws).unsqueeze(1)
    w2 = g.mm(g.t()).unsqueeze(0).unsqueeze(0)
    return w2.expand(ch, 1, ws, ws).contiguous().to(device=device, dtype=dtype)


def ssim_map(a, b, ws=11):
    """Repo-exact SSIM map (zero-padded conv). a,b: [1,3,H,W] in [0,1]."""
    ch = a.shape[-3]
    w = window(ws, ch, a.device, a.dtype)
    p = ws // 2
    mu1 = F.conv2d(a, w, padding=p, groups=ch)
    mu2 = F.conv2d(b, w, padding=p, groups=ch)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(a * a, w, padding=p, groups=ch) - mu1s
    s2 = F.conv2d(b * b, w, padding=p, groups=ch) - mu2s
    s12 = F.conv2d(a * b, w, padding=p, groups=ch) - mu12
    return ((2 * mu12 + _C1) * (2 * s12 + _C2)) / ((mu1s + mu2s + _C1) * (s1 + s2 + _C2))


def ssim(a, b, ws=11):
    return ssim_map(a, b, ws).mean()


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def score(P, S, L):
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))


def to_t(arr_u8, device="cpu"):
    """HxWx3 uint8 -> [1,3,H,W] float in [0,1]"""
    return torch.from_numpy(np.ascontiguousarray(arr_u8)).float().div_(255.0) \
        .permute(2, 0, 1).unsqueeze(0).to(device)


def load_u8(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def jpeg_roundtrip(arr_u8, quality=95, subsampling=2, keep_rgb=False,
                   optimize=False, progressive=False, qtables=None):
    buf = io.BytesIO()
    kw = dict(quality=quality, subsampling=subsampling)
    if optimize:
        kw["optimize"] = True
    if progressive:
        kw["progressive"] = True
    if keep_rgb:
        kw["keep_rgb"] = True
    if qtables is not None:
        kw["qtables"] = qtables
    Image.fromarray(arr_u8).save(buf, "JPEG", **kw)
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return out, buf.getbuffer().nbytes


class LP:
    def __init__(self, device="cpu", net="vgg"):
        import lpips as lpips_pkg
        self.m = lpips_pkg.LPIPS(net=net, verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, a, b):
        return float(self.m(a * 2 - 1, b * 2 - 1).item())

    @torch.no_grad()
    def layers(self, a, b):
        """Per-layer LPIPS contributions (they sum to the total)."""
        m = self.m
        i0, i1 = m.scaling_layer(a * 2 - 1), m.scaling_layer(b * 2 - 1)
        f0, f1 = m.net.forward(i0), m.net.forward(i1)
        import lpips as lpips_pkg
        out = []
        for k in range(m.L):
            d = (lpips_pkg.normalize_tensor(f0[k]) - lpips_pkg.normalize_tensor(f1[k])) ** 2
            v = m.lins[k](d).mean([2, 3], keepdim=False).squeeze()
            out.append(float(v))
        return out
