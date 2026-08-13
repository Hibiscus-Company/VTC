"""B1 phase-2: delivery models + chroma operators.  CPU ONLY.  Extends core.py."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(1)

import core  # noqa: E402  (metric core, verified against scripts/eval_score.py)

# ---------------------------------------------------------------- JPEG colour space
# JPEG / BT.601 full-range, exactly what libjpeg uses.
_M_FWD = torch.tensor([[0.299, 0.587, 0.114],
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]], dtype=torch.float32)
_M_INV = torch.tensor([[1.0, 0.0, 1.402],
                       [1.0, -0.344136, -0.714136],
                       [1.0, 1.772, 0.0]], dtype=torch.float32)
_OFF = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)


def rgb2ycc(x):
    return torch.einsum("ij,bjhw->bihw", _M_FWD, x) + _OFF


def ycc2rgb(x):
    return torch.einsum("ij,bjhw->bihw", _M_INV, x - _OFF)


# ---------------------------------------------------------------- libjpeg h2v2 down/up
def h2v2_down(c):
    """libjpeg h2v2_downsample: simple 2x2 box average.  c: (B,K,H,W), H,W even-padded."""
    B, K, H, W = c.shape
    ph, pw = H % 2, W % 2
    if ph or pw:
        c = F.pad(c, (0, pw, 0, ph), mode="replicate")
    return F.avg_pool2d(c, 2)


def h2v2_fancy_up(d, H, W):
    """libjpeg h2v2_fancy_upsample: separable triangle filter, weights 3/4 & 1/4 -> 9/16 3/16 3/16 1/16.
    Implemented as: replicate-pad, then for each output parity take 0.75*near + 0.25*far."""
    B, K, h, w = d.shape
    p = F.pad(d, (1, 1, 1, 1), mode="replicate")          # (B,K,h+2,w+2)
    # horizontal pass -> width 2w
    left = 0.75 * p[:, :, :, 1:w + 1] + 0.25 * p[:, :, :, 0:w]        # even output cols
    right = 0.75 * p[:, :, :, 1:w + 1] + 0.25 * p[:, :, :, 2:w + 2]   # odd output cols
    hcat = torch.stack([left, right], dim=-1).reshape(B, K, h + 2, 2 * w)
    # vertical pass -> height 2h
    top = 0.75 * hcat[:, :, 1:h + 1, :] + 0.25 * hcat[:, :, 0:h, :]
    bot = 0.75 * hcat[:, :, 1:h + 1, :] + 0.25 * hcat[:, :, 2:h + 2, :]
    out = torch.stack([top, bot], dim=-2).reshape(B, K, 2 * h, 2 * w)
    return out[:, :, :H, :W]


def chroma_DU(c, H, W):
    """The exact linear map the 4:2:0 encoder applies to the chroma planes."""
    return h2v2_fancy_up(h2v2_down(c), H, W)


def sim420(img, quant=True):
    """Float-domain simulation of 4:2:0 chroma sub/upsampling only -- no DCT, no entropy coding.
    quant=True also rounds Y,Cb,Cr to 8 bits at the points libjpeg does."""
    H, W = img.shape[-2:]
    y = rgb2ycc(img)
    if quant:
        y = torch.round(y * 255.0) / 255.0
    c = chroma_DU(y[:, 1:], H, W)
    if quant:
        c = torch.round(c * 255.0) / 255.0
    out = ycc2rgb(torch.cat([y[:, :1], c], 1))
    return out.clamp(0, 1)


# ---------------------------------------------------------------- real codec deliveries
def _jpeg_roundtrip(img, **kw):
    a = (img.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", **kw)
    n = buf.tell()
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"))
    return torch.from_numpy(out.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0), n


SHIP = dict(quality=100, subsampling=2, optimize=True, progressive=True)

DELIVERIES = {
    "float":   lambda x: (x, -1),
    "u8":      lambda x: (core.quantize8(x), -1),
    "ship":    lambda x: _jpeg_roundtrip(x, **SHIP),
    "j444":    lambda x: _jpeg_roundtrip(x, quality=100, subsampling=0, optimize=True, progressive=True),
    "j422":    lambda x: _jpeg_roundtrip(x, quality=100, subsampling=1, optimize=True, progressive=True),
    "j420seq": lambda x: _jpeg_roundtrip(x, quality=100, subsampling=2, optimize=True, progressive=False),
    "sim420":  lambda x: (sim420(x, quant=True), -1),
    "sim420f": lambda x: (sim420(x, quant=False), -1),
}


def deliver(img, name):
    return DELIVERIES[name](img)


# ---------------------------------------------------------------- chroma operators
def op_chroma_precomp(img, iters=3, lam=1.0, clampc=True):
    """Pre-compensate the render's chroma for the encoder's 4:2:0 down/up-sample chain.

    Solves (approximately)   argmin_x || DU(x) - c ||^2   by van Cittert / Landweber
    iteration against the EXACT linear operator DU (h2v2 box-down + libjpeg fancy triangle up):
        x_0 = c ;  x_{k+1} = x_k + lam * (c - DU(x_k))
    so that after the encoder subsamples, the reconstructed chroma is closer to the render's
    own chroma.  Uses NOTHING but the render itself -- no GT of any kind, so it cannot leak.
    Luma is left bit-exact.
    """
    if iters <= 0:
        return img
    H, W = img.shape[-2:]
    y = rgb2ycc(img)
    c = y[:, 1:]
    x = c.clone()
    for _ in range(int(iters)):
        x = x + lam * (c - chroma_DU(x, H, W))
        if clampc:
            x = x.clamp(0.0, 1.0)
    return ycc2rgb(torch.cat([y[:, :1], x], 1)).clamp(0, 1)


def op_chroma_prefilt(img, sigma=0.0):
    """Anti-alias the chroma BEFORE the encoder's box downsample (the opposite direction to
    precomp): trades colour sharpness for less subsampling aliasing.  Luma untouched."""
    if sigma <= 0:
        return img
    y = rgb2ycc(img)
    c = core.gaussian_blur(y[:, 1:], sigma)
    return ycc2rgb(torch.cat([y[:, :1], c], 1)).clamp(0, 1)


def op_luma_only_precomp(img, iters=3, lam=1.0):
    """Control: same iteration applied to LUMA (which the encoder does NOT subsample).
    Should be a no-op-ish; used to prove the gain is chroma-specific."""
    if iters <= 0:
        return img
    H, W = img.shape[-2:]
    y = rgb2ycc(img)
    l0 = y[:, :1]
    x = l0.clone()
    for _ in range(int(iters)):
        x = (x + lam * (l0 - chroma_DU(x, H, W))).clamp(0, 1)
    return ycc2rgb(torch.cat([x, y[:, 1:]], 1)).clamp(0, 1)


# ---------------------------------------------------------------- combined operator
OPS = {}


def apply_op2(img, p):
    """Extended pipeline.  Order:  [core.apply_op]  ->  chroma prefilt  ->  chroma precomp.
    Deterministic, side-effect free."""
    q = dict(p or {})
    cp_iters = q.pop("cp_iters", 0)
    cp_lam = q.pop("cp_lam", 1.0)
    cp_clamp = q.pop("cp_clamp", 1)
    pf_sigma = q.pop("pf_sigma", 0.0)
    lp_iters = q.pop("lp_iters", 0)
    x = core.apply_op(img, q) if q else img
    if pf_sigma:
        x = op_chroma_prefilt(x, float(pf_sigma))
    if lp_iters:
        x = op_luma_only_precomp(x, int(lp_iters), float(cp_lam))
    if cp_iters:
        x = op_chroma_precomp(x, int(cp_iters), float(cp_lam), bool(cp_clamp))
    return x.clamp(0, 1)
