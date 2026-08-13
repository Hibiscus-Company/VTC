"""Shared machinery: field construction (grid + parametric) and production-exact scoring."""
import os, sys, io, json
import numpy as np
from PIL import Image
import cv2
import torch

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim  # noqa: E402

FLOW = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/flow"
PUB = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
PRIV = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674"]

# ---------------------------------------------------------------- scoring


def load_u8(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_t(a, dev):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev).float().div_(255.0) \
        .permute(2, 0, 1).unsqueeze(0)


def score(P, S, L):
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))


class Scorer:
    def __init__(self, dev="cuda"):
        import lpips as lpips_pkg
        self.dev = dev
        self.lp = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    @torch.no_grad()
    def __call__(self, y_u8, g_t):
        y = to_t(y_u8, self.dev)
        mse = float(((y - g_t) ** 2).mean())
        return (10 * np.log10(1.0 / max(mse, 1e-12)),
                float(repo_ssim(y, g_t)),
                float(self.lp(y * 2 - 1, g_t * 2 - 1).item()))


def jpeg_rt(a_u8, quality=100, subsampling=2):
    buf = io.BytesIO()
    Image.fromarray(a_u8).save(buf, "JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)

# ---------------------------------------------------------------- warping

INTERP = {"cubic": cv2.INTER_CUBIC, "lanczos4": cv2.INTER_LANCZOS4,
          "linear": cv2.INTER_LINEAR}


def make_maps(field, H, W, up=cv2.INTER_CUBIC):
    """field: (h,w,2) grid at some downsample, or (H,W,2) already full res."""
    fu = field if field.shape[:2] == (H, W) else cv2.resize(field, (W, H), interpolation=up)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return (xx + fu[..., 0]).astype(np.float32), (yy + fu[..., 1]).astype(np.float32)


def warp_u8(img_u8, mx, my, interp=cv2.INTER_LANCZOS4):
    out = cv2.remap(img_u8.astype(np.float32) / 255.0, mx, my, interp,
                    borderMode=cv2.BORDER_REFLECT)
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)

# ---------------------------------------------------------------- fields


def pool(stack, est):
    if est == "mean":
        return stack.mean(0)
    if est == "median":
        return np.median(stack, axis=0)
    if est.startswith("trim"):          # trimmed mean, e.g. trim20 -> drop 20% each tail
        f = int(est[4:]) / 100.0
        s = np.sort(stack, axis=0)
        k = int(round(len(stack) * f))
        return s[k:len(stack) - k].mean(0)
    raise ValueError(est)


def _basis(u, v, kind):
    """u,v normalised to [-1,1]; returns (N,K) design matrix."""
    one = np.ones_like(u)
    r2 = u * u + v * v
    if kind == "trans":
        return np.stack([one], 1)
    if kind == "affine":
        return np.stack([one, u, v], 1)
    if kind == "poly2":
        return np.stack([one, u, v, u * u, u * v, v * v], 1)
    if kind == "poly3":
        return np.stack([one, u, v, u * u, u * v, v * v,
                         u ** 3, u * u * v, u * v * v, v ** 3], 1)
    if kind == "poly4":
        cols = [one, u, v]
        for d in (2, 3, 4):
            cols += [u ** (d - k) * v ** k for k in range(d + 1)]
        return np.stack(cols, 1)
    raise ValueError(kind)


def fit_parametric(field_grid, kind, out_hw=None, weights=None):
    """LS-fit a low-order model to a pooled grid field; return it evaluated at out_hw."""
    h, w, _ = field_grid.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    u = (u + 0.5) / w * 2 - 1
    v = (v + 0.5) / h * 2 - 1
    uf, vf = u.ravel(), v.ravel()
    if kind == "brown":
        A = _brown(uf, vf)          # (N,2,K) coupled model
        K = A.shape[2]
        M = A.reshape(-1, K)
        y = field_grid.reshape(-1, 2).reshape(-1)
        c, *_ = np.linalg.lstsq(M, y, rcond=None)
        H, W = out_hw if out_hw else (h, w)
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        uu = (uu + 0.5) / W * 2 - 1
        vv = (vv + 0.5) / H * 2 - 1
        Ao = _brown(uu.ravel(), vv.ravel())
        return (Ao @ c).reshape(H, W, 2).astype(np.float32), c
    B = _basis(uf, vf, kind)
    cx, *_ = np.linalg.lstsq(B, field_grid[..., 0].ravel().astype(np.float64), rcond=None)
    cy, *_ = np.linalg.lstsq(B, field_grid[..., 1].ravel().astype(np.float64), rcond=None)
    H, W = out_hw if out_hw else (h, w)
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    uu = (uu + 0.5) / W * 2 - 1
    vv = (vv + 0.5) / H * 2 - 1
    Bo = _basis(uu.ravel(), vv.ravel(), kind)
    out = np.stack([(Bo @ cx).reshape(H, W), (Bo @ cy).reshape(H, W)], -1)
    return out.astype(np.float32), (cx, cy)


def _brown(u, v):
    """Brown-Conrady residual-lens model + affine.  returns (N,2,K)."""
    r2 = u * u + v * v
    one = np.ones_like(u); zero = np.zeros_like(u)
    # columns: t0,t1, a(u),b(v) shear/scale (4), k1,k2,k3 radial, p1,p2 tangential
    cols_x = [one, zero, u, v, zero, zero, u * r2, u * r2 * r2, u * r2 ** 3,
              2 * u * v, r2 + 2 * u * u]
    cols_y = [zero, one, zero, zero, u, v, v * r2, v * r2 * r2, v * r2 ** 3,
              r2 + 2 * v * v, 2 * u * v]
    return np.stack([np.stack(cols_x, 1), np.stack(cols_y, 1)], 1)


def upsample_lattice(small, H, W, win, stride, interp=cv2.INTER_CUBIC):
    """Upsample a field sampled on window CENTRES (y = i*stride + win/2) to full res.

    cv2.resize would stretch the lattice to the image corners, which is a real
    half-window geometric error at this scale, so map the coordinates explicitly.
    """
    h, w, _ = small.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gx = np.clip((xx - win / 2.0) / stride, 0, w - 1).astype(np.float32)
    gy = np.clip((yy - win / 2.0) / stride, 0, h - 1).astype(np.float32)
    return cv2.remap(small, gx, gy, interp, borderMode=cv2.BORDER_REPLICATE)


def load_stack(name, ds):
    z = np.load(os.path.join(FLOW, name + ".npz"))
    return z[f"ds{ds}"], list(z["stems"])
