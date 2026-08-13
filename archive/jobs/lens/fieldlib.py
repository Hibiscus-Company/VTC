#!/usr/bin/env python
"""Field-model zoo + leave-one-view-out pooling, all reading one cached flow stack.

A "variant" is a (pooling, resolution, regularisation, interpolation) recipe that turns a
flow stack into a full-resolution warp.  The shipped production recipe is
    pool=median, ds=8, no smoothing, free-form grid, INTER_CUBIC upsample, INTER_CUBIC remap
which is the baseline every delta is quoted against.
"""
import numpy as np
import cv2

# ---------------------------------------------------------------- LOO pooling


class LooPool:
    """Leave-one-view-out mean and median of a stack, without an O(N^2) refit.

    median: sort the stack once along the view axis.  Dropping the element whose rank is r
    from n sorted values leaves n-1; the LOO median is then a pure index lookup into the
    sorted array, so every held-out view costs one np.where instead of a fresh np.median.
    Verified elementwise against np.median(np.delete(stack, i, 0), 0).
    """

    def __init__(self, stack):
        self.stack = stack.astype(np.float32)
        self.n = stack.shape[0]
        self.sum = self.stack.sum(axis=0)
        self._sorted = None
        self._rank = None

    def _prep_median(self):
        if self._sorted is not None:
            return
        order = np.argsort(self.stack, axis=0, kind="stable")
        self._sorted = np.take_along_axis(self.stack, order, axis=0)
        rank = np.empty(self.stack.shape, dtype=np.int16)
        vals = np.broadcast_to(
            np.arange(self.n, dtype=np.int16).reshape((-1,) + (1,) * (self.stack.ndim - 1)),
            self.stack.shape)
        np.put_along_axis(rank, order, vals, axis=0)
        self._rank = rank
        del order

    def mean(self, i=None):
        if i is None:
            return self.sum / self.n
        return (self.sum - self.stack[i]) / (self.n - 1)

    def median(self, i=None):
        if i is None:
            return np.median(self.stack, axis=0)
        self._prep_median()
        v, r = self._sorted, self._rank[i]
        m = self.n - 1

        def pick(j):
            return np.where(j < r, v[j], v[j + 1])

        if m % 2 == 1:
            return pick((m - 1) // 2)
        return 0.5 * (pick(m // 2 - 1) + pick(m // 2))

    def pooled(self, kind, i=None):
        return self.mean(i) if kind == "mean" else self.median(i)


# ---------------------------------------------------------------- field models


def gauss_smooth(field, sigma):
    if sigma <= 0:
        return field
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(field, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)


def poly_design(h, w, deg):
    """Vandermonde of a 2-D polynomial of total degree `deg` on normalised pixel centres."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx + 0.5) / w * 2.0 - 1.0
    v = (yy + 0.5) / h * 2.0 - 1.0
    cols = [np.ones_like(u)]
    for d in range(1, deg + 1):
        for a in range(d + 1):
            cols.append((u ** (d - a)) * (v ** a))
    return np.stack([c.ravel() for c in cols], axis=1).astype(np.float32)


class PolyModel:
    """Least-squares 2-D polynomial fitted to the pooled field, evaluated at full res.

    Evaluating the polynomial directly at full resolution also removes the grid-upsample
    interpolation entirely, so this is both a lower-dof and a smoother warp.
    """

    def __init__(self, deg, fh, fw, H, W):
        self.deg = deg
        A = poly_design(fh, fw, deg)
        self.pinv = np.linalg.pinv(A)              # K x (fh*fw)
        self.full = poly_design(H, W, deg)         # (H*W) x K
        self.fh, self.fw, self.H, self.W = fh, fw, H, W
        self.K = A.shape[1]

    def full_field(self, field):
        c = self.pinv @ field.reshape(-1, 2)       # K x 2
        return (self.full @ c).reshape(self.H, self.W, 2)


# ---------------------------------------------------------------- application

UP = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR,
      "lanczos": cv2.INTER_LANCZOS4, "area": cv2.INTER_AREA}


def upsample(field, H, W, how="cubic"):
    return cv2.resize(field, (W, H), interpolation=UP[how])


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def warp(img, fu, how="cubic"):
    """img float32 HxWx3 in [0,1], fu full-res displacement HxWx2."""
    if how.endswith("_lin"):
        # resample in LINEAR light instead of sRGB: sub-pixel interpolation of gamma-encoded
        # values is not photometrically correct, and this is the cheapest test of whether
        # that matters at our sub-pixel displacements.
        return lin_to_srgb(warp(srgb_to_lin(img), fu, how[:-4]))
    H, W, _ = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mx = (xx + fu[..., 0]).astype(np.float32)
    my = (yy + fu[..., 1]).astype(np.float32)
    if how in ("cubic", "linear", "lanczos"):
        fl = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR,
              "lanczos": cv2.INTER_LANCZOS4}[how]
        return cv2.remap(img, mx, my, fl, borderMode=cv2.BORDER_REFLECT)
    if how.startswith("spline"):
        # scipy's interpolating B-spline: no 1/32-px coordinate LUT, unlike cv2.remap,
        # and order 5 has wider support (sharper reconstruction) than order 3
        from scipy.ndimage import map_coordinates
        order = int(how[6:])
        out = np.empty_like(img)
        coords = np.stack([my, mx])
        for c in range(3):
            out[..., c] = map_coordinates(img[..., c], coords, order=order,
                                          mode="reflect", prefilter=True)
        return out
    if how.startswith("fft"):
        # band-limited: exact k-fold spectral upsample, then an ordinary kernel on the fine
        # grid where the signal is k-times oversampled. "fft2lan" = k=2, lanczos on the fine
        # grid. Lazy import: bandlimit imports this module.
        from bandlimit import warp_bandlimited
        k = int(how[3])
        fine = {"lan": "lanczos", "cub": "cubic", "lin": "linear"}[how[4:]]
        return warp_bandlimited(img, fu, k, fine)
    if how == "torch_bicubic":
        import torch
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        gx = torch.from_numpy((2 * mx + 1) / W - 1).unsqueeze(0).unsqueeze(-1)
        gy = torch.from_numpy((2 * my + 1) / H - 1).unsqueeze(0).unsqueeze(-1)
        grid = torch.cat([gx, gy], dim=-1)
        o = torch.nn.functional.grid_sample(t, grid, mode="bicubic",
                                            padding_mode="reflection", align_corners=False)
        return o.squeeze(0).permute(1, 2, 0).numpy()
    raise ValueError(how)
