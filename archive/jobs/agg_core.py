"""Aggregator zoo. X is a float32 array (k, H, W, 3) in [0,1]. All return (H,W,3)."""
import numpy as np


# ---------- helpers ----------
def _box1d(x, r, ax):
    n = x.shape[ax]
    pad = [(0, 0)] * x.ndim
    pad[ax] = (r, r)
    xp = np.pad(x, pad, mode="edge")
    c = np.cumsum(xp, axis=ax, dtype=np.float64)
    z = np.zeros_like(np.take(c, [0], axis=ax))
    c = np.concatenate([z, c], axis=ax)               # len n+2r+1
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax)
    lo = np.take(c, np.arange(0, n), axis=ax)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def _boxblur(a, r):
    if r <= 0:
        return a
    return _box1d(_box1d(a, r, 0), r, 1)


def _mad_scale(X, ref, r=4):
    """robust per-pixel scale = smoothed MAD across members, floored at 1 LSB."""
    mad = np.median(np.abs(X - ref[None]), axis=0)
    s = _boxblur(mad, r) * 1.4826
    return np.maximum(s, 1.0 / 255.0)


def _wmean(X, W):
    return (X * W).sum(0) / np.maximum(W.sum(0), 1e-8)


def _lum(X):
    return X[..., 0] * 0.299 + X[..., 1] * 0.587 + X[..., 2] * 0.114


# ---------- parameter-free ----------
def agg_mean(X):
    return X.mean(0)


def agg_median(X):
    return np.median(X, axis=0)


def agg_trim(X, t=1):
    S = np.sort(X, axis=0)
    return S[t:X.shape[0] - t].mean(0)


def agg_winsor(X, t=1):
    S = np.sort(X, axis=0).copy()
    k = X.shape[0]
    S[:t] = S[t]
    S[k - t:] = S[k - t - 1]
    return S.mean(0)


def agg_vecmed(X):
    """per-pixel vector median: member RGB triple with min sum of RGB distances to others."""
    k = X.shape[0]
    D = np.empty(X.shape[:3], dtype=np.float32)
    for i in range(k):
        d = np.zeros(X.shape[1:3], dtype=np.float32)
        for j in range(k):
            if i != j:
                d += np.sqrt(((X[i] - X[j]) ** 2).sum(-1))
        D[i] = d
    idx = np.argmin(D, axis=0)
    return np.take_along_axis(X, idx[None, ..., None].repeat(3, -1), 0)[0]


def agg_closest2med(X):
    med = np.median(X, axis=0)
    idx = np.argmin(np.abs(X - med[None]), axis=0)
    return np.take_along_axis(X, idx[None], 0)[0]


def agg_pmean(X, p):
    Xc = np.maximum(X, 1e-6)
    if abs(p) < 1e-9:
        return np.exp(np.log(Xc).mean(0))
    return np.power(np.power(Xc, p).mean(0), 1.0 / p)


def _s2l(x):
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0) + 0.055) / 1.055) ** 2.4)


def _l2s(x):
    x = np.maximum(x, 0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def agg_mean_linear(X):
    return _l2s(_s2l(X).mean(0))


# ---------- robust / weighted ----------
def _irls(X, psi, r=4, lum=False, iters=3):
    """IRLS with the robust scale estimated ONCE from the initial median (nuisance
    parameter; re-estimating it each sweep moved the output by <1e-4 and cost 3x)."""
    out = np.median(X, axis=0)
    if lum:
        Y = _lum(X)
        sy = np.maximum(_boxblur(np.median(np.abs(Y - _lum(out)[None]), axis=0), r) * 1.4826,
                        1.0 / 255.0)[None]
    else:
        s = _mad_scale(X, out, r)[None]
    for _ in range(iters):
        if lum:
            W = np.repeat(psi(np.abs(Y - _lum(out)[None]) / sy)[..., None], 3, -1).astype(np.float32)
        else:
            W = psi(np.abs(X - out[None]) / s).astype(np.float32)
        out = _wmean(X, W)
    return out


def agg_hybrid_freq(X, r=8):
    """low frequencies from the MEAN (which owns PSNR/SSIM), high frequencies from the
    MEDIAN (which owns LPIPS):  out = LP(mean) + HP(median) = median + LP(mean - median)."""
    m = X.mean(0)
    md = np.median(X, axis=0)
    return md + _boxblur(m - md, r)


def agg_huber(X, c=1.345, r=4, lum=False):
    return _irls(X, lambda rr: np.where(rr <= c, 1.0, c / np.maximum(rr, 1e-8)), r, lum)


def agg_tukey(X, c=4.685, r=4, lum=False):
    def psi(rr):
        u = np.minimum(rr / c, 1.0)
        return np.maximum((1 - u ** 2) ** 2, 1e-6)
    return _irls(X, psi, r, lum)


def agg_softmax(X, b=1.0, r=4):
    med = np.median(X, axis=0)
    s = _mad_scale(X, med, r)
    rr = (X - med[None]) / s[None]
    W = np.exp(-0.5 * (rr / b) ** 2).astype(np.float32)
    return _wmean(X, W)


def agg_invvar(X, r=8, eps=1e-4):
    """per-member local squared-deviation-from-consensus -> w = 1/(v+eps)."""
    med = np.median(X, axis=0)
    W = np.stack([(1.0 / (_boxblur((X[i] - med) ** 2, r) + eps)) for i in range(X.shape[0])],
                 0).astype(np.float32)
    return _wmean(X, W)


def agg_blend(X, a):
    """a*mean + (1-a)*median"""
    return a * X.mean(0) + (1 - a) * np.median(X, axis=0)


def agg_wex(X, t=1.0):
    """EXTRAPOLATED winsorisation: mean + t*(winsor1 - mean).
    t=0 -> mean, t=1 -> winsor1. winsor1 is the only aggregator that beat the mean
    at k=7, so this asks whether a LARGER dose of the same correction helps."""
    m = X.mean(0)
    return m + t * (agg_winsor(X, 1) - m)


def agg_lumguided(X, base="median"):
    """chroma from the plain mean, luminance from a robust estimator."""
    m = X.mean(0)
    rb = np.median(X, axis=0) if base == "median" else agg_trim(X, 1)
    return np.clip(m + (_lum(rb) - _lum(m))[..., None], 0, 1)


def agg_chromarobust(X):
    """luminance from the plain mean, chroma from the median (the opposite split)."""
    m = X.mean(0)
    md = np.median(X, axis=0)
    return np.clip(md + (_lum(m) - _lum(md))[..., None], 0, 1)


def agg_patchsel(X, ps=64):
    """per-patch, pick the single member closest (L2) to the per-patch median image."""
    k, H, W, _ = X.shape
    med = np.median(X, axis=0)
    err = ((X - med[None]) ** 2).sum(-1)
    Hp = -(-H // ps) * ps
    Wp = -(-W // ps) * ps
    E = np.zeros((k, Hp, Wp), dtype=np.float32)
    E[:, :H, :W] = err
    Eb = E.reshape(k, Hp // ps, ps, Wp // ps, ps).sum((2, 4))
    idx = np.argmin(Eb, axis=0)
    big = np.repeat(np.repeat(idx, ps, 0), ps, 1)[:H, :W]
    return np.take_along_axis(X, big[None, ..., None].repeat(3, -1), 0)[0]
