#!/usr/bin/env python
"""Legal per-frame blur (sharpness) predictor for the bonsai scene.

Predicts the sharpness statistic of a HELD-OUT frame using ONLY the sharpness of the
TRAIN frames.  No test/eval ground truth is ever read at fit time.

Usage
-----
    from blur_predictor import BlurPredictor, sharpness_stats

    obs = {frame_int: log_sharpness_value, ...}      # train frames only
    bp  = BlurPredictor.fit(obs)                     # LOO-selects bandwidth on obs alone
    yhat = bp.predict(1590)                          # held-out frame index

The default estimator is a Gaussian-kernel local-LINEAR regression (Nadaraya-Watson
degenerates at the sequence edges); `mode='nw'` gives the plain kernel average and
`mode='nn2'` gives the mean of the two +-stride neighbours.
"""
import numpy as np

STRIDE = 10


# --------------------------------------------------------------------------- estimators
def _kern(d, h):
    return np.exp(-0.5 * (d / h) ** 2)


def _predict_one(fx, fy, x0, h, mode, min_w=1e-8):
    """fx,fy: observed frame indices / values (x0 already excluded).  Returns scalar."""
    d = fx.astype(np.float64) - x0
    if mode == "nn2":
        m = np.abs(d) <= STRIDE + 1e-9
        if m.sum() == 0:                       # fall back to the nearest observation
            j = np.argmin(np.abs(d))
            return float(fy[j])
        return float(fy[m].mean())
    if mode == "nnk":                          # mean of the k nearest, k=h
        k = int(round(h))
        idx = np.argsort(np.abs(d))[:k]
        return float(fy[idx].mean())
    w = _kern(d, h)
    if w.sum() < min_w:
        j = np.argmin(np.abs(d))
        return float(fy[j])
    if mode == "nw":
        return float((w * fy).sum() / w.sum())
    deg = {"ll": 1, "lq": 2}[mode]
    X = np.vander(d, deg + 1, increasing=True)          # [1, d, d^2]
    W = w
    A = X.T @ (W[:, None] * X)
    b = X.T @ (W * fy)
    A[np.diag_indices_from(A)] += 1e-9 * max(np.trace(A), 1e-12)
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return float((w * fy).sum() / w.sum())
    return float(c[0])                                   # value of the fit at d = 0


class BlurPredictor:
    """Local-regression sharpness predictor over the frame-index axis."""

    def __init__(self, frames, values, h=30.0, mode="ll"):
        o = np.argsort(np.asarray(frames))
        self.fx = np.asarray(frames, dtype=np.int64)[o]
        self.fy = np.asarray(values, dtype=np.float64)[o]
        self.h = float(h)
        self.mode = mode

    # ------------------------------------------------------------------ prediction
    def predict(self, frame, drop=()):
        """Predict at `frame`.  `drop` = extra frame indices to hide (simulates a
        neighbouring frame also being part of the held-out test set)."""
        hide = set(drop) | {int(frame)}
        m = ~np.isin(self.fx, list(hide))
        return _predict_one(self.fx[m], self.fy[m], float(frame), self.h, self.mode)

    def predict_many(self, frames, drop_map=None):
        drop_map = drop_map or {}
        return np.array([self.predict(f, drop_map.get(int(f), ())) for f in frames])

    # ------------------------------------------------------------------ LOO fitting
    def loo(self, h=None, mode=None, drop_map=None):
        """Leave-one-out predictions on the TRAINING observations themselves."""
        h = self.h if h is None else h
        mode = self.mode if mode is None else mode
        drop_map = drop_map or {}
        out = np.empty(len(self.fx))
        for i, x0 in enumerate(self.fx):
            hide = set(drop_map.get(int(x0), ())) | {int(x0)}
            m = ~np.isin(self.fx, list(hide))
            out[i] = _predict_one(self.fx[m], self.fy[m], float(x0), h, mode)
        return out

    @classmethod
    def fit(cls, obs, mode="ll", grid=None, drop_map=None, verbose=False):
        """obs: dict {frame:value} or (frames, values).  Bandwidth by LOO on obs ALONE."""
        if isinstance(obs, dict):
            frames = np.array(sorted(obs))
            values = np.array([obs[f] for f in frames], dtype=np.float64)
        else:
            frames, values = map(np.asarray, obs)
        grid = grid if grid is not None else np.concatenate(
            [np.arange(6, 40, 1.0), np.arange(40, 121, 5.0)])
        self = cls(frames, values, h=grid[0], mode=mode)
        best, bh = np.inf, grid[0]
        for h in grid:
            p = self.loo(h=h, drop_map=drop_map)
            e = float(np.mean((p - self.fy) ** 2))
            if verbose:
                print(f"  h={h:6.1f}  loo_mse={e:.5f}")
            if e < best:
                best, bh = e, h
        self.h = float(bh)
        self.loo_mse = best
        self.loo_r2 = 1.0 - best / float(np.var(self.fy))
        return self

    def __repr__(self):
        return (f"BlurPredictor(mode={self.mode!r}, h={self.h:.1f}, n={len(self.fx)}, "
                f"loo_r2={getattr(self, 'loo_r2', float('nan')):.3f})")


# --------------------------------------------------------------- sharpness statistics
def sharpness_stats(path):
    """log(laplacian variance) and the Crete re-blur drop of one image.  cv2 required."""
    import cv2
    cv2.setNumThreads(1)
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lp = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    b = cv2.GaussianBlur(g, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
    def ge(x):
        return float(np.abs(cv2.Sobel(x, cv2.CV_32F, 1, 0, 3)).sum()
                     + np.abs(cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)).sum())
    e0 = ge(g)
    return dict(log_lapvar=float(np.log(lp.var())), reblur=1.0 - ge(b) / max(e0, 1e-20))


# --------------------------------------------------------------------- fitted constants
# Fitted 2026-07-30 on the 220 bonsai eval-split train_sub frames (see A2 report).
BONSAI_FIT = dict(target="log_lapvar", mode="ll", h=14.0)
