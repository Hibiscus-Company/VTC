#!/usr/bin/env python
"""LEGAL per-frame blur (sharpness) predictor for a strided video capture (bonsai).

Predicts the sharpness statistic of a HELD-OUT frame from the sharpness of the TRAIN
frames only.  Nothing about the held-out frame's own pixels is ever used, so this is
usable on the real private test set (Competition Rule 10 clean).

    from blur_predictor import BlurPredictor, sharpness_stats, BONSAI_FIT

    obs  = {frame_int: sharpness_stats(path)['log_lapvar'] for train frames}
    bp   = BlurPredictor.fit(obs, mode='nw')      # LOO-selects the bandwidth on obs alone
    yhat = bp.predict(1590)                       # held-out frame index
    # a test frame whose neighbour is ALSO a test frame:
    yhat = bp.predict(1870, drop=(1880,))

Measured on the bonsai eval split (220 train_sub -> 28 held-out holes), 2026-07-30:

  target        estimator      R2 (both +-10 nb)   R2 (one nb)   R2 on real 18/10 mix
  log_lapvar    nw   h=13            0.657            0.560           0.639
  log_lapvar    nn2  (+-10 mean)     0.645            0.518            --
  reblur        ll   h=28            0.764            0.745            --
  log(GT/render lapvar) ratio        0.428            0.35            ~0.40
"""
import numpy as np

STRIDE = 10


# --------------------------------------------------------------------------- estimators
def _predict_one(fx, fy, x0, h, mode):
    d = fx.astype(np.float64) - x0
    if mode == "nn2":                                    # mean of the two +-STRIDE frames
        m = np.abs(d) <= STRIDE + 1e-9
        if not m.any():
            return float(fy[np.argmin(np.abs(d))])
        return float(fy[m].mean())
    w = np.exp(-0.5 * (d / h) ** 2)
    if w.sum() < 1e-8:
        return float(fy[np.argmin(np.abs(d))])
    if mode == "nw":                                     # Nadaraya-Watson kernel average
        return float((w * fy).sum() / w.sum())
    deg = {"ll": 1, "lq": 2}[mode]                       # local linear / quadratic
    X = np.vander(d, deg + 1, increasing=True)
    A = X.T @ (w[:, None] * X)
    A[np.diag_indices_from(A)] += 1e-9 * max(np.trace(A), 1e-12)
    try:
        return float(np.linalg.solve(A, X.T @ (w * fy))[0])
    except np.linalg.LinAlgError:
        return float((w * fy).sum() / w.sum())


class BlurPredictor:
    """Local kernel regression of a sharpness statistic over the frame-index axis."""

    def __init__(self, frames, values, h=13.0, mode="nw"):
        o = np.argsort(np.asarray(frames))
        self.fx = np.asarray(frames, dtype=np.int64)[o]
        self.fy = np.asarray(values, dtype=np.float64)[o]
        self.h, self.mode = float(h), mode

    def predict(self, frame, drop=()):
        """`drop` = extra frame indices to hide, for a test frame that is adjacent to
        another test frame (its +-STRIDE neighbour is not in the train set)."""
        hide = set(int(x) for x in drop) | {int(frame)}
        m = ~np.isin(self.fx, list(hide))
        return _predict_one(self.fx[m], self.fy[m], float(frame), self.h, self.mode)

    def predict_many(self, frames, drop_map=None):
        drop_map = drop_map or {}
        return np.array([self.predict(f, drop_map.get(int(f), ())) for f in frames])

    def loo(self, h=None, mode=None, drop_map=None):
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
    def fit(cls, obs, mode="nw", grid=None, drop_map=None, verbose=False):
        """Bandwidth by leave-one-out on the TRAIN observations alone."""
        if isinstance(obs, dict):
            frames = np.array(sorted(obs))
            values = np.array([obs[f] for f in frames], float)
        else:
            frames, values = map(np.asarray, obs)
        grid = grid if grid is not None else np.concatenate(
            [np.arange(6, 40, 1.0), np.arange(40, 121, 5.0)])
        self = cls(frames, values, h=float(grid[0]), mode=mode)
        best, bh = np.inf, float(grid[0])
        for h in grid:
            e = float(np.mean((self.loo(h=h, drop_map=drop_map) - self.fy) ** 2))
            if verbose:
                print(f"  h={h:6.1f} loo_mse={e:.5f}")
            if e < best:
                best, bh = e, float(h)
        self.h, self.loo_mse = bh, best
        self.loo_r2 = 1.0 - best / float(np.var(self.fy))
        return self

    def __repr__(self):
        return (f"BlurPredictor(mode={self.mode!r}, h={self.h:.1f}, n={len(self.fx)}, "
                f"loo_r2={getattr(self, 'loo_r2', float('nan')):.3f})")


# --------------------------------------------------------------- sharpness statistics
def sharpness_stats(path):
    """log(laplacian variance), log(HF spectral fraction) and the Crete re-blur drop."""
    import cv2
    cv2.setNumThreads(1)
    g = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    lv = float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var())
    # HF fraction on a windowed 1024^2 centre crop, above 0.25 Nyquist
    n = 1024
    c = g[(g.shape[0] - n) // 2:(g.shape[0] - n) // 2 + n,
          (g.shape[1] - n) // 2:(g.shape[1] - n) // 2 + n]
    w1 = np.hanning(n).astype(np.float32)
    c = (c - c.mean()) * np.outer(w1, w1)
    F = np.fft.rfft2(c)
    P = F.real ** 2 + F.imag ** 2
    r = np.sqrt((np.fft.fftfreq(n)[:, None] * 2) ** 2 + (np.fft.rfftfreq(n)[None, :] * 2) ** 2)
    hf = float(P[r > 0.25].sum() / max(P.sum(), 1e-20))
    # Crete re-blur drop (content-normalised)
    b = cv2.GaussianBlur(g, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
    ge = lambda x: float(np.abs(cv2.Sobel(x, cv2.CV_32F, 1, 0, 3)).sum()
                         + np.abs(cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)).sum())
    return dict(lapvar=lv, log_lapvar=float(np.log(lv)), hf025=hf,
                log_hf025=float(np.log(hf)), reblur=1.0 - ge(b) / max(ge(g), 1e-20))


# ----------------------------------------------------------------- fitted configuration
# LOO-selected on the 220 bonsai eval-split train_sub frames, 2026-07-30.
BONSAI_FIT = {"log_lapvar": dict(mode="nw", h=13.0),
              "log_hf025":  dict(mode="nw", h=15.0),
              "reblur":     dict(mode="ll", h=28.0)}
# Renders at held-out views compress the blur dynamic range:
#   render_log_lapvar = -2.711 + 0.556 * predicted_gt_log_lapvar   (R2 0.652)
# and are on average exp(0.964) = 2.62x less sharp than the photo.
RENDER_COMPRESSION = dict(intercept=-2.711, slope=0.556, mean_log_ratio=0.964)


if __name__ == "__main__":
    import os, sys, json, csv
    # self-test / re-derivation against the cached CSVs
    here = os.path.dirname(os.path.abspath(__file__))
    L = lambda p: {int(r["frame"]): {k: float(v) for k, v in r.items() if k != "name"}
                   for r in csv.DictReader(open(p))}
    T = L(f"{here}/train248_sharp.csv")
    EV = sorted(int(x) for x in
                json.load(open("/mnt/d/avv/evalsplit/bonsai/split.json"))["eval_frames"])
    SUB = [f for f in sorted(T) if f not in set(EV)]
    for st, cfg in BONSAI_FIT.items():
        bp = BlurPredictor.fit({f: T[f][st] for f in SUB}, mode=cfg["mode"])
        y = np.array([T[f][st] for f in EV])
        p = bp.predict_many(EV)
        p1 = bp.predict_many(EV, {f: (f + STRIDE,) for f in EV})
        r2 = lambda a, b: 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
        print(f"{st:11s} {bp}  R2(both)={r2(y,p):.3f}  R2(one nb)={r2(y,p1):.3f}")
