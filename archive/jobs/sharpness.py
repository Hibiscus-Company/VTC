#!/usr/bin/env python
"""Sharpness statistics for bonsai frames.  CPU only.

Three statistics per image (grayscale, float [0,1]):
  lapvar   : variance of the 3x3 Laplacian over the whole frame  (raw, content-confounded)
  hf025    : fraction of windowed-FFT power above 0.25 Nyquist, on a 1024x1024 centre crop
  reblur   : Crete-style re-blur drop, 1 - sum|grad(G_1.0 * I)| / sum|grad(I)|.
             Content-normalised: a already-blurry image loses little gradient energy when
             blurred again, so reblur is small.  Higher = sharper.
"""
import os, sys, json
import numpy as np
import cv2

cv2.setNumThreads(1)


def gray(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(path)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return g


def _centre_crop(g, n=1024):
    h, w = g.shape
    y0 = (h - n) // 2
    x0 = (w - n) // 2
    return g[y0:y0 + n, x0:x0 + n]


_WIN = None


def hf_frac(g, n=1024, cut=0.25):
    global _WIN
    c = _centre_crop(g, n)
    if _WIN is None or _WIN.shape[0] != n:
        w1 = np.hanning(n).astype(np.float32)
        _WIN = np.outer(w1, w1)
    c = (c - c.mean()) * _WIN
    F = np.fft.rfft2(c)
    P = (F.real ** 2 + F.imag ** 2)
    fy = np.fft.fftfreq(n)[:, None] * 2.0          # in units of Nyquist
    fx = np.fft.rfftfreq(n)[None, :] * 2.0
    r = np.sqrt(fy ** 2 + fx ** 2)
    tot = P.sum()
    return float(P[r > cut].sum() / max(tot, 1e-20))


def reblur(g, sigma=1.0):
    b = cv2.GaussianBlur(g, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    def ge(x):
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.abs(gx).sum() + np.abs(gy).sum())
    e0 = ge(g)
    e1 = ge(b)
    return 1.0 - e1 / max(e0, 1e-20)


def lapvar(g):
    lp = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(lp.var())


def stats(path):
    g = gray(path)
    return dict(lapvar=lapvar(g), hf025=hf_frac(g), reblur=reblur(g))


def _job(args):
    name, path = args
    s = stats(path)
    s["name"] = name
    return s


def run_dir(d, out_csv, workers=6):
    from multiprocessing import Pool
    files = sorted(f for f in os.listdir(d)
                   if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"))
    jobs = [(os.path.splitext(f)[0], os.path.join(d, f)) for f in files]
    with Pool(workers) as p:
        rows = p.map(_job, jobs, chunksize=2)
    rows.sort(key=lambda r: r["name"])
    with open(out_csv, "w") as fh:
        fh.write("name,frame,lapvar,log_lapvar,hf025,log_hf025,reblur\n")
        for r in rows:
            fr = int(r["name"].split("_")[-1])
            fh.write(f'{r["name"]},{fr},{r["lapvar"]:.8e},{np.log(r["lapvar"]):.6f},'
                     f'{r["hf025"]:.8e},{np.log(r["hf025"]):.6f},{r["reblur"]:.6f}\n')
    print("wrote", out_csv, len(rows))


if __name__ == "__main__":
    run_dir(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
