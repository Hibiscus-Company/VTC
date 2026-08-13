"""A3 (d): radially-averaged power spectra, render vs GT, averaged over eval holes.
Normalised radius: 1.0 == Nyquist along each axis independently (elliptical bands),
so scenes of different aspect/resolution are comparable. CPU only."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_common import *  # noqa
import numpy as np

RENDER, GT, OUT, TAG = sys.argv[1:5]
NB = 256
os.makedirs(OUT, exist_ok=True)
pairs = pair_list(RENDER, GT)
print(f"{TAG}: {len(pairs)} pairs", flush=True)


def radial(g, rbin, nb):
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2((g - g.mean()) * win))
    P = (F.real ** 2 + F.imag ** 2)
    s = np.bincount(rbin, weights=P.ravel(), minlength=nb)
    c = np.bincount(rbin, minlength=nb)
    return s, c


sumR = np.zeros(NB); sumG = np.zeros(NB); cnt = np.zeros(NB)
rbin = None
for i, (stem, rp, gp) in enumerate(pairs):
    r = gray(load(rp)); g = gray(load(gp))
    if rbin is None:
        H, W = g.shape
        fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None] / 0.5   # -1..1
        fx = np.fft.fftshift(np.fft.fftfreq(W))[None, :] / 0.5
        rad = np.sqrt(fy ** 2 + fx ** 2) / np.sqrt(2.0) * np.sqrt(2.0)  # keep euclid
        rad = np.sqrt(fy ** 2 + fx ** 2)
        rbin = np.clip((rad * NB).astype(np.int64), 0, NB - 1).ravel()
        cnt = np.bincount(rbin, minlength=NB).astype(np.float64)
    sR, _ = radial(r, rbin, NB)
    sG, _ = radial(g, rbin, NB)
    sumR += sR; sumG += sG
    print(f"  [{i+1}/{len(pairs)}] {stem}", flush=True)

n = len(pairs)
mR = sumR / np.maximum(cnt, 1) / n
mG = sumG / np.maximum(cnt, 1) / n
np.save(f"{OUT}/{TAG}_radial_render.npy", mR)
np.save(f"{OUT}/{TAG}_radial_gt.npy", mG)
np.save(f"{OUT}/{TAG}_radial_counts.npy", cnt)

# 5 bands from DC to Nyquist (radius 0..1); ignore corner region r>1
edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
rc = (np.arange(NB) + 0.5) / NB
bands = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (rc >= a) & (rc < b) & (cnt > 0)
    eR = float((sumR[m]).sum()); eG = float((sumG[m]).sum())
    bands.append(dict(lo=a, hi=b, energy_render=eR / n, energy_gt=eG / n,
                      ratio=eR / eG, ratio_db=10 * np.log10(eR / eG)))
    print(f"{TAG} band {a:.1f}-{b:.1f} Nyq: render/GT = {eR/eG:.4f} "
          f"({10*np.log10(eR/eG):+.2f} dB)", flush=True)
with open(f"{OUT}/{TAG}_bands.json", "w") as f:
    json.dump(dict(tag=TAG, n=n, edges=edges, bands=bands), f, indent=1)
