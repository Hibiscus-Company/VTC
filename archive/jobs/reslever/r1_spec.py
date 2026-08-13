"""R1: power-spectrum decomposition of the residual.

P = g(f)*GT + N,  N incoherent with GT.  R = P-GT = (g-1)GT + N.
  E|R|^2 = (1-g)^2 E|GT|^2 + E|N|^2   -> coherent deficit vs added wrong energy.
"""
import sys, json, numpy as np
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
H, W = 989, 1320
wy = np.hanning(H)[:, None].astype(np.float32)
wx = np.hanning(W)[None, :].astype(np.float32)
WIN = wy * wx

fy = np.fft.fftfreq(H)[:, None]
fx = np.fft.fftfreq(W)[None, :]
rad = np.sqrt(fy ** 2 + fx ** 2)
NB = 40
edges = np.linspace(0, 0.5, NB + 1)
bidx = np.clip(np.digitize(rad.ravel(), edges) - 1, 0, NB - 1)
cnt = np.bincount(bidx, minlength=NB).astype(np.float64)


def luma(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def acc(pred_fn, tag):
    Sgg = np.zeros(NB); Spp = np.zeros(NB); Srr = np.zeros(NB); Spg = np.zeros(NB)
    for i in range(60):
        g = luma(G[i].astype(np.float32) / 255.0)
        p = luma(pred_fn(i))
        Fg = np.fft.fft2((g - g.mean()) * WIN)
        Fp = np.fft.fft2((p - p.mean()) * WIN)
        Fr = Fp - Fg
        Sgg += np.bincount(bidx, weights=(np.abs(Fg) ** 2).ravel(), minlength=NB)
        Spp += np.bincount(bidx, weights=(np.abs(Fp) ** 2).ravel(), minlength=NB)
        Srr += np.bincount(bidx, weights=(np.abs(Fr) ** 2).ravel(), minlength=NB)
        Spg += np.bincount(bidx, weights=(Fp.real * Fg.real + Fp.imag * Fg.imag).ravel(), minlength=NB)
    g_f = Spg / Sgg
    coh = (1 - g_f) ** 2 * Sgg          # power lost by attenuation
    inc = np.maximum(Srr - coh, 0)      # power added that is not in GT
    out = dict(tag=tag, f=((edges[:-1] + edges[1:]) / 2).tolist(), cnt=cnt.tolist(),
               Sgg=Sgg.tolist(), Spp=Spp.tolist(), Srr=Srr.tolist(),
               gain=g_f.tolist(), coh=coh.tolist(), inc=inc.tolist())
    return out


k4i = IDX["k4"]
res = {}
res["k4_field"] = acc(lambda i: apply_field(R[k4i, i].astype(np.float32) / 255.0, field), "k4_field")
res["k4_raw"] = acc(lambda i: R[k4i, i].astype(np.float32) / 255.0, "k4_raw")
res["mem"] = acc(lambda i: R[IDX["gsplatB9ut"], i].astype(np.float32) / 255.0, "mem")
json.dump(res, open(OUT + "/r1.json", "w"))

for k in ["k4_field", "k4_raw", "mem"]:
    d = res[k]
    f = np.array(d["f"]); Sgg = np.array(d["Sgg"]); Srr = np.array(d["Srr"])
    Spp = np.array(d["Spp"]); gg = np.array(d["gain"])
    coh = np.array(d["coh"]); inc = np.array(d["inc"])
    print("\n=== " + k + " ===  total resid/gt power = %.4f" % (Srr.sum() / Sgg.sum()))
    print(" band(cyc/px)   GTpow%   RESpow%   gain    |R|^2/|G|^2   coh%ofR  inc%ofR")
    # octave-ish bands
    bnds = [(0, .0125), (.0125, .025), (.025, .05), (.05, .1), (.1, .2), (.2, .35), (.35, .5)]
    for lo, hi in bnds:
        m = (f >= lo) & (f < hi)
        if not m.any():
            continue
        gp = Sgg[m].sum(); rp = Srr[m].sum()
        gain = (np.array(d["Sgg"])[m] * gg[m]).sum() / gp
        print(f" {lo:.4f}-{hi:.4f}  {100*gp/Sgg.sum():7.3f}  {100*rp/Srr.sum():7.3f}"
              f"  {gain:6.3f}  {rp/gp:11.4f}  {100*coh[m].sum()/rp:7.2f}  {100*inc[m].sum()/rp:7.2f}")
