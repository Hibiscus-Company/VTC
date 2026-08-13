"""R13: part of the HF deficit may be the field-warp resampler (INTER_CUBIC).
   Free alternative: sharper resampling.  Also USM before vs after the warp."""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
import lib
lib.DEV = "cuda:0"
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
k4i = IDX["k4"]
VIEWS = list(range(0, 60, 3))
H, W = 989, 1320
fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
MX, MY = (xx + fu[..., 0]).astype(np.float32), (yy + fu[..., 1]).astype(np.float32)


def warp(p, interp):
    return np.clip(cv2.remap(p, MX, MY, interp, borderMode=cv2.BORDER_REFLECT), 0, 1)


def usm(p, s, a):
    return np.clip(p + a * (p - cv2.GaussianBlur(p, (0, 0), s)), 0, 1)


cfgs = [
    ("cubic (shipped)", lambda p: warp(p, cv2.INTER_CUBIC)),
    ("lanczos4", lambda p: warp(p, cv2.INTER_LANCZOS4)),
    ("linear", lambda p: warp(p, cv2.INTER_LINEAR)),
    ("cubic+usm(.8,.10) after", lambda p: usm(warp(p, cv2.INTER_CUBIC), 0.8, 0.10)),
    ("cubic+usm(.8,.10) before", lambda p: warp(usm(p, 0.8, 0.10), cv2.INTER_CUBIC)),
    ("lanczos4+usm(.8,.10)", lambda p: usm(warp(p, cv2.INTER_LANCZOS4), 0.8, 0.10)),
    ("lanczos4+usm(.8,.05)", lambda p: usm(warp(p, cv2.INTER_LANCZOS4), 0.8, 0.05)),
    ("no warp+usm(.8,.10)", lambda p: usm(p, 0.8, 0.10)),
]
base = None
for tag, fn in cfgs:
    P = S = L = 0.0
    for i in VIEWS:
        g = G[i].astype(np.float32) / 255.0
        p = fn(R[k4i, i].astype(np.float32) / 255.0)
        a, b, c = score(p, g)
        P += a; S += b; L += c
    n = len(VIEWS); c_ = comp(P / n, S / n, L / n)
    if base is None:
        base = c_
    print(f" {tag:26s} PSNR {P/n:7.4f} SSIM {S/n:.5f} LPIPS {L/n:.5f} comp {c_:8.4f}  d {c_-base:+.4f}")
