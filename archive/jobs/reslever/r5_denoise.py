"""R5: the diagnosis says we ADD high-frequency noise in regions where GT is smooth,
and those regions carry ~49% of LPIPS.  Test edge-preserving / flat-selective denoise."""
import sys, json, numpy as np, cv2, itertools
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
k4i = IDX["k4"]
H, W = 989, 1320
VIEWS = list(range(0, 60, 3))   # 20-view screen


def flatmask(p, q):
    y = (p * np.array([.299, .587, .114], np.float32)).sum(2)
    gx = cv2.Sobel(cv2.GaussianBlur(y, (0, 0), 1.0), cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(cv2.GaussianBlur(y, (0, 0), 1.0), cv2.CV_32F, 0, 1, 3)
    gm = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2.0)
    thr = np.quantile(gm, q)
    return np.clip((thr - gm) / max(thr, 1e-6), 0, 1)


def treat(p, kind, par):
    if kind == "none":
        return p
    if kind == "gauss_flat":
        sig, alpha, q = par
        w = flatmask(p, q)[..., None] * alpha
        return p * (1 - w) + cv2.GaussianBlur(p, (0, 0), sig) * w
    if kind == "bilat":
        sc, ss = par
        return cv2.bilateralFilter(p, 5, sc, ss)
    if kind == "nlm":
        h, = par
        u = (p * 255).clip(0, 255).astype(np.uint8)
        return cv2.fastNlMeansDenoisingColored(u, None, h, h, 5, 11).astype(np.float32) / 255.0
    if kind == "gauss_all":
        sig, alpha = par
        return p * (1 - alpha) + cv2.GaussianBlur(p, (0, 0), sig) * alpha
    raise ValueError(kind)


def run(kind, par, views=VIEWS, tag=""):
    P = S = L = 0.0
    for i in views:
        g = G[i].astype(np.float32) / 255.0
        p = apply_field(R[k4i, i].astype(np.float32) / 255.0, field)
        p = np.clip(treat(p, kind, par), 0, 1)
        a, b, c = score(p, g)
        P += a; S += b; L += c
    n = len(views)
    P, S, L = P / n, S / n, L / n
    return P, S, L, comp(P, S, L)


b = run("none", None)
print(f"baseline(20v)  PSNR {b[0]:7.4f} SSIM {b[1]:.5f} LPIPS {b[2]:.5f} comp {b[3]:8.4f}")
rows = []
grid = []
for sig in [0.5, 0.7, 1.0]:
    for al in [0.3, 0.6, 1.0]:
        for q in [0.4, 0.6]:
            grid.append(("gauss_flat", (sig, al, q)))
for sc in [0.01, 0.02, 0.04]:
    for ss in [2.0, 4.0]:
        grid.append(("bilat", (sc, ss)))
for h in [1.0, 2.0, 3.0]:
    grid.append(("nlm", (h,)))
for sig in [0.5, 0.7]:
    for al in [0.2, 0.4]:
        grid.append(("gauss_all", (sig, al)))

for kind, par in grid:
    r = run(kind, par)
    d = r[3] - b[3]
    rows.append(dict(kind=kind, par=list(par), psnr=r[0], ssim=r[1], lpips=r[2], comp=r[3], d=d))
    print(f" {kind:11s} {str(par):18s} dPSNR {r[0]-b[0]:+7.4f} dSSIM {r[1]-b[1]:+.5f}"
          f" dLPIPS {r[2]-b[2]:+.5f}  dCOMP {d:+.4f}")
json.dump(dict(base=b, rows=rows), open(OUT + "/r5.json", "w"))
