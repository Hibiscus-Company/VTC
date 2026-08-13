#!/usr/bin/env python
"""Is the pixel MEAN the right aggregator? Classical alternatives, same members.

  mean      pixel mean (shipped)
  median    per-pixel median (already known dead, included as a control)
  lapmax    Laplacian-pyramid fusion: mean the low band, MAX-|coef| the high bands
            (Burt & Adelson 1983 multi-resolution fusion)
  lapmag    mean the low band; in each high band keep the MEAN's phase/sign but
            restore the members' mean MAGNITUDE (undoes the displacement-induced
            cancellation without injecting a member's noise)
  lapmix    lapmag at half strength
"""
import os, argparse
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--levels", type=int, default=5)
a = ap.parse_args()

VAR = ["mean", "median", "lapmax", "lapmag", "lapmix"]
for v in VAR:
    os.makedirs(os.path.join(a.out, v), exist_ok=True)

stems = None
for d in a.dirs:
    s = {os.path.splitext(f)[0] for f in os.listdir(d)
         if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")}
    stems = s if stems is None else stems & s
stems = sorted(stems)


def pyr(img, L):
    g = [img]
    for _ in range(L):
        g.append(cv2.pyrDown(g[-1]))
    lap = []
    for i in range(L):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lap.append(g[i] - up)
    lap.append(g[L])
    return lap


def collapse(lap):
    out = lap[-1]
    for i in range(len(lap) - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(lap[i].shape[1], lap[i].shape[0])) + lap[i]
    return out


for s in stems:
    ms = []
    for d in a.dirs:
        for e in (".png", ".jpg", ".JPG"):
            p = os.path.join(d, s + e)
            if os.path.exists(p):
                ms.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)); break
    M = np.stack(ms)
    res = {"mean": M.mean(0), "median": np.median(M, 0)}
    P = [pyr(m, a.levels) for m in ms]
    for name in ("lapmax", "lapmag", "lapmix"):
        out = []
        for lvl in range(a.levels + 1):
            B = np.stack([p[lvl] for p in P])
            mb = B.mean(0)
            if lvl == a.levels:
                out.append(mb); continue
            if name == "lapmax":
                idx = np.abs(B).argmax(0)
                out.append(np.take_along_axis(B, idx[None], 0)[0])
            else:
                magm = np.abs(B).mean(0)                     # member magnitude
                magc = np.abs(mb)                            # magnitude after averaging
                g = magm / np.maximum(magc, 1e-3)
                g = np.minimum(g, 3.0)
                if name == "lapmix":
                    g = 1.0 + 0.5 * (g - 1.0)
                out.append(mb * g)
        res[name] = collapse(out)
    for k, v in res.items():
        Image.fromarray(np.clip(v + 0.5, 0, 255).astype(np.uint8)).save(
            os.path.join(a.out, k, s + ".png"))
print("wrote", len(stems), "x", len(VAR))
