"""R15: cross-scene, held-out test of the RESAMPLER change (cubic -> lanczos4)
   for the lens-field warp.  Uses each scene's own cached train-fit flow stack."""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
import lib
lib.DEV = "cuda:0"
from lib import *

CACHE = TMP + "/lens/cache/pub_{s}.npz"
for sc in SCENES:
    d = np.load(CACHE.format(s=sc))
    fld = np.median(d["s8"].astype(np.float32), axis=0)      # shipped recipe: median, ds=8
    Rs, Gs, st = scene_stack(sc)
    H, W = Gs.shape[1:3]
    fu = cv2.resize(fld, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    MX, MY = (xx + fu[..., 0]).astype(np.float32), (yy + fu[..., 1]).astype(np.float32)
    sel = list(range(0, len(st), 2))
    out = {}
    for tag, it in [("nofield", None), ("cubic", cv2.INTER_CUBIC), ("lanczos4", cv2.INTER_LANCZOS4)]:
        P = S = L = 0.0
        for i in sel:
            p = Rs[i].astype(np.float32) / 255.0
            if it is not None:
                p = np.clip(cv2.remap(p, MX, MY, it, borderMode=cv2.BORDER_REFLECT), 0, 1)
            g = Gs[i].astype(np.float32) / 255.0
            a, b, c = score(p, g)
            P += a; S += b; L += c
        n = len(sel)
        out[tag] = comp(P / n, S / n, L / n)
    print(f" {sc:9s} nofield {out['nofield']:8.4f} | cubic {out['cubic']:8.4f} "
          f"({out['cubic']-out['nofield']:+.4f}) | lanczos4 {out['lanczos4']:8.4f} "
          f"({out['lanczos4']-out['nofield']:+.4f})  ->  lcz-cub {out['lanczos4']-out['cubic']:+.4f}")
