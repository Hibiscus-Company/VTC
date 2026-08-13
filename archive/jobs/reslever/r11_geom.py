"""R11: is the residual a LOCAL SUB-PIXEL MISREGISTRATION (geometry) or photometric?
   Per block, regress r on (dGT/dx, dGT/dy, 1).  R^2 = share explained by a local shift."""
import sys, json, numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
k4i = IDX["k4"]
code_all = np.load(OUT + "/code_all.npy")
VIEWS = list(range(0, 60, 3))
B = 32


def blocks(x, b=B):
    h, w = x.shape[0] // b * b, x.shape[1] // b * b
    return x[:h, :w].reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(-1, b * b)


for LBL, blur in [("post-field k4", True)]:
    tot_r = tot_res_shift = tot_res_shiftgain = tot_res_gain = 0.0
    shifts = []
    per_region = {1: [0., 0.], 2: [0., 0.], 3: [0., 0.], 4: [0., 0.]}
    for i in VIEWS:
        g = G[i].astype(np.float32) / 255.0
        p = apply_field(R[k4i, i].astype(np.float32) / 255.0, field)
        y = (g * np.array([.299, .587, .114], np.float32)).sum(2)
        yp = (p * np.array([.299, .587, .114], np.float32)).sum(2)
        r = yp - y
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, 3) / 8.0
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, 3) / 8.0
        yc = y - cv2.GaussianBlur(y, (0, 0), 8.0)   # local AC part of GT (for a gain term)
        Rb, Xb, Yb, Cb, Db = blocks(r), blocks(gx), blocks(gy), blocks(np.ones_like(r)), blocks(yc)
        reg = blocks(code_all[i].astype(np.float32))
        for k in range(Rb.shape[0]):
            rr = Rb[k]
            A1 = np.stack([Cb[k]], 1)                       # DC only
            A2 = np.stack([Cb[k], Xb[k], Yb[k]], 1)          # DC + local shift
            A3 = np.stack([Cb[k], Xb[k], Yb[k], Db[k]], 1)   # + contrast gain
            v = (rr ** 2).sum()
            if v <= 0:
                continue
            tot_r += v
            for A, acc in [(A1, "dc"), (A2, "sh"), (A3, "sg")]:
                sol, *_ = np.linalg.lstsq(A, rr, rcond=None)
                res = ((rr - A @ sol) ** 2).sum()
                if acc == "dc":
                    tot_res_gain += res
                elif acc == "sh":
                    tot_res_shift += res
                    if np.abs(Xb[k]).mean() > 1e-3:
                        shifts.append((sol[1], sol[2]))
                else:
                    tot_res_shiftgain += res
            rid = int(round(reg[k].mean()))
            if abs(reg[k].mean() - rid) < 0.05 and rid in per_region:
                sol, *_ = np.linalg.lstsq(A2, rr, rcond=None)
                per_region[rid][0] += v
                per_region[rid][1] += ((rr - A2 @ sol) ** 2).sum()
    print(f"== {LBL}: residual variance explained per {B}x{B} block ==")
    print(f"  DC (local brightness offset)       : {100*(1-tot_res_gain/tot_r):5.2f}%")
    print(f"  DC + local sub-pixel shift         : {100*(1-tot_res_shift/tot_r):5.2f}%")
    print(f"  DC + shift + local contrast gain   : {100*(1-tot_res_shiftgain/tot_r):5.2f}%")
    print(f"  UNEXPLAINED (genuinely wrong content): {100*tot_res_shiftgain/tot_r:5.2f}%")
    s = np.array(shifts)
    print(f"  fitted local shift magnitude: median {np.median(np.abs(s)):.4f} px, "
          f"p90 {np.percentile(np.abs(s),90):.4f} px  (n={len(s)})")
    print("\n  by region (share of that region's residual explained by DC+shift):")
    for rid, nm in [(1, "sky"), (2, "flat_nonsky"), (3, "midtex"), (4, "hightex")]:
        v, res = per_region[rid]
        if v > 0:
            print(f"    {nm:12s} {100*(1-res/v):5.2f}%   (residual power share {100*v/tot_r:5.2f}%)")
