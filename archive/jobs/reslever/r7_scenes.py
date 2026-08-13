"""R7: (d) best vs worst public scene, and the GT-noise floor."""
import sys, json, numpy as np, cv2
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

field181 = np.load(TMP + "/HCM0181_median.npy")
rows = {}
for sc in SCENES:
    Rs, Gs, stems = scene_stack(sc)
    n = len(stems)
    P = S = L = 0.0
    conc = np.zeros(4); qs = [0.01, 0.05, 0.10, 0.25]
    gtHP = resHP = gtLP = resLP = 0.0
    flat_gtHP = flat_resHP = flat_px = 0.0
    lpu = 0.0   # LPIPS top-10% share
    bands = np.zeros(4); gbands = np.zeros(4)
    sel = list(range(0, n, 2))
    for i in sel:
        g = Gs[i].astype(np.float32) / 255.0
        p = Rs[i].astype(np.float32) / 255.0
        a, b, c = score(p, g)
        P += a; S += b; L += c
        se = ((p - g) ** 2).mean(2)
        v = np.sort(se.ravel())[::-1]; cc = np.cumsum(v) / v.sum()
        for j, q in enumerate(qs):
            conc[j] += cc[int(q * v.size) - 1]
        y = (g * np.array([.299, .587, .114], np.float32)).sum(2)
        yp = (p * np.array([.299, .587, .114], np.float32)).sum(2)
        r = yp - y
        # high-pass = signal minus gaussian sigma=1 (cut ~ >0.2 cyc/px)
        hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 1.0)
        gtHP += (hp(y) ** 2).mean(); resHP += (hp(r) ** 2).mean()
        gtLP += (y - y.mean()).var(); resLP += (r ** 2).mean()
        gm = cv2.GaussianBlur(np.abs(cv2.Sobel(cv2.GaussianBlur(y, (0, 0), 1.0), cv2.CV_32F, 1, 0, 3))
                              + np.abs(cv2.Sobel(cv2.GaussianBlur(y, (0, 0), 1.0), cv2.CV_32F, 0, 1, 3)),
                              (0, 0), 2.0)
        fl = gm < np.quantile(gm, 0.40)
        flat_gtHP += (hp(y)[fl] ** 2).sum(); flat_resHP += (hp(r)[fl] ** 2).sum()
        flat_px += fl.sum()
    m = len(sel)
    rows[sc] = dict(psnr=P / m, ssim=S / m, lpips=L / m, comp=comp(P / m, S / m, L / m),
                    conc=(conc / m).tolist(),
                    gtHP=float(gtHP / m), resHP=float(resHP / m),
                    gtvar=float(gtLP / m), resvar=float(resLP / m),
                    flat_gtHP=float(flat_gtHP / flat_px), flat_resHP=float(flat_resHP / flat_px))
    r = rows[sc]
    print(f"{sc:9s} PSNR {r['psnr']:7.4f} SSIM {r['ssim']:.5f} LPIPS {r['lpips']:.5f} "
          f"comp {r['comp']:8.4f} | SEtop1% {100*r['conc'][0]:5.1f}% top10% {100*r['conc'][2]:5.1f}%"
          f" | resid/GT var {r['resvar']/r['gtvar']:.4f}"
          f" | HPres/HPgt {r['resHP']/r['gtHP']:.4f}"
          f" | FLAT: HPgt {r['flat_gtHP']:.3e} HPres {r['flat_resHP']:.3e}"
          f" ratio {r['flat_resHP']/r['flat_gtHP']:.3f}")
json.dump(rows, open(OUT + "/r7.json", "w"))
