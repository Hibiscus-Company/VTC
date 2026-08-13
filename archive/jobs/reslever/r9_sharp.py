"""R9: spectrum says gain(f)=0.88/0.74/0.61 in the top 3 bands -> we are attenuated.
   LPIPS punishes missing HF hard (NLM h=2 on GT costs 0.053 LPIPS for 3e-5 MSE).
   Sweep unsharp masking on the composite (not MSE)."""
import sys, json, numpy as np, cv2, io
from PIL import Image
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
k4i = IDX["k4"]
VIEWS = list(range(0, 60, 3))


def usm(p, sig, a):
    return p + a * (p - cv2.GaussianBlur(p, (0, 0), sig))


def band(p, s1, s2, a):
    """boost only the band between sigma s1 and s2."""
    return p + a * (cv2.GaussianBlur(p, (0, 0), s1) - cv2.GaussianBlur(p, (0, 0), s2))


def jp(p, q=100, ss=2):
    b = io.BytesIO()
    Image.fromarray((np.clip(p, 0, 1) * 255).round().astype(np.uint8)).save(b, "JPEG", quality=q, subsampling=ss)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"), np.float32) / 255.0, b.getbuffer().nbytes


def run(fn, jpeg=False):
    P = S = L = 0.0; sz = 0
    for i in VIEWS:
        g = G[i].astype(np.float32) / 255.0
        p = np.clip(fn(apply_field(R[k4i, i].astype(np.float32) / 255.0, field)), 0, 1)
        if jpeg:
            p, n = jp(p); sz += n
        a, b, c = score(p, g)
        P += a; S += b; L += c
    n = len(VIEWS)
    return P / n, S / n, L / n, comp(P / n, S / n, L / n), sz / n


b = run(lambda p: p)
print(f"baseline PNG   PSNR {b[0]:7.4f} SSIM {b[1]:.5f} LPIPS {b[2]:.5f} comp {b[3]:8.4f}")
rows = []
for sig in [0.5, 0.8, 1.2]:
    for a in [0.05, 0.10, 0.20, 0.35]:
        r = run(lambda p, s=sig, aa=a: usm(p, s, aa))
        rows.append(("usm", sig, a, *r))
        print(f" usm  sig{sig:4.1f} a{a:5.2f}  dPSNR {r[0]-b[0]:+7.4f} dSSIM {r[1]-b[1]:+.5f}"
              f" dLPIPS {r[2]-b[2]:+.5f}  dCOMP {r[3]-b[3]:+.4f}")
for s1, s2 in [(0.6, 1.5), (1.0, 2.5), (1.5, 4.0)]:
    for a in [0.1, 0.25, 0.5]:
        r = run(lambda p, x=s1, y=s2, aa=a: band(p, x, y, aa))
        rows.append(("band", (s1, s2), a, *r))
        print(f" band {s1:4.1f}-{s2:4.1f} a{a:5.2f}  dPSNR {r[0]-b[0]:+7.4f} dSSIM {r[1]-b[1]:+.5f}"
              f" dLPIPS {r[2]-b[2]:+.5f}  dCOMP {r[3]-b[3]:+.4f}")
json.dump([[str(x) for x in r] for r in rows], open(OUT + "/r9.json", "w"))
