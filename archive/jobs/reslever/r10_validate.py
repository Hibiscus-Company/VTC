"""R10: validate the USM candidate -- full 60 views, post-JPEG (production encode),
   2-fold CV on the parameter, and cross-scene transfer on 4 held-out towers."""
import sys, json, numpy as np, cv2, io
from PIL import Image
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
k4i = IDX["k4"]
CAND = [("base", None), ("usm0.5_0.20", (0.5, 0.20)), ("usm0.8_0.10", (0.8, 0.10)),
        ("usm0.8_0.05", (0.8, 0.05)), ("usm0.5_0.10", (0.5, 0.10))]


def usm(p, par):
    if par is None:
        return p
    s, a = par
    return np.clip(p + a * (p - cv2.GaussianBlur(p, (0, 0), s)), 0, 1)


def jp(p):
    b = io.BytesIO()
    Image.fromarray((np.clip(p, 0, 1) * 255).round().astype(np.uint8)).save(
        b, "JPEG", quality=100, subsampling=2)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"), np.float32) / 255.0, b.getbuffer().nbytes


print("== HCM0181  k4 + median field, all 60 views ==")
per = {}
for tag, par in CAND:
    for enc in ["png", "jpg"]:
        P = S = L = 0.0; sz = 0; pv = []
        for i in range(60):
            g = G[i].astype(np.float32) / 255.0
            p = usm(apply_field(R[k4i, i].astype(np.float32) / 255.0, field), par)
            if enc == "jpg":
                p, n = jp(p); sz += n
            a, b, c = score(p, g)
            pv.append(comp(a, b, c)); P += a; S += b; L += c
        P, S, L = P / 60, S / 60, L / 60
        per[(tag, enc)] = pv
        print(f" {tag:12s} {enc}  PSNR {P:7.4f} SSIM {S:.5f} LPIPS {L:.5f} comp {comp(P,S,L):8.4f}"
              + (f"  {sz/60/1e6:.2f} MB/img" if enc == "jpg" else ""))

bp = np.array(per[("base", "jpg")])
print("\n per-view paired delta vs base (jpg):")
for tag, par in CAND[1:]:
    d = np.array(per[(tag, "jpg")]) - bp
    print(f"  {tag:12s} mean {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  "
          f"se {d.std(ddof=1)/np.sqrt(60):.4f}  t={d.mean()/(d.std(ddof=1)/np.sqrt(60)):+.2f}"
          f"  win {int((d>0).sum())}/60")

# 2-fold CV: pick alpha on fold A, score on fold B and vice versa
print("\n== 2-fold CV over views (pick best candidate on the other fold) ==")
fa = list(range(0, 60, 2)); fb = list(range(1, 60, 2))
tot = 0.0
for tr, te in [(fa, fb), (fb, fa)]:
    best = max(CAND[1:], key=lambda c: np.mean(np.array(per[(c[0], "jpg")])[tr] - bp[tr]))
    d = np.mean(np.array(per[(best[0], "jpg")])[te] - bp[te])
    print(f"  train fold picks {best[0]:12s} -> held-out delta {d:+.4f}")
    tot += d / 2
print(f"  CV mean delta {tot:+.4f}")

# ---- cross-scene: single-member renders, no ensemble, no scene field ----
print("\n== cross-scene transfer (single member gsplatB9ut, PNG, every 2nd view) ==")
out = {}
for sc in SCENES:
    Rs, Gs, st = scene_stack(sc)
    sel = list(range(0, len(st), 2))
    line = f" {sc:9s}"
    base = None
    for tag, par in CAND:
        P = S = L = 0.0
        for i in sel:
            g = Gs[i].astype(np.float32) / 255.0
            p = usm(Rs[i].astype(np.float32) / 255.0, par)
            a, b, c = score(p, g)
            P += a; S += b; L += c
        n = len(sel); c_ = comp(P / n, S / n, L / n)
        if tag == "base":
            base = c_; line += f"  base {c_:7.4f}"
        else:
            line += f" | {tag} {c_-base:+.4f}"
    out[sc] = line
    print(line)
