"""R14: production surrogate -- k-member pixel-mean + median lens field + JPEG q100 ss2,
   all 60 views.  Sharpness axis: resampler (cubic vs lanczos4) x USM alpha."""
import sys, json, numpy as np, cv2, io
from PIL import Image
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/reslever")
from lib import *

R, G = cache()
field = np.load(TMP + "/HCM0181_median.npy")
H, W = 989, 1320
fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
MX, MY = (xx + fu[..., 0]).astype(np.float32), (yy + fu[..., 1]).astype(np.float32)
POOL7 = ["m31b_nolpips", "sh3", "m31b_taillpips", "gsplatB10ut8M", "gsplatB12ut8Ms7",
         "e15ceil95", "gsplatB9ut"]
CUR4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
SIG = 0.8
hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), SIG)


def jp(p):
    b = io.BytesIO()
    Image.fromarray((np.clip(p, 0, 1) * 255).round().astype(np.uint8)).save(
        b, "JPEG", quality=100, subsampling=2)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"), np.float32) / 255.0, b.getbuffer().nbytes


for nm, pool in [("k7_pool", POOL7), ("k4_curated", CUR4)]:
    pid = [IDX[n] for n in pool]
    ens = {}
    Ds = []
    for i in range(60):
        ms = [R[j, i].astype(np.float32) / 255.0 for j in pid]
        m = np.clip(np.mean(ms, 0), 0, 1)
        Ds.append(np.mean([(hp(x) ** 2).mean() for x in ms]) / (hp(m) ** 2).mean() - 1)
        for it, tag in [(cv2.INTER_CUBIC, "cub"), (cv2.INTER_LANCZOS4, "lcz")]:
            ens[(i, tag)] = np.clip(cv2.remap(m, MX, MY, it, borderMode=cv2.BORDER_REFLECT), 0, 1)
    print(f"\n== {nm}: GT-free HF deficit D = {np.mean(Ds):.4f} ==")
    per = {}
    for tag in ["cub", "lcz"]:
        for a in [0.0, 0.05, 0.10, 0.15, 0.20]:
            P = S = L = 0.0; sz = 0; pv = []
            for i in range(60):
                p = ens[(i, tag)]
                q = np.clip(p + a * hp(p), 0, 1) if a else p
                q, n = jp(q); sz += n
                g = G[i].astype(np.float32) / 255.0
                x, y, z = score(q, g)
                pv.append(comp(x, y, z)); P += x; S += y; L += z
            key = f"{tag}_a{a:.2f}"
            per[key] = pv
            print(f" {key:10s} PSNR {P/60:7.4f} SSIM {S/60:.5f} LPIPS {L/60:.5f} "
                  f"comp {comp(P/60,S/60,L/60):8.4f}  {sz/60/1e6:.3f} MB/img")
    b0 = np.array(per["cub_a0.00"])
    print("  paired delta vs shipped (cubic, no usm):")
    for k in per:
        if k == "cub_a0.00":
            continue
        d = np.array(per[k]) - b0
        print(f"   {k:10s} mean {d.mean():+.4f} se {d.std(ddof=1)/np.sqrt(60):.4f} "
              f"t {d.mean()/(d.std(ddof=1)/np.sqrt(60)):+6.2f} win {int((d>0).sum())}/60")
    fa, fb = list(range(0, 60, 2)), list(range(1, 60, 2))
    tot = 0.0
    for tr, te in [(fa, fb), (fb, fa)]:
        best = max([k for k in per if k != "cub_a0.00"],
                   key=lambda k: np.mean(np.array(per[k])[tr] - b0[tr]))
        dd = np.mean(np.array(per[best])[te] - b0[te]); tot += dd / 2
        print(f"   CV fold picks {best:10s} -> held-out {dd:+.4f}")
    print(f"   CV mean delta {tot:+.4f}")
    json.dump({k: v for k, v in per.items()}, open(OUT + f"/r14_{nm}.json", "w"))
