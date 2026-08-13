"""ADVERSARIAL REPRO of the 'radial energy restoration at t=0.21' claim.
The claim interpolates a parabola through 3 points (t=0,0.5,1.0) and reads off a vertex.
Here we MEASURE the vertex region directly, per-scene, and under BOTH resamplers
(INTER_CUBIC = what LENS3.py used, INTER_LANCZOS4 = what actually ships).
"""
import os, sys, io, json
import numpy as np, cv2, torch
sys.path.insert(0, "/mnt/d/avv/metric_probe")
import mlib
import lpips as L
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(6)
dev = "cuda"
GTD = "/mnt/d/avv/data/phase1/public_set/{s}/test/images"
SCENES = [("HCM0181", "/mnt/d/avv/prodharness/k4/png"),
          ("HCM0193", "/mnt/d/avv/output/HCM0193_gsplatB9ut/test_poses_renders_png"),
          ("HCM0204", "/mnt/d/avv/output/HCM0204_gsplatB9ut/test_poses_renders_png"),
          ("hcm0031", "/mnt/d/avv/output/hcm0031_gsplatB9ut/test_poses_renders_png"),
          ("hcm0034", "/mnt/d/avv/output/hcm0034_gsplatB9ut/test_poses_renders_png")]
NIMG = 12
NRB = 8
SIG = [0.8, 1.6]
TS = [0.15, 0.21, 0.25, 0.30, 0.50, 1.00]
lp = L.LPIPS(net='vgg', verbose=False).to(dev).eval()
RESAMP = int(sys.argv[1]) if len(sys.argv) > 1 else cv2.INTER_CUBIC


def apply_field(img8, field, interp):
    H, W, _ = img8.shape
    fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.clip(cv2.remap(img8.astype(np.float32) / 255.0,
                             (xx + fu[..., 0]).astype(np.float32),
                             (yy + fu[..., 1]).astype(np.float32),
                             interp, borderMode=cv2.BORDER_REFLECT) * 255., 0, 255).astype(np.uint8)


def jpg(a):
    b = io.BytesIO()
    Image.fromarray(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)).save(
        b, "JPEG", quality=100, subsampling=2, optimize=True, progressive=True)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"), np.float32) / 255.


def bands(img):
    out, cur = [], img
    for s in SIG:
        lo = cv2.GaussianBlur(img, (0, 0), s)
        out.append(cur - lo); cur = lo
    return out, cur


def rmask(H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    r = r / r.max()
    return np.clip((r * NRB).astype(np.int32), 0, NRB - 1), r


def load(scene, rdir, st):
    FLD = np.load(f"/mnt/d/avv/fields/{scene}.npy")
    gd = GTD.format(s=scene)
    gmap = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(gd))}
    G8 = mlib.load_u8(os.path.join(gd, gmap[st]))
    R8 = apply_field(mlib.load_u8(os.path.join(rdir, st + ".png")), FLD, RESAMP)
    return G8.astype(np.float32) / 255., R8.astype(np.float32) / 255.


STEMS = {}
for scene, rdir in SCENES:
    gd = GTD.format(s=scene)
    gm = sorted(os.path.splitext(f)[0] for f in os.listdir(gd))
    s2 = [s for s in gm if os.path.exists(os.path.join(rdir, s + ".png"))]
    STEMS[scene] = s2[::max(1, len(s2) // NIMG)][:NIMG]

pw = {s: np.zeros((len(SIG), NRB, 2)) for s, _ in SCENES}
for scene, rdir in SCENES:
    for st in STEMS[scene]:
        Gf, Rf = load(scene, rdir, st)
        idx, _ = rmask(*Gf.shape[:2])
        bR, _ = bands(Rf); bG, _ = bands(Gf)
        for k in range(len(SIG)):
            a = bR[k].mean(2); b = bG[k].mean(2)
            for j in range(NRB):
                m = idx == j
                pw[scene][k, j, 0] += float((a[m] ** 2).sum())
                pw[scene][k, j, 1] += float((b[m] ** 2).sum())
    print("pass1", scene, flush=True)

ARMS = ["base"] + [f"t{t:.2f}" for t in TS]
acc = {a: np.zeros(3) for a in ARMS}
per = {s: {a: np.zeros(3) for a in ARMS} for s, _ in SCENES}
NT = 0
for scene, rdir in SCENES:
    g_en = np.stack([np.sqrt(pw[scene][k, :, 1] / pw[scene][k, :, 0]) for k in range(len(SIG))])
    for st in STEMS[scene]:
        Gf, Rf = load(scene, rdir, st)
        idx, rn = rmask(*Gf.shape[:2])
        bR, lo = bands(Rf)
        tg = torch.from_numpy(np.ascontiguousarray(Gf)).permute(2, 0, 1)[None].to(dev)
        V = {"base": Rf}
        for t in TS:
            gp = 1.0 + t * (g_en - 1.0)
            out = lo.copy()
            for k in range(len(SIG)):
                gm = np.interp(rn * NRB - 0.5, np.arange(NRB), gp[k]).astype(np.float32)
                out = out + bR[k] * gm[..., None]
            V[f"t{t:.2f}"] = np.clip(out, 0, 1)
        for a, arr in V.items():
            tt = torch.from_numpy(np.ascontiguousarray(jpg(arr))).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                m = np.array([mlib.psnr(tt, tg), float(mlib.ssim(tt, tg)),
                              float(lp(tt * 2 - 1, tg * 2 - 1).item())])
            acc[a] += m; per[scene][a] += m
        NT += 1
    print("pass2", scene, flush=True)

print(f"\n=== DIRECT VERTEX MEASUREMENT resamp={RESAMP} (4=LANCZOS4,2=CUBIC), {NT} imgs ===")
b = acc["base"] / NT
for a in ARMS:
    v = acc[a] / NT
    print(f"  {a:8s} PSNR {v[0]:8.4f} SSIM {v[1]:8.5f} LPIPS {v[2]:8.5f} "
          f"score {mlib.score(*v):9.4f}  dScore {mlib.score(*v)-mlib.score(*b):+9.4f}")
print("\n--- PER SCENE dScore ---")
print("  scene      " + " ".join(f"{a:>9s}" for a in ARMS[1:]))
for s, _ in SCENES:
    bb = per[s]["base"] / len(STEMS[s])
    row = [mlib.score(*(per[s][a] / len(STEMS[s]))) - mlib.score(*bb) for a in ARMS[1:]]
    print(f"  {s:9s}  " + " ".join(f"{x:+9.4f}" for x in row))
print("DONEB")
