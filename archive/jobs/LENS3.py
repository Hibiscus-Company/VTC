"""LENS part 3: WHY is LPIPS error radial/corner-heavy, and is a radial MTF field legal+worth it?

Hypothesis: a real lens is sharper on axis than at the corners (field curvature / MTF falloff).
A 3D gaussian model cannot represent a per-camera 2D sharpness field: it fits ONE 3D scene to
photos whose corner-softness varies with where content lands, so it renders a compromise ->
corners too sharp / centre too soft RELATIVE TO GT. That is a PER-IMAGE, radially-parameterised
amplitude error, invisible to the published 'HF amplitude ratio 0.98-1.00' check (binned by
ensemble-variance decile, not radius).

PASS 1: per-radial-bin MSE-optimal gain g*(r) = <b_R.b_G>/<b_R.b_R> on the two finest bands.
PASS 2: score base / oracle-radial / LEAVE-ONE-SCENE-OUT-radial / global-scalar-gain,
        through the shipped q100 ss2 encode.
"""
import os, sys, io
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
lp = L.LPIPS(net='vgg', verbose=False).to(dev).eval()


def apply_field(img8, field):
    H, W, _ = img8.shape
    fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.clip(cv2.remap(img8.astype(np.float32) / 255.0,
                             (xx + fu[..., 0]).astype(np.float32),
                             (yy + fu[..., 1]).astype(np.float32),
                             cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT) * 255., 0, 255).astype(np.uint8)


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
    idx = np.clip((r * NRB).astype(np.int32), 0, NRB - 1)
    return idx, r


def load(scene, rdir, st):
    FLD = np.load(f"/mnt/d/avv/fields/{scene}.npy")
    gd = GTD.format(s=scene)
    gmap = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(gd))}
    G8 = mlib.load_u8(os.path.join(gd, gmap[st]))
    R8 = apply_field(mlib.load_u8(os.path.join(rdir, st + ".png")), FLD)
    return G8.astype(np.float32) / 255., R8.astype(np.float32) / 255.


STEMS = {}
for scene, rdir in SCENES:
    gd = GTD.format(s=scene)
    gm = sorted(os.path.splitext(f)[0] for f in os.listdir(gd))
    s2 = [s for s in gm if os.path.exists(os.path.join(rdir, s + ".png"))]
    STEMS[scene] = s2[::max(1, len(s2) // NIMG)][:NIMG]

# ---------- PASS 1 ----------
num = {s: np.zeros((len(SIG), NRB)) for s, _ in SCENES}
den = {s: np.zeros((len(SIG), NRB)) for s, _ in SCENES}
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
                num[scene][k, j] += float((a[m] * b[m]).sum())
                den[scene][k, j] += float((a[m] * a[m]).sum())
                pw[scene][k, j, 0] += float((a[m] ** 2).sum())
                pw[scene][k, j, 1] += float((b[m] ** 2).sum())
    print("pass1", scene, flush=True)

print("\n=== RADIAL BAND AMPLITUDE: render vs GT, by radius (0=centre,7=corner) ===")
print("  band0 (sigma<0.8px): rms ratio |R|/|G| then MSE-optimal gain g*")
for k in range(len(SIG)):
    print(f"  --- band{k} ---")
    for scene, _ in SCENES:
        rr = np.sqrt(pw[scene][k, :, 0] / pw[scene][k, :, 1])
        g = num[scene][k] / den[scene][k]
        print(f"   {scene:9s} |R|/|G| " + " ".join(f"{x:5.3f}" for x in rr)
              + "   g* " + " ".join(f"{x:5.3f}" for x in g))
    RR = np.mean([np.sqrt(pw[s][k, :, 0] / pw[s][k, :, 1]) for s, _ in SCENES], 0)
    GG = np.mean([num[s][k] / den[s][k] for s, _ in SCENES], 0)
    print(f"   {'POOLED':9s} |R|/|G| " + " ".join(f"{x:5.3f}" for x in RR)
          + "   g* " + " ".join(f"{x:5.3f}" for x in GG))

# ---------- PASS 2 ----------
ARMS = ["base", "oracle_rad", "loso_rad", "global_scalar", "oracle_energy", "oracle_energy_half"]
acc = {a: np.zeros(3) for a in ARMS}
NT = 0
for scene, rdir in SCENES:
    g_or = np.stack([num[scene][k] / den[scene][k] for k in range(len(SIG))])
    oth = [s for s, _ in SCENES if s != scene]
    g_lo = np.stack([sum(num[o][k] for o in oth) / sum(den[o][k] for o in oth)
                     for k in range(len(SIG))])
    g_gl = np.array([[float(sum(num[o][k].sum() for o in oth) /
                            sum(den[o][k].sum() for o in oth))] * NRB
                     for k in range(len(SIG))])
    # radial ENERGY-MATCH gain (opposite sign family): scale render band to GT band rms
    g_en = np.stack([np.sqrt(pw[scene][k, :, 1] / pw[scene][k, :, 0]) for k in range(len(SIG))])
    g_en_h = 1.0 + 0.5 * (g_en - 1.0)
    for st in STEMS[scene]:
        Gf, Rf = load(scene, rdir, st)
        idx, rn = rmask(*Gf.shape[:2])
        bR, lo = bands(Rf)
        tg = torch.from_numpy(np.ascontiguousarray(Gf)).permute(2, 0, 1)[None].to(dev)
        V = {"base": Rf}
        for nm, gp in [("oracle_rad", g_or), ("loso_rad", g_lo), ("global_scalar", g_gl),
                       ("oracle_energy", g_en), ("oracle_energy_half", g_en_h)]:
            out = lo.copy()
            for k in range(len(SIG)):
                # smooth radial gain map from the 8-bin profile
                gm = np.interp(rn * NRB - 0.5, np.arange(NRB), gp[k]).astype(np.float32)
                out = out + bR[k] * gm[..., None]
            V[nm] = np.clip(out, 0, 1)
        for a, arr in V.items():
            t = torch.from_numpy(np.ascontiguousarray(jpg(arr))).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                acc[a] += [mlib.psnr(t, tg), float(mlib.ssim(t, tg)),
                           float(lp(t * 2 - 1, tg * 2 - 1).item())]
        NT += 1
    print("pass2", scene, flush=True)

print("\n=== RADIAL MTF OPERATOR, through q100/ss2 encode, 5 towers x %d imgs ===" % NIMG)
b = acc["base"] / NT
for a in ARMS:
    v = acc[a] / NT
    print(f"  {a:14s} PSNR {v[0]:8.4f} SSIM {v[1]:8.5f} LPIPS {v[2]:8.5f} "
          f"score {mlib.score(*v):9.4f}  dScore {mlib.score(*v)-mlib.score(*b):+9.4f}")
print("DONE3")
