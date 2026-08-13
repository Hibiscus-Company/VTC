"""LENS part 2.
(A) Is the LPIPS border ramp a VGG-padding artifact or real content error?
    -> ring profile for SPATIALLY-UNIFORM control perturbations of GT itself.
(B) Is it rectangular (frame/resample) or radial (optics/coverage)?
(C) Exploit test for 'LPIPS is 87.5% luma': chroma-only smoothing, scored THROUGH the
    shipped encode (q100, subsampling=2, optimize, progressive).
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
lp = L.LPIPS(net='vgg', verbose=False).to(dev).eval()
mS = L.LPIPS(net='vgg', spatial=True, verbose=False).to(dev).eval()
rng = np.random.default_rng(0)


def apply_field(img8, field):
    H, W, _ = img8.shape
    fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.clip(cv2.remap(img8.astype(np.float32) / 255.0,
                             (xx + fu[..., 0]).astype(np.float32),
                             (yy + fu[..., 1]).astype(np.float32),
                             cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT) * 255., 0, 255).astype(np.uint8)


def jpg(arr01):
    b = io.BytesIO()
    Image.fromarray(np.clip(arr01 * 255 + 0.5, 0, 255).astype(np.uint8)).save(
        b, "JPEG", quality=100, subsampling=2, optimize=True, progressive=True)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"), np.float32) / 255.


@torch.no_grad()
def spatial_total(a, b):
    i0, i1 = mS.scaling_layer(a * 2 - 1), mS.scaling_layer(b * 2 - 1)
    f0, f1 = mS.net.forward(i0), mS.net.forward(i1)
    tot = 0
    for k in range(mS.L):
        d = (L.normalize_tensor(f0[k]) - L.normalize_tensor(f1[k])) ** 2
        v = torch.nn.functional.interpolate(mS.lins[k](d), size=a.shape[-2:],
                                            mode='bilinear', align_corners=False)
        tot = tot + v[0, 0]
    return tot.float().cpu().numpy()


def chroma_op(rgb01, fn):
    y = cv2.cvtColor(rgb01, cv2.COLOR_RGB2YCrCb)
    y[..., 1] = fn(y[..., 1]); y[..., 2] = fn(y[..., 2])
    return np.clip(cv2.cvtColor(y, cv2.COLOR_YCrCb2RGB), 0, 1)


def luma_op(rgb01, fn):
    y = cv2.cvtColor(rgb01, cv2.COLOR_RGB2YCrCb)
    y[..., 0] = fn(y[..., 0])
    return np.clip(cv2.cvtColor(y, cv2.COLOR_YCrCb2RGB), 0, 1)


ARMS = ["base", "chroma_g1", "chroma_g2", "chroma_g4", "chroma_med5", "luma_g1"]
acc = {a: np.zeros(3) for a in ARMS}
NR = 4
ctrl_names = ["render(base)", "ctrl:noise.02", "ctrl:blur0.7", "ctrl:shift0.3px"]
ring_s = np.zeros((len(ctrl_names), NR)); ring_n = np.zeros(NR)
rad_s = np.zeros((len(ctrl_names), NR)); rad_n = np.zeros(NR)
cor_s = np.zeros((len(ctrl_names), 2)); cor_n = np.zeros(2)
NTOT = 0
for scene, rdir in SCENES:
    FLD = np.load(f"/mnt/d/avv/fields/{scene}.npy")
    gd = GTD.format(s=scene)
    gmap = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(gd))}
    st = sorted(s for s in gmap if os.path.exists(os.path.join(rdir, s + ".png")))
    st = st[::max(1, len(st) // NIMG)][:NIMG]
    for s in st:
        G8 = mlib.load_u8(os.path.join(gd, gmap[s]))
        R8 = apply_field(mlib.load_u8(os.path.join(rdir, s + ".png")), FLD)
        Gf = G8.astype(np.float32) / 255.; Rf = R8.astype(np.float32) / 255.
        tg = torch.from_numpy(np.ascontiguousarray(Gf)).permute(2, 0, 1)[None].to(dev)
        V = {"base": Rf,
             "chroma_g1": chroma_op(Rf, lambda c: cv2.GaussianBlur(c, (0, 0), 1.0)),
             "chroma_g2": chroma_op(Rf, lambda c: cv2.GaussianBlur(c, (0, 0), 2.0)),
             "chroma_g4": chroma_op(Rf, lambda c: cv2.GaussianBlur(c, (0, 0), 4.0)),
             "chroma_med5": chroma_op(Rf, lambda c: cv2.medianBlur(
                 np.clip(c * 255 + 128, 0, 255).astype(np.uint8), 5).astype(np.float32) / 255. - 128 / 255.),
             "luma_g1": luma_op(Rf, lambda c: cv2.GaussianBlur(c, (0, 0), 1.0))}
        for a, arr in V.items():
            t = torch.from_numpy(np.ascontiguousarray(jpg(arr))).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                acc[a] += [mlib.psnr(t, tg), float(mlib.ssim(t, tg)),
                           float(lp(t * 2 - 1, tg * 2 - 1).item())]
        # --- spatial controls (pre-encode, float) ---
        H, W = Gf.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        pert = [Rf,
                np.clip(Gf + rng.normal(0, 0.02, Gf.shape).astype(np.float32), 0, 1),
                cv2.GaussianBlur(Gf, (0, 0), 0.7),
                cv2.remap(Gf, xx + 0.3, yy + 0.3, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)]
        d2b = np.minimum(np.minimum(yy, H - 1 - yy), np.minimum(xx, W - 1 - xx))
        RB = [8, 32, 96]
        rm = [d2b < RB[0], (d2b >= RB[0]) & (d2b < RB[1]),
              (d2b >= RB[1]) & (d2b < RB[2]), d2b >= RB[2]]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        rq = np.quantile(r, [.25, .5, .75])
        radm = [r < rq[0], (r >= rq[0]) & (r < rq[1]), (r >= rq[1]) & (r < rq[2]), r >= rq[2]]
        # corner vs edge-mid, both within outer 48px band
        band = d2b < 48
        nearx = np.minimum(xx, W - 1 - xx) < 48; neary = np.minimum(yy, H - 1 - yy) < 48
        corner = band & nearx & neary
        edgemid = band & ~corner
        cm = [corner, edgemid]
        for ci, p in enumerate(pert):
            tp = torch.from_numpy(np.ascontiguousarray(np.clip(p, 0, 1))).permute(2, 0, 1)[None].to(dev)
            M = spatial_total(tp, tg)
            for j in range(NR):
                ring_s[ci, j] += M[rm[j]].sum(); rad_s[ci, j] += M[radm[j]].sum()
            for j in range(2):
                cor_s[ci, j] += M[cm[j]].sum()
        for j in range(NR):
            ring_n[j] += rm[j].sum(); rad_n[j] += radm[j].sum()
        for j in range(2):
            cor_n[j] += cm[j].sum()
        NTOT += 1
    print("scene", scene, "done", flush=True)

print("\n=== (C) chroma exploit, 5 towers x %d imgs, THROUGH q100/ss2 encode ===" % NIMG)
n = NTOT
b = acc["base"] / n
print(f"  {'arm':12s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s} {'score':>9s} {'dScore':>9s}")
for a in ARMS:
    v = acc[a] / n
    print(f"  {a:12s} {v[0]:8.4f} {v[1]:8.5f} {v[2]:8.5f} {mlib.score(*v):9.4f} "
          f"{mlib.score(*v)-mlib.score(*b):+9.4f}   dL {100*(v[2]-b[2]):+.4f}pp")

print("\n=== (A)+(B) spatial profile: is the border ramp real or a metric artifact? ===")
print(f"  {'pair':16s} {'<8':>8s} {'8-32':>8s} {'32-96':>8s} {'>=96':>8s}   (density, mean=1)")
for ci, nm in enumerate(ctrl_names):
    d = ring_s[ci] / ring_n / (ring_s[ci].sum() / ring_n.sum())
    print(f"  {nm:16s} " + " ".join(f"{x:8.3f}" for x in d))
print(f"\n  radial quartiles (r from image centre):")
print(f"  {'pair':16s} {'q1(in)':>8s} {'q2':>8s} {'q3':>8s} {'q4(out)':>8s}")
for ci, nm in enumerate(ctrl_names):
    d = rad_s[ci] / rad_n / (rad_s[ci].sum() / rad_n.sum())
    print(f"  {nm:16s} " + " ".join(f"{x:8.3f}" for x in d))
print(f"\n  within the outer 48px band: corner vs edge-midpoint density")
for ci, nm in enumerate(ctrl_names):
    d = cor_s[ci] / cor_n
    print(f"  {nm:16s} corner {d[0]:.5f}  edge-mid {d[1]:.5f}  ratio {d[0]/d[1]:.3f}")
print("DONE2")
