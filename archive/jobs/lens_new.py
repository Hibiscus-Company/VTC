"""TWO UNTRIED PER-IMAGE OPERATORS ON THE LENS-FIELD WARP.

(1) WARP DEADBAND.  apply_field.py reads png_er, which is uint8, i.e. an EXACT INTEGER n per
    pixel, and writes (out*255+0.5).astype(uint8) = round(n + d) = n + round(d).  So every pixel
    whose intended geometric correction is |d| < 0.5 LSB is delivered as ZERO.  This is the same
    defect r33 fixed for energy_restore, but on the *field* -- and it is still live in r32.
    Fix = feed the float restore output straight into the warp (round once, at the end).
    Mechanism is NOT "add HF energy": the warp is ~energy preserving (lanczos keeps 97.6%);
    it delivers a sub-pixel GEOMETRIC alignment that the LB already priced at +0.7345.

(2) LINEAR-LIGHT RESAMPLING.  The lanczos remap runs on sRGB-encoded values.  A sub-pixel
    resample is an integral of scene radiance over a shifted footprint, which is linear in
    LIGHT, not in sRGB code value.  Resampling in sRGB systematically biases every edge
    (Jensen: the code-value mean of an edge is darker than the sRGB of the light mean).
    Fix = sRGB->linear, remap, linear->sRGB.  Adds nothing; it re-weights the same taps.
    NOTE: linear-light *ensemble averaging* was tested on 27/06 and LOST (-0.012).  That is a
    different operator (estimator choice under member noise); this one is a resampling kernel,
    the class with LB-confirmed ~1x transfer.

Both are PER-IMAGE.  Scored through the FULL SHIPPED CHAIN against REAL public test GT.
"""
import io, os, sys, time
import numpy as np, torch
from PIL import Image
import cv2

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from energy_restore import restore
from fieldlib import LooPool, gauss_smooth, upsample, warp
Image.MAX_IMAGE_PIXELS = None

GAIN = 1.30
SHIP = dict(quality=100, subsampling=2, optimize=True, progressive=True)
SCENES = {
    "HCM0181": ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"],
    "HCM0193": ["gsplatB9ut"],
    "HCM0204": ["gsplatB9ut"],
    "hcm0031": ["gsplatB9ut"],
    "hcm0034": ["gsplatB9ut"],
}
NMAX = int(os.environ.get("NMAX", "60"))

# ---- exact sRGB transfer pair -------------------------------------------------
def s2l(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)

def l2s(y):
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1 / 2.4) - 0.055).astype(np.float32)

_t = np.linspace(0, 1, 4096, dtype=np.float32)
assert np.abs(l2s(s2l(_t)) - _t).max() < 1e-5

from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
dev = "cuda" if torch.cuda.is_available() else "cpu"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()


def ldf(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def enc_score(x, g):
    b = io.BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIP)
    j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.float32) / 255.
    r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        P = 10 * np.log10(1. / max(((r - g) ** 2).mean().item(), 1e-12))
        S = float(repo_ssim(r, g)); L = float(vgg(r * 2 - 1, g * 2 - 1).item())
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.)), P, S, L, len(b.getvalue())


results = {}
for TAG, POOL in SCENES.items():
    D = lambda m: f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(D(m), s + ".png")) for m in POOL))[:NMAX]
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in cache["HW"]]
    base = LooPool(cache["s8"]).pooled("median")
    FU = upsample(gauss_smooth(base, 1), H, W, "cubic") * GAIN
    multi = len(POOL) > 1
    arms = ["A_prod", "C_lin"] + (["B_float", "D_both"] if multi else [])
    acc = {a: [0., 0., 0., 0] for a in arms}
    per = {a: [] for a in arms}
    dead = []          # fraction of intended-correction L2 destroyed by the pre-warp uint8
    t0 = time.time()
    for n, s in enumerate(stems):
        mem = [ldf(os.path.join(D(m), s + ".png")) for m in POOL]
        if multi:
            e = 0.8 * np.mean(mem[:3], axis=0) + 0.2 * mem[3]
            et = torch.from_numpy(e).permute(2, 0, 1).unsqueeze(0)
            mt = [torch.from_numpy(m).permute(2, 0, 1).unsqueeze(0) for m in mem]
            o = restore(et, mt, 1.0, len(mt)).clamp(0, 1)[0].permute(1, 2, 0).numpy()
        else:
            o = mem[0]
        o8 = np.ascontiguousarray((np.clip(o, 0, 1) * 255 + 0.5).astype(np.uint8).astype(np.float32) / 255.)
        g = torch.from_numpy(ldf(os.path.join(gtd, gt_by[s]))).permute(2, 0, 1).unsqueeze(0).to(dev)

        wA = np.clip(warp(o8, FU, "lanczos"), 0, 1)
        # how much of the warp's intended change survives the output rounding, given an
        # integer input?  d = intended change in LSB; delivered = round(n+d)-n = round(d).
        d = (wA - o8) * 255.0
        dead.append(1.0 - float((np.round(d) ** 2).sum() / max((d ** 2).sum(), 1e-9)))
        wC = l2s(np.clip(warp(s2l(o8), FU, "lanczos"), 0, 1))
        outs = {"A_prod": wA, "C_lin": wC}
        if multi:
            outs["B_float"] = np.clip(warp(np.clip(o, 0, 1).astype(np.float32), FU, "lanczos"), 0, 1)
            outs["D_both"] = l2s(np.clip(warp(s2l(np.clip(o, 0, 1).astype(np.float32)), FU, "lanczos"), 0, 1))
        for a in arms:
            sc, P, S, L, nb = enc_score(outs[a], g)
            acc[a][0] += P; acc[a][1] += S; acc[a][2] += L; acc[a][3] += nb
            per[a].append(sc)
        if n % 20 == 0:
            print(f"  {TAG} {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    N = len(stems)
    print(f"\n== {TAG}  n={N}  members={len(POOL)}  "
          f"mean|d| {np.abs(FU).mean():.4f}px  deadbanded fraction of warp L2 = {np.mean(dead):.3f}")
    print(f"{'arm':>9} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs A':>9} {'se':>7} {'wins':>8} {'MB':>7}")
    for a in arms:
        P, S, L, B = acc[a][0] / N, acc[a][1] / N, acc[a][2] / N, acc[a][3] / N
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.))
        dd = np.array(per[a]) - np.array(per["A_prod"])
        print(f"{a:>9} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {dd.mean():+9.4f} "
              f"{dd.std(ddof=1)/np.sqrt(N):7.4f} {int((dd>0).sum()):3d}/{N:<3d} {B/1e6:7.3f}")
        results.setdefault(a, {})[TAG] = (dd.mean(), dd.std(ddof=1) / np.sqrt(N), int((dd > 0).sum()), N)
    sys.stdout.flush()

print("\n===== CROSS-SCENE SUMMARY (delta vs A_prod, score points per scene) =====")
for a, per_s in results.items():
    if a == "A_prod":
        continue
    ds = [v[0] for v in per_s.values()]
    print(f"{a:>9}  " + "  ".join(f"{k} {v[0]:+.4f}({v[2]}/{v[3]})" for k, v in per_s.items())
          + f"   MEAN {np.mean(ds):+.4f}  scenes+ {sum(1 for x in ds if x>0)}/{len(ds)}")
