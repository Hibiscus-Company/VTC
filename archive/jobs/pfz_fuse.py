#!/usr/bin/env python
"""PERCEPTUAL ENSEMBLE FUSION probe.

Replace mean + global-scalar energy restore with spatially adaptive band-0 fusion rules.
Chain per arm (identical to production): k-member fuse -> band0 operator -> median lens field
lanczos4 -> JPEG q100/ss2/optimize/progressive -> PSNR/SSIM/LPIPS-vgg vs real test GT.

TRANSFER CONTROL: every rule here is POOL-DEPENDENT.  The public k=8 pool disagrees at 4.62/255,
our shipped private pool at 2.20/255.  We therefore run every arm TWICE on the same images:
  pool A  alpha=1.000  -> 4.62/255 (raw public pool)
  pool B  alpha=0.476  -> 2.20/255 (member deviations from the mean shrunk to match private)
alpha-shrink leaves the pixel MEAN bit-identical, so the two runs share a common reference image
and differ only in how much the members disagree -- a clean dial on the one variable that governs
transfer.
"""
import io, os, sys, json, time
import numpy as np, cv2, torch
from PIL import Image

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0, TMP)
sys.path.insert(0, os.path.join(TMP, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, fuse, _K
from fieldlib import LooPool, upsample, warp
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
cv2.setNumThreads(8); Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)

SC = "HCM0181"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MEMN = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7", "gsplatB8pure",
        "e17visnorm", "e15ceil95", "e16app"]
MEM = [f"/mnt/d/avv/output/{SC}_{t}/test_poses_renders_png" for t in MEMN]
GTD = f"/mnt/d/avv/data/phase1/public_set/{SC}/test/images"
K = len(MEM)
ALPHAS = {"A_pub4.62": 1.0, "B_matched2.20": 0.476}
dev = "cuda"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def enc(x):
    b = io.BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
    v = b.getvalue()
    return np.asarray(Image.open(io.BytesIO(v)).convert("RGB"), np.float32) / 255., len(v)


gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:N]
cache = np.load(f"{TMP}/lens/cache/pub_{SC}.npz")
lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")

# arm name -> (band0 rule, kw, lam, also_band1)
ARMS = [
    ("base_lam1.0", "mean", {}, 1.0, False),          # PRODUCTION OPERATOR = reference
    ("mean_lam0", "mean", {}, 0.0, False),
    ("a_pnorm2_lam0", "pnorm", dict(p=2.0), 0.0, False),
    ("a_pnorm2_lam1", "pnorm", dict(p=2.0), 1.0, False),
    ("a_pnorm2_b01_lam1", "pnorm", dict(p=2.0), 1.0, True),
    ("b_median_lam0", "median", {}, 0.0, False),
    ("b_median_lam1", "median", {}, 1.0, False),
    ("c_gam0.7_lam1", "mean", dict(gam=0.7), 1.0, False),
    ("c_gam1.5_lam1", "mean", dict(gam=1.5), 1.0, False),
    ("d_maxmag_lam0", "maxmag", {}, 0.0, False),
    ("d_maxmag_lam1", "maxmag", {}, 1.0, False),
]
keys = [f"{p}|{a}" for p in ALPHAS for a, *_ in ARMS]
per = {k: [] for k in keys}
nby = {k: 0 for k in keys}
dis = {p: [] for p in ALPHAS}
print(f"{SC} k={K} n={len(stems)} arms={len(ARMS)} pools={list(ALPHAS)}", flush=True)
t0 = time.time()
for i, s in enumerate(stems):
    stack = torch.stack([load(os.path.join(d, s + ".png")) for d in MEM])   # [k,3,H,W]
    laps, res, sizes = lap_pyr(stack, 5, _K)
    lm = [L.mean(0, keepdim=True) for L in laps]
    rm = res.mean(0, keepdim=True)
    G = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"), np.float32) / 255.
    g = torch.from_numpy(np.ascontiguousarray(G)).permute(2, 0, 1).unsqueeze(0).to(dev)
    for pn, al in ALPHAS.items():
        L0s = lm[0] + al * (laps[0] - lm[0])
        L1s = lm[1] + al * (laps[1] - lm[1])
        dis[pn].append(float((al * (stack - stack.mean(0, keepdim=True))).abs().mean()) * 255)
        Eb = boxf((lm[0] ** 2).sum(1, keepdim=True), 3)
        V = boxf(((L0s - lm[0]) ** 2).sum(1, keepdim=True), 3).mean(0, keepdim=True) * (K / (K - 1.0))
        r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=4.0)
        for aname, rule, kw, lam, b1 in ARMS:
            b0 = lm[0] if rule == "mean" else fuse(L0s, rule, **{k_: v for k_, v in kw.items() if k_ != "gam"})
            if lam > 0:
                gam = kw.get("gam", 1.0)
                b0 = b0 * (1.0 + lam * (r - 1.0).clamp(min=0) ** gam)
            bands = [b0] + ([fuse(L1s, rule, **{k_: v for k_, v in kw.items() if k_ != "gam"})] if b1 else [lm[1]]) + lm[2:]
            out = lap_recon(bands, rm, sizes, _K).clamp(0, 1)
            x = np.ascontiguousarray(out[0].permute(1, 2, 0).numpy())
            x = np.clip(warp(x, lens, "lanczos"), 0, 1)
            j, nb = enc(x)
            kk = f"{pn}|{aname}"
            nby[kk] += nb
            rr = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(((rr - g) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(rr, g))
                L = float(vgg(rr * 2 - 1, g * 2 - 1).item())
            per[kk].append((P, S, L))
    if i % 5 == 0:
        print(f"  {i} {time.time()-t0:.0f}s", flush=True)

n = len(stems)
for pn in ALPHAS:
    print(f"\npool {pn}: measured disagreement {np.mean(dis[pn]):.3f}/255")
    print(f"{'arm':>20} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'dSCORE':>8} {'t':>6} {'win':>6} {'MB':>6}")
    ref = np.array(per[f"{pn}|base_lam1.0"])
    refs = 100 * (0.4 * (1 - ref[:, 2]) + 0.3 * ref[:, 1] + 0.3 * np.minimum(ref[:, 0] / 50, 1))
    base = refs.mean()
    for aname, *_ in ARMS:
        kk = f"{pn}|{aname}"
        A = np.array(per[kk])
        sc = 100 * (0.4 * (1 - A[:, 2]) + 0.3 * A[:, 1] + 0.3 * np.minimum(A[:, 0] / 50, 1))
        d = sc - refs
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n) + 1e-12)
        print(f"{aname:>20} {sc.mean():9.4f} {A[:,0].mean():8.4f} {A[:,1].mean():7.4f} "
              f"{A[:,2].mean():8.4f} {sc.mean()-base:+8.4f} {t:6.2f} {int((d>0).sum()):3d}/{n} {nby[kk]/1e6:6.1f}")
json.dump({k: dict(psnr=float(np.array(v)[:, 0].mean()), ssim=float(np.array(v)[:, 1].mean()),
                   lpips=float(np.array(v)[:, 2].mean()), mb=nby[k] / 1e6) for k, v in per.items()},
          open(f"{TMP}/pfz_fuse_{SC}.json", "w"), indent=1)
