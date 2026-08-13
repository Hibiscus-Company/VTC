#!/usr/bin/env python
"""Round 2: (a)/(b)/(d) are dead (see pfz_fuse.py, n=3).  Only (c) -- spatially-varying lambda
driven by the GT-free r-map -- survived, so this run isolates it against the ONE control that
matters: is gamma<1 anything more than a larger GLOBAL lambda?

band0 = L0_mean * (1 + lam * (r-1)^gamma),   gamma=1 == production.

THREE POOLS:
  A  raw public k=8            4.62/255   (harness regime)
  B  A with member deviations shrunk x0.476 -> 2.30/255 (synthetic match to our private pool;
     NOTE the pixel mean is unchanged, so the mean is as over-smoothed as A's while the r-map
     says it is not -- this pool is BIASED IN FAVOUR of any operator that adds restoration, which
     is exactly why the lambda ladder is run inside it)
  C  real k=4 UT-family sub-pool  3.48/255 (genuinely homogeneous members, mean differs)
"""
import io, os, sys, json, time
import numpy as np, cv2, torch
from PIL import Image

TMP = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0, TMP)
sys.path.insert(0, os.path.join(TMP, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, upsample, warp
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
cv2.setNumThreads(8); Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)

SC = sys.argv[2] if len(sys.argv) > 2 else "HCM0181"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MEMN = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7", "gsplatB8pure",
        "e17visnorm", "e15ceil95", "e16app"]
MEM = [f"/mnt/d/avv/output/{SC}_{t}/test_poses_renders_png" for t in MEMN]
GTD = f"/mnt/d/avv/data/phase1/public_set/{SC}/test/images"
POOLS = {"A_pub": dict(idx=list(range(8)), alpha=1.0),
         "B_shrunk": dict(idx=list(range(8)), alpha=0.476),
         "C_ut4real": dict(idx=[0, 1, 2, 3], alpha=1.0)}
ARMS = [("base_lam1.0", 1.0, 1.0), ("lam1.25", 1.25, 1.0), ("lam1.5", 1.5, 1.0),
        ("lam2.0", 2.0, 1.0),
        ("gam0.85", 1.0, 0.85), ("gam0.7", 1.0, 0.7), ("gam0.5", 1.0, 0.5),
        ("gam0.7_lam0.8", 0.8, 0.7)]
dev = "cuda"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.).permute(2, 0, 1)


def enc(x):
    b = io.BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
    v = b.getvalue()
    return np.asarray(Image.open(io.BytesIO(v)).convert("RGB"), np.float32) / 255., len(v)


gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:N]
cache = np.load(f"{TMP}/lens/cache/pub_{SC}.npz")
lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
per = {f"{p}|{a}": [] for p in POOLS for a, *_ in ARMS}
nby = {k: 0 for k in per}
dis = {p: [] for p in POOLS}
print(f"{SC} n={len(stems)} arms={len(ARMS)} pools={list(POOLS)}", flush=True)
t0 = time.time()
for i, s in enumerate(stems):
    stack = torch.stack([load(os.path.join(d, s + ".png")) for d in MEM])
    laps, res, sizes = lap_pyr(stack, 5, _K)
    G = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"), np.float32) / 255.
    g = torch.from_numpy(np.ascontiguousarray(G)).permute(2, 0, 1).unsqueeze(0).to(dev)
    for pn, cfg in POOLS.items():
        ix, al = cfg["idx"], cfg["alpha"]
        kk_ = len(ix)
        lm = [L[ix].mean(0, keepdim=True) for L in laps]
        rm = res[ix].mean(0, keepdim=True)
        L0s = lm[0] + al * (laps[0][ix] - lm[0])
        sub = stack[ix]
        dis[pn].append(float((al * (sub - sub.mean(0, keepdim=True))).abs().mean()) * 255)
        Eb = boxf((lm[0] ** 2).sum(1, keepdim=True), 3)
        V = boxf(((L0s - lm[0]) ** 2).sum(1, keepdim=True), 3).mean(0, keepdim=True) * (kk_ / (kk_ - 1.0))
        r1 = (torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=4.0) - 1.0).clamp(min=0)
        for aname, lam, gam in ARMS:
            b0 = lm[0] * (1.0 + lam * (r1 ** gam if gam != 1.0 else r1))
            out = lap_recon([b0] + lm[1:], rm, sizes, _K).clamp(0, 1)
            x = np.clip(warp(np.ascontiguousarray(out[0].permute(1, 2, 0).numpy()), lens, "lanczos"), 0, 1)
            j, nb = enc(x)
            key = f"{pn}|{aname}"
            nby[key] += nb
            rr = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                per[key].append((10 * np.log10(1.0 / max(((rr - g) ** 2).mean().item(), 1e-12)),
                                 float(repo_ssim(rr, g)), float(vgg(rr * 2 - 1, g * 2 - 1).item())))
    if i % 5 == 0:
        print(f"  {i} {time.time()-t0:.0f}s", flush=True)

n = len(stems)
for pn in POOLS:
    print(f"\npool {pn}: disagreement {np.mean(dis[pn]):.3f}/255  k={len(POOLS[pn]['idx'])}")
    print(f"{'arm':>16} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'dSCORE':>8} {'t':>6} {'win':>7} {'MB':>6}")
    ref = np.array(per[f"{pn}|base_lam1.0"])
    refs = 100 * (0.4 * (1 - ref[:, 2]) + 0.3 * ref[:, 1] + 0.3 * np.minimum(ref[:, 0] / 50, 1))
    for aname, *_ in ARMS:
        A = np.array(per[f"{pn}|{aname}"])
        sc = 100 * (0.4 * (1 - A[:, 2]) + 0.3 * A[:, 1] + 0.3 * np.minimum(A[:, 0] / 50, 1))
        d = sc - refs
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n) + 1e-12) if n > 1 else 0.0
        print(f"{aname:>16} {sc.mean():9.4f} {A[:,0].mean():8.4f} {A[:,1].mean():7.4f} {A[:,2].mean():8.4f} "
              f"{d.mean():+8.4f} {t:6.2f} {int((d>0).sum()):3d}/{n} {nby[f'{pn}|{aname}']/1e6:6.1f}")
json.dump({k: dict(psnr=float(np.array(v)[:, 0].mean()), ssim=float(np.array(v)[:, 1].mean()),
                   lpips=float(np.array(v)[:, 2].mean()), mb=nby[k] / 1e6,
                   sc=[float(z) for z in (100 * (0.4 * (1 - np.array(v)[:, 2]) + 0.3 * np.array(v)[:, 1]
                       + 0.3 * np.minimum(np.array(v)[:, 0] / 50, 1)))]) for k, v in per.items()},
          open(f"{TMP}/pfz_fuse2_{SC}.json", "w"), indent=1)
