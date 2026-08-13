#!/usr/bin/env python
"""ORACLE BOUND: where does the score actually live, by geography x frequency?

Full shipped chain (4-member mean -> energy restore lam=1.0 -> median lens field lanczos4),
then an ORACLE edit is applied just before the JPEG encode, then JPEG q100/ss2 and score.
Masks are derived from OUR OWN output (test-time computable), never from GT.

Arms bound what any operator of that class could ever win.
"""
import io, os, sys, time
import numpy as np
import cv2
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, _K
from energy_restore import restore
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SIG = 16.0


def loadt(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def enc(x):
    b = io.BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.float32) / 255.0


def lp(a, s=SIG):
    return np.stack([cv2.GaussianBlur(a[..., c], (0, 0), s) for c in range(3)], -1)


gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
stems = stems[::max(1, len(stems) // N)][:N]
cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")

dev = "cuda:0"
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

ARMS = ["base", "lf_flat", "hf_flat", "flat_all"]
acc = {a: [0.0, 0.0, 0.0] for a in ARMS}
t0 = time.time()
for si, s in enumerate(stems):
    mem = [loadt(os.path.join(d, s + ".png")) for d in MEM]
    ens = torch.stack(mem).mean(0)
    out = restore(ens, mem, 1.0, len(mem), 3).clamp(0, 1)
    X = np.clip(warp(out[0].permute(1, 2, 0).numpy(), lens, "lanczos"), 0, 1)
    G = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"), dtype=np.float32) / 255.0

    # masks from OUR OWN image
    xg = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(xg, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(xg, cv2.CV_32F, 0, 1, 3)
    gm = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 3.0)
    t_lo, t_hi = np.percentile(gm, [35.0, 73.0])
    mflat = cv2.GaussianBlur((gm <= t_lo).astype(np.float32), (0, 0), 4.0)[..., None]
    medge = cv2.GaussianBlur((gm >= t_hi).astype(np.float32), (0, 0), 4.0)[..., None]

    Xlf, Glf = lp(X), lp(G)
    outs = {
        "base": X,
        "lf_flat": X + (Glf - Xlf) * mflat,
        "hf_flat": X + ((G - Glf) - (X - Xlf)) * mflat,
        "flat_all": X + (G - X) * mflat,
    }
    gt_t = torch.from_numpy(np.ascontiguousarray(G)).permute(2, 0, 1).unsqueeze(0).to(dev)
    for a in ARMS:
        j = enc(outs[a])
        r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            acc[a][0] += 10 * np.log10(1.0 / max(((r - gt_t) ** 2).mean().item(), 1e-12))
            acc[a][1] += float(repo_ssim(r, gt_t))
            acc[a][2] += float(vgg(r * 2 - 1, gt_t * 2 - 1).item())
        del r
    del gt_t
    torch.cuda.empty_cache()
    m = si + 1
    line = f"  {m}/{len(stems)} {time.time()-t0:.0f}s |"
    b0 = None
    for a in ARMS:
        P, S, L = (v / m for v in acc[a])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        b0 = sc if a == "base" else b0
        line += f" {a}={sc-b0:+.4f}" if a != "base" else f" base={sc:.4f}"
    print(line, flush=True)

n = len(stems)
print(f"\n=== ORACLE BOUNDS, HCM0181, n={n}, full shipped chain + JPEG ===")
print(f"{'arm':>10} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs base':>9}")
base = None
for a in ARMS:
    P, S, L = (x / n for x in acc[a])
    sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    if a == "base":
        base = sc
    print(f"{a:>10} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {sc-base:+9.4f}")
