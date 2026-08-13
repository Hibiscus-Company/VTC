#!/usr/bin/env python
"""Geography-split energy restoration.

The shipped operator applies one lambda everywhere. arms.py showed LPIPS is steeply sensitive to
incoherent fine detail (+0.0021 LPIPS for only 0.7/255 of added noise over 26% of pixels), and the
error geography says edges and flats behave differently. So: does the restoration want a DIFFERENT
lambda in flat regions than at edges? Shipping this is one mask multiply inside restore().

Chain per arm: 4-member mean -> restore(lambda MAP) -> median lens field lanczos4 -> JPEG q100/ss2.
The mask is built from our own ensemble mean, never from GT.
"""
import io, os, sys, json
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
SC = "HCM0181"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MEM = [f"/mnt/d/avv/output/{SC}_{t}/test_poses_renders_png"
       for t in ("gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7")]
GTD = f"/mnt/d/avv/data/phase1/public_set/{SC}/test/images"
dev = "cuda"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def enc(x):
    b = io.BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
    nb = len(b.getvalue())
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float32) / 255., nb


gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:N]
cache = np.load(f"{TMP}/lens/cache/pub_{SC}.npz")
lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
# lambda in flat, lambda elsewhere
ARMS = {"base": (1.0, 1.0), "flat0.0": (0.0, 1.0), "flat0.5": (0.5, 1.0), "flat1.5": (1.5, 1.0),
        "flat0.5_edge1.25": (0.5, 1.25), "edge1.25": (1.0, 1.25), "all1.25": (1.25, 1.25)}
acc = {a: np.zeros(3) for a in ARMS}
nby = {a: 0 for a in ARMS}
print(f"{SC} n={len(stems)}", flush=True)
for i, s in enumerate(stems):
    mem = [load(os.path.join(d, s + ".png")) for d in MEM]
    ens = torch.stack(mem).mean(0)
    K = _K
    laps, res, sizes = lap_pyr(ens, 5, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), 3)
    V = 0.0
    for m in mem:
        V = V + boxf(((lap_pyr(m, 5, K)[0][0] - L0) ** 2).sum(1, keepdim=True), 3)
    V = V / len(mem) * (len(mem) / (len(mem) - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=4.0)
    ey = cv2.cvtColor(ens[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
    gm = cv2.GaussianBlur(np.abs(cv2.Sobel(ey, cv2.CV_32F, 1, 0, 3)) +
                          np.abs(cv2.Sobel(ey, cv2.CV_32F, 0, 1, 3)), (0, 0), 1.0)
    Db = cv2.distanceTransform((gm <= np.percentile(gm, 80)).astype(np.uint8), cv2.DIST_L2, 3)
    M = torch.from_numpy(cv2.GaussianBlur((Db > 6).astype(np.float32), (0, 0), 2.0))[None, None]
    G = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"), np.float32) / 255.
    g = torch.from_numpy(np.ascontiguousarray(G)).permute(2, 0, 1).unsqueeze(0).to(dev)
    for a, (lf, le) in ARMS.items():
        lam = le + (lf - le) * M
        out = lap_recon([L0 * (1.0 + lam * (r - 1.0))] + laps[1:], res, sizes, K).clamp(0, 1)
        x = np.ascontiguousarray(out[0].permute(1, 2, 0).numpy())
        x = np.clip(warp(x, lens, "lanczos"), 0, 1)
        j, nb = enc(x)
        nby[a] += nb
        rr = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            acc[a][0] += 10 * np.log10(1.0 / max(((rr - g) ** 2).mean().item(), 1e-12))
            acc[a][1] += float(repo_ssim(rr, g))
            acc[a][2] += float(vgg(rr * 2 - 1, g * 2 - 1).item())
    if i % 10 == 0:
        print(f"  {i}", flush=True)

n = len(stems)
print(f"\n{'arm (lam flat/edge)':>20} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'dSCORE':>8} {'MB/60':>7}")
base = None; out = {}
for a in ARMS:
    P, S, L = acc[a] / n
    sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    base = sc if a == "base" else base
    out[a] = dict(score=sc, psnr=P, ssim=S, lpips=L, d=sc - base, mb=nby[a] / 1e6)
    print(f"{a:>20} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {sc-base:+8.4f} {nby[a]/1e6:7.1f}")
json.dump(out, open(f"/mnt/d/avv/geoflat/arms2_{SC}.json", "w"), indent=1)
