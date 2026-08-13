#!/usr/bin/env python
"""ORACLE DECOMPOSITION of the flat-region error.

Everything runs through the tail of the SHIPPED chain: cached B (mean -> energy restore lam=1.0 ->
median lens field, lanczos4)  ->  operator  ->  JPEG q100/ss2  ->  score against real test GT.

Each arm hands the render one band of GT truth inside the flat mask only, so the resulting score
delta is the CEILING of any operator that works on that band in that geography.
"""
import io, os, sys, json
import numpy as np, cv2, torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
cv2.setNumThreads(8); Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
SC = sys.argv[1] if len(sys.argv) > 1 else "HCM0181"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
CD = f"/mnt/d/avv/geoflat/cache_{SC}"
dev = "cuda"
vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()


def enc(x):
    b = io.BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
    n = len(b.getvalue())
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float32) / 255., n


def t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0).to(dev)


ARMS = ["base", "oALL_flat", "oLF_flat", "oMF_flat", "oHF_flat", "oALL_near", "oLF_all"]
acc = {a: np.zeros(3) for a in ARMS}
stems = sorted(f[:-4] for f in os.listdir(CD) if f.endswith(".npz"))[:N]
print(f"{SC} n={len(stems)}", flush=True)
for i, s in enumerate(stems):
    z = np.load(os.path.join(CD, s + ".npz"))
    B = z["B"].astype(np.float32); G = z["G"].astype(np.float32); D = z["D"].astype(np.float32)
    M = (cv2.GaussianBlur((D > 6).astype(np.float32), (0, 0), 2.0))[..., None]   # soft flat mask
    Mn = (cv2.GaussianBlur((D <= 2).astype(np.float32), (0, 0), 2.0))[..., None]
    Bl8, Gl8 = cv2.GaussianBlur(B, (0, 0), 8.0), cv2.GaussianBlur(G, (0, 0), 8.0)
    Bl2, Gl2 = cv2.GaussianBlur(B, (0, 0), 2.0), cv2.GaussianBlur(G, (0, 0), 2.0)
    outs = {
        "base": B,
        "oALL_flat": B + M * (G - B),
        "oLF_flat": B + M * (Gl8 - Bl8),
        "oMF_flat": B + M * ((Gl2 - Gl8) - (Bl2 - Bl8)),
        "oHF_flat": B + M * ((G - Gl2) - (B - Bl2)),
        "oALL_near": B + Mn * (G - B),
        "oLF_all": B + (Gl8 - Bl8),
    }
    g = t(G)
    for a in ARMS:
        j, _ = enc(outs[a])
        r = t(j)
        with torch.no_grad():
            acc[a][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
            acc[a][1] += float(repo_ssim(r, g))
            acc[a][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
    if i % 10 == 0:
        print(f"  {i}", flush=True)

n = len(stems)
print(f"\n{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'dSCORE':>8}")
base = None
res = {}
for a in ARMS:
    P, S, L = acc[a] / n
    sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    base = sc if a == "base" else base
    res[a] = dict(score=sc, psnr=P, ssim=S, lpips=L, d=sc - base)
    print(f"{a:>12} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {sc-base:+8.4f}")
json.dump(res, open(f"/mnt/d/avv/geoflat/oracle_{SC}.json", "w"), indent=1)
