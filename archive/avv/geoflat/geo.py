#!/usr/bin/env python
"""Where does our LPIPS actually live on the PRODUCTION harness? Spatial LPIPS map of the shipped
chain output vs real test GT, split by distance to the nearest strong GT edge."""
import io, os, sys
import numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import lpips as lpips_pkg
cv2.setNumThreads(8); Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
CD = "/mnt/d/avv/geoflat/cache_HCM0181"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
dev = "cuda"
vgg = lpips_pkg.LPIPS(net="vgg", spatial=True).to(dev).eval()
BINS = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 10), (10, 1e9)]
tot = np.zeros((len(BINS), 3))   # npix, lpips sum, sq-err sum
stems = sorted(f[:-4] for f in os.listdir(CD) if f.endswith(".npz"))[:N]
for i, s in enumerate(stems):
    z = np.load(os.path.join(CD, s + ".npz"))
    B = z["B"].astype(np.float32); G = z["G"].astype(np.float32); D = z["D"].astype(np.float32)
    b = io.BytesIO()
    Image.fromarray((np.clip(B, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
    J = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float32) / 255.
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        m = vgg(t(J) * 2 - 1, t(G) * 2 - 1)[0, 0].cpu().numpy()
    se = ((J - G) ** 2).mean(2)
    for k, (lo, hi) in enumerate(BINS):
        w = (D >= lo) & (D < hi)
        tot[k] += [w.sum(), m[w].sum(), se[w].sum()]
    if i % 5 == 0:
        print(i, flush=True)
n, l, e = tot[:, 0], tot[:, 1], tot[:, 2]
print(f"\nLPIPS/MSE geography, shipped chain, HCM0181, n={len(stems)}   (mean LPIPS {l.sum()/n.sum():.5f})")
print(f"{'dist to edge':>14} {'%pix':>6} {'%LPIPS':>8} {'%MSE':>7} {'LPIPS dens':>11} {'MSE dens':>9}")
for k, (lo, hi) in enumerate(BINS):
    print(f"{lo:5.0f}-{hi if hi<1e8 else 999:5.0f} px {100*n[k]/n.sum():6.2f} {100*l[k]/l.sum():8.2f} "
          f"{100*e[k]/e.sum():7.2f} {l[k]/n[k]/(l.sum()/n.sum()):11.3f} {e[k]/n[k]/(e.sum()/n.sum()):9.3f}")
w = np.array([lo >= 6 for lo, _ in BINS])
print(f"\nFLAT (>=6px from any edge): {100*n[w].sum()/n.sum():.1f}% of pixels, "
      f"{100*l[w].sum()/l.sum():.1f}% of LPIPS, {100*e[w].sum()/e.sum():.1f}% of MSE")
