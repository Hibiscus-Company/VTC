#!/usr/bin/env python
"""DIAGNOSIS: what is actually different about our FLAT regions vs GT?

Runs the FULL shipped chain (4-member mean -> energy restore lam=1.0 -> median lens field
lanczos4 -> JPEG q100/ss2) on HCM0181 test poses against real test GT, then decomposes the
residual by edge-geography and by frequency.  CPU only, no LPIPS here -- pure statistics.
"""
import io, os, sys
import numpy as np
import cv2
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from energy_restore import restore
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def loadt(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def enc(x):
    b = io.BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.float32) / 255.0


gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
stems = stems[::max(1, len(stems) // N)][:N]
cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")

acc = {}
LFstack, HWs = [], None
Gsky_ps, Xsky_ps = [], []
rows = []
for si, s in enumerate(stems):
    mem = [loadt(os.path.join(d, s + ".png")) for d in MEM]
    ens = torch.stack(mem).mean(0)
    out = restore(ens, mem, 1.0, len(mem), 3).clamp(0, 1)
    x = out[0].permute(1, 2, 0).numpy()
    x = np.clip(warp(x, lens, "lanczos"), 0, 1)
    X = enc(x)
    G = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"), dtype=np.float32) / 255.0
    H, W, _ = G.shape
    HWs = (H, W)

    gg = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gg, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gg, cv2.CV_32F, 0, 1, 3)
    gm = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 3.0)
    t_lo, t_hi = np.percentile(gm, [35.0, 73.0])
    flat = gm <= t_lo          # 35% farthest from edges
    edge = gm >= t_hi          # 27% near strong edges
    mid = ~flat & ~edge

    R = X - G
    # low-freq / high-freq split of the residual
    Rlf = np.stack([cv2.GaussianBlur(R[..., c], (0, 0), 16.0) for c in range(3)], -1)
    Rhf = R - Rlf
    # local std (noise proxy) at 5x5
    def lstd(A):
        m = cv2.blur(A, (5, 5))
        return np.sqrt(np.maximum(cv2.blur(A * A, (5, 5)) - m * m, 0))
    sG = lstd(cv2.cvtColor(G, cv2.COLOR_RGB2GRAY))
    sX = lstd(cv2.cvtColor(X, cv2.COLOR_RGB2GRAY))

    d = {}
    for nm, M in (("flat", flat), ("mid", mid), ("edge", edge)):
        d[nm] = dict(
            frac=float(M.mean()),
            mse=float((R[M] ** 2).mean()),
            mse_lf=float((Rlf[M] ** 2).mean()),
            mse_hf=float((Rhf[M] ** 2).mean()),
            bias=[float(R[..., c][M].mean()) for c in range(3)],
            sG=float(sG[M].mean()), sX=float(sX[M].mean()),
        )
    rows.append(d)
    # store a coarse LF residual field for across-view consistency
    LFstack.append(cv2.resize(Rlf, (W // 16, H // 16), interpolation=cv2.INTER_AREA))
    # sky/flat radial power spectrum: take the largest flat 256x256 tile
    ii = np.argmax(cv2.blur(flat.astype(np.float32), (64, 64))[128:-128, 128:-128])
    yy, xx = np.unravel_index(ii, (H - 256, W - 256))
    gt_t = cv2.cvtColor(G[yy:yy + 256, xx:xx + 256], cv2.COLOR_RGB2GRAY)
    x_t = cv2.cvtColor(X[yy:yy + 256, xx:xx + 256], cv2.COLOR_RGB2GRAY)
    win = np.hanning(256)[:, None] * np.hanning(256)[None, :]
    Gsky_ps.append(np.abs(np.fft.fftshift(np.fft.fft2((gt_t - gt_t.mean()) * win))) ** 2)
    Xsky_ps.append(np.abs(np.fft.fftshift(np.fft.fft2((x_t - x_t.mean()) * win))) ** 2)
    print(f"  {si+1}/{len(stems)} {s}", flush=True)

print(f"\n=== HCM0181, n={len(stems)}, full shipped chain ===")
print(f"{'region':>6} {'frac':>6} {'MSE':>10} {'LF share':>9} {'HF share':>9} "
      f"{'bias R/G/B (x1e3)':>26} {'sigma_GT':>9} {'sigma_ours':>10} {'ratio':>7}")
for nm in ("flat", "mid", "edge"):
    a = {k: np.mean([r[nm][k] for r in rows]) if not isinstance(rows[0][nm][k], list)
         else np.mean([r[nm][k] for r in rows], 0) for k in rows[0][nm]}
    print(f"{nm:>6} {a['frac']:6.3f} {a['mse']:10.3e} {a['mse_lf']/a['mse']:9.3f} "
          f"{a['mse_hf']/a['mse']:9.3f} "
          f"{a['bias'][0]*1e3:8.3f}{a['bias'][1]*1e3:9.3f}{a['bias'][2]*1e3:9.3f}   "
          f"{a['sG']:9.5f} {a['sX']:10.5f} {a['sX']/a['sG']:7.3f}")

LF = np.stack(LFstack)              # [n, h, w, 3]
med = np.median(LF, 0)
tot = float((LF ** 2).mean())
res = float(((LF - med[None]) ** 2).mean())
print(f"\nLF-residual across-view consistency (coarse 1/16 grid):")
print(f"  mean per-view LF energy      {tot:.4e}")
print(f"  after removing median field  {res:.4e}   -> explained {100*(1-res/tot):.1f}%")
# leave-one-out honest version
lo = LooPool(LF.reshape(LF.shape[0], -1))
resl = np.mean([((LF[i].ravel() - lo.median(i)) ** 2).mean() for i in range(LF.shape[0])])
print(f"  LOO median field             {resl:.4e}   -> explained {100*(1-resl/tot):.1f}%")
print(f"  per-view LF global mean energy (pure per-image offset): "
      f"{float((LF.mean((1,2))**2).mean()):.4e} ({100*float((LF.mean((1,2))**2).mean())/tot:.1f}%)")

Gp, Xp = np.mean(Gsky_ps, 0), np.mean(Xsky_ps, 0)
cy = 128
yy, xx = np.mgrid[0:256, 0:256]
rr = np.sqrt((yy - cy) ** 2 + (xx - cy) ** 2).astype(int)
print(f"\nFLAT-TILE radial power spectrum (256x256, luma, mean of {len(stems)} tiles)")
print(f"{'cyc/px':>8} {'GT':>12} {'ours':>12} {'ours/GT':>9}")
for lo_, hi_ in [(1, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 96), (96, 128)]:
    m = (rr >= lo_) & (rr < hi_)
    g_, x_ = Gp[m].mean(), Xp[m].mean()
    print(f"{(lo_+hi_)/2/256:8.3f} {g_:12.4e} {x_:12.4e} {x_/g_:9.3f}")
