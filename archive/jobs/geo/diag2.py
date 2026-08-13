#!/usr/bin/env python
"""DIAGNOSIS 2: is the flat-region error a SYSTEMATIC TONE offset, and is our flat-region HF
coherent with GT (i.e. is amplifying our own HF amplifying signal or noise)?"""
import io, os, sys
import numpy as np
import cv2
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, boxf, _K
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

NB = 16
edges = np.linspace(0, 1, NB + 1)
tone = np.zeros((len(stems), NB, 3)); tcnt = np.zeros((len(stems), NB, 3))
per_view_bias = []
coh = {k: [0.0, 0.0, 0.0] for k in ("flat", "mid", "edge")}   # <K,G>, E_K, E_G on L0
sig = {k: [0.0, 0.0, 0.0] for k in ("flat", "mid", "edge")}   # sx2, sy2, sxy at 11x11

for si, s in enumerate(stems):
    mem = [loadt(os.path.join(d, s + ".png")) for d in MEM]
    ens = torch.stack(mem).mean(0)
    out = restore(ens, mem, 1.0, len(mem), 3).clamp(0, 1)
    x = np.clip(warp(out[0].permute(1, 2, 0).numpy(), lens, "lanczos"), 0, 1)
    X = enc(x)
    G = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"), dtype=np.float32) / 255.0
    H, W, _ = G.shape
    gg = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gg, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(gg, cv2.CV_32F, 0, 1, 3)
    gm = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 3.0)
    t_lo, t_hi = np.percentile(gm, [35.0, 73.0])
    masks = {"flat": gm <= t_lo, "edge": gm >= t_hi}
    masks["mid"] = ~masks["flat"] & ~masks["edge"]

    R = X - G
    per_view_bias.append(R.mean((0, 1)))
    # tone curve: mean residual as a function of OUR OWN value (usable at test time)
    for c in range(3):
        idx = np.clip((X[..., c] * NB).astype(int), 0, NB - 1)
        np.add.at(tone[si, :, c], idx.ravel(), R[..., c].ravel())
        np.add.at(tcnt[si, :, c], idx.ravel(), 1)
    # L0 coherence
    Xt = torch.from_numpy(X).permute(2, 0, 1).unsqueeze(0)
    Gt = torch.from_numpy(G).permute(2, 0, 1).unsqueeze(0)
    LX = lap_pyr(Xt, 5, _K)[0][0][0].permute(1, 2, 0).numpy()
    LG = lap_pyr(Gt, 5, _K)[0][0][0].permute(1, 2, 0).numpy()
    for nm, M in masks.items():
        coh[nm][0] += float((LX[M] * LG[M]).sum())
        coh[nm][1] += float((LX[M] ** 2).sum())
        coh[nm][2] += float((LG[M] ** 2).sum())
    # local 11x11 stats on luma for the SSIM contrast argument
    xg = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
    mx = cv2.GaussianBlur(xg, (0, 0), 1.5); my = cv2.GaussianBlur(gg, (0, 0), 1.5)
    sx2 = cv2.GaussianBlur(xg * xg, (0, 0), 1.5) - mx * mx
    sy2 = cv2.GaussianBlur(gg * gg, (0, 0), 1.5) - my * my
    sxy = cv2.GaussianBlur(xg * gg, (0, 0), 1.5) - mx * my
    for nm, M in masks.items():
        sig[nm][0] += sx2[M].mean(); sig[nm][1] += sy2[M].mean(); sig[nm][2] += sxy[M].mean()
    print(f"  {si+1}/{len(stems)}", flush=True)

n = len(stems)
pv = np.array(per_view_bias)
print(f"\n=== per-view global bias (ours - GT), x1e3 ===")
print(f"  mean {pv.mean(0)*1e3}  std across views {pv.std(0)*1e3}")
print(f"  sign consistency (frac of views with bias<0): {(pv<0).mean(0)}")

T = tone.sum(0) / np.maximum(tcnt.sum(0), 1)
print(f"\n=== TONE CURVE: mean residual (ours-GT) vs our own value, x1e3 ===")
print(f"{'bin':>6} {'npix%':>7} {'R':>8} {'G':>8} {'B':>8} {'LOO-consistency r':>18}")
for b in range(NB):
    frac = tcnt.sum(0)[b, 1] / tcnt.sum(0)[:, 1].sum() * 100
    print(f"{(b+0.5)/NB:6.3f} {frac:7.2f} {T[b,0]*1e3:8.3f} {T[b,1]*1e3:8.3f} {T[b,2]*1e3:8.3f}")
# honest LOO: does a tone curve fit on n-1 views predict the held-out view's curve?
pv_tone = tone / np.maximum(tcnt, 1)
w = tcnt / np.maximum(tcnt.sum(0, keepdims=True), 1)
tot = 0.0; res = 0.0
for i in range(n):
    m = np.delete(pv_tone, i, 0).mean(0)
    ok = tcnt[i] > 500
    tot += float((pv_tone[i][ok] ** 2 * tcnt[i][ok]).sum())
    res += float(((pv_tone[i] - m)[ok] ** 2 * tcnt[i][ok]).sum())
print(f"  LOO tone-curve: per-view tone energy {tot:.4e} -> residual {res:.4e} "
      f"=> explained {100*(1-res/tot):.1f}%")

print(f"\n=== L0 (top-octave) coherence with GT, by geography ===")
print(f"{'region':>6} {'|corr|':>8} {'E_ours/E_GT':>12} {'opt gain (MSE)':>15} {'gain to match E':>16}")
for nm in ("flat", "mid", "edge"):
    d, ek, eg = coh[nm]
    print(f"{nm:>6} {d/np.sqrt(ek*eg):8.4f} {ek/eg:12.4f} {d/ek:15.4f} {np.sqrt(eg/ek):16.4f}")

print(f"\n=== local sigma (gauss 1.5) by geography; SSIM contrast term ===")
print(f"{'region':>6} {'sx':>9} {'sy':>9} {'sx/sy':>7} {'struct sxy/(sx*sy)':>19} "
      f"{'cs now':>8} {'cs at sx=sy':>12}")
C2 = 0.03 ** 2
for nm in ("flat", "mid", "edge"):
    sx2, sy2, sxy = (v / n for v in sig[nm])
    sx, sy = np.sqrt(max(sx2, 0)), np.sqrt(max(sy2, 0))
    cs_now = (2 * sxy + C2) / (sx2 + sy2 + C2)
    k = sy / max(sx, 1e-9)
    cs_m = (2 * k * sxy + C2) / (k * k * sx2 + sy2 + C2)
    print(f"{nm:>6} {sx:9.5f} {sy:9.5f} {sx/sy:7.4f} {sxy/(sx*sy):19.4f} "
          f"{cs_now:8.4f} {cs_m:12.4f}")
