#!/usr/bin/env python
"""How much score is sitting in RESIDUAL (frame-specific) sub-pixel misregistration?

DIAGNOSTIC ONLY -- uses the eval-split GT, which is HELD-OUT TRAIN PHOTOS, never test.
Mirrors the D5-D8 oracle that sized the global lens field before it was made legal.

Arms:
  plain      ensemble mean
  global     mean warped by the MEAN residual field over all frames  (= the lens field,
             the legal frame-invariant part)
  oracle     mean warped by each frame's OWN residual field          (upper bound)
Also saves the per-frame fields for a PCA/predictability study.
"""
import os, argparse
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--outroot", required=True)
ap.add_argument("--ds", type=int, default=8)
ap.add_argument("--clip", type=float, default=4.0)
a = ap.parse_args()

os.makedirs(a.outroot, exist_ok=True)
for k in ("plain", "global", "oracle"):
    os.makedirs(os.path.join(a.outroot, k), exist_ok=True)

gtmap = {os.path.splitext(f)[0]: os.path.join(a.gt, f) for f in os.listdir(a.gt)}
srcs = {os.path.splitext(f)[0]: os.path.join(a.src, f) for f in os.listdir(a.src)
        if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")}
stems = sorted(set(gtmap) & set(srcs))


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def g8(a_):
    return np.clip(0.299 * a_[..., 0] + 0.587 * a_[..., 1] + 0.114 * a_[..., 2], 0, 255).astype(np.uint8)


def warp(img, fu):
    H, W, _ = img.shape
    f = cv2.resize(fu, (W, H), interpolation=cv2.INTER_CUBIC) if fu.shape[:2] != (H, W) else fu
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, (xx + f[..., 0]).astype(np.float32),
                     (yy + f[..., 1]).astype(np.float32),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
fields = {}
print(f"{len(stems)} frames; pass 1 flows")
for s in stems:
    r = load(srcs[s]); g = load(gtmap[s])
    if r.shape != g.shape:
        continue
    fl = np.clip(dis.calc(g8(g), g8(r), None), -a.clip, a.clip)
    H, W = fl.shape[:2]
    fields[s] = cv2.resize(fl, (W // a.ds, H // a.ds), interpolation=cv2.INTER_AREA)

F = np.stack([fields[s] for s in sorted(fields)])
mu = F.mean(0)
np.save(os.path.join(a.outroot, "fields.npy"), F)
np.save(os.path.join(a.outroot, "field_mean.npy"), mu)
res = F - mu
print(f"global field mean|d| {np.linalg.norm(mu,axis=2).mean():.4f} px   "
      f"frame-specific residual rms/axis {res.std():.4f} px  mean|d| "
      f"{np.linalg.norm(res,axis=3).mean():.4f} px")
print(f"energy: invariant {100*(mu**2).sum()*len(F)/ (F**2).sum():.2f}%  "
      f"frame-specific {100*(res**2).sum()/(F**2).sum():.2f}%")

print("pass 2 renders")
for s in sorted(fields):
    r = load(srcs[s])
    Image.fromarray(np.clip(r + 0.5, 0, 255).astype(np.uint8)).save(
        os.path.join(a.outroot, "plain", s + ".png"))
    Image.fromarray(np.clip(warp(r, mu) + 0.5, 0, 255).astype(np.uint8)).save(
        os.path.join(a.outroot, "global", s + ".png"))
    Image.fromarray(np.clip(warp(r, fields[s]) + 0.5, 0, 255).astype(np.uint8)).save(
        os.path.join(a.outroot, "oracle", s + ".png"))

# PCA of the frame-specific residual: is it low-dimensional (=> predictable)?
X = res.reshape(len(res), -1)
X -= X.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
ev = S ** 2 / (S ** 2).sum()
print("\nPCA of frame-specific residual field (explained variance):")
print("  " + "  ".join(f"{v*100:5.2f}%" for v in ev[:10]))
print(f"  cumulative top-3 {ev[:3].sum()*100:.2f}%  top-5 {ev[:5].sum()*100:.2f}%  "
      f"top-10 {ev[:10].sum()*100:.2f}%  (n={len(res)} frames)")
np.save(os.path.join(a.outroot, "pca_coeff.npy"), U[:, :10] * S[:10])
