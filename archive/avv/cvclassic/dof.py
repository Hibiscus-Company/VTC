#!/usr/bin/env python
"""DOF ladder on the per-frame residual warp.

The oracle per-frame warp is worth +1.58 over the shipped global field, but a dense
flow fitted against GT can partly COPY GT structure. Ask how many degrees of freedom
the gain actually needs:

  global      frame-invariant field                (shipped; 0 per-frame DOF)
  trans       + per-frame translation              (2)
  affine      + per-frame affine                   (6)
  quad        + per-frame quadratic                (12)
  smooth32/16 + per-frame field, heavily smoothed  (~10^2..10^3)
  dense       + per-frame field at 1/8             (oracle)

A gain that survives at LOW DOF is a real per-frame geometric/camera effect and is a
candidate for prediction from pose alone. A gain that only appears at high DOF is flow
overfitting to GT and is NOT reachable.
"""
import os, argparse
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="/tmp/cvdiag/res")
ap.add_argument("--src", default="/tmp/cvdiag/o_plain")
ap.add_argument("--gt", default="/mnt/d/avv/evalsplit/HCM0181/eval_gt")
a = ap.parse_args()

F = np.load(os.path.join(a.res, "fields.npy"))        # n,h,w,2  (per-frame, 1/8 res)
mu = np.load(os.path.join(a.res, "field_mean.npy"))
gt = sorted(os.listdir(a.gt))
stems = sorted(os.path.splitext(f)[0] for f in gt)
srcs = {os.path.splitext(f)[0]: os.path.join(a.src, f) for f in os.listdir(a.src)}
stems = [s for s in stems if s in srcs][:len(F)]
n, h, w, _ = F.shape
yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
X = (xx / w - 0.5).ravel(); Y = (yy / h - 0.5).ravel()
ones = np.ones_like(X)
BAS = {"trans": np.stack([ones], 1),
       "affine": np.stack([ones, X, Y], 1),
       "quad": np.stack([ones, X, Y, X * X, X * Y, Y * Y], 1)}

R = F - mu                       # per-frame residual on top of the global field
variants = {"global": np.zeros_like(F)}
for name, B in BAS.items():
    P = B @ np.linalg.lstsq(B, B, rcond=None)[0]        # placeholder, replaced below
    fit = np.zeros_like(R)
    for i in range(n):
        for c in (0, 1):
            coef, *_ = np.linalg.lstsq(B, R[i, ..., c].ravel(), rcond=None)
            fit[i, ..., c] = (B @ coef).reshape(h, w)
    variants[name] = fit
for k, sig in (("smooth32", 4.0), ("smooth16", 2.0), ("smooth8", 1.0)):
    variants[k] = np.stack([np.stack([cv2.GaussianBlur(R[i, ..., c], (0, 0), sig)
                                      for c in (0, 1)], -1) for i in range(n)])
variants["dense"] = R

for k in variants:
    os.makedirs(os.path.join(a.res, "dof_" + k), exist_ok=True)


def warp(img, fu):
    H, W, _ = img.shape
    f = cv2.resize(fu, (W, H), interpolation=cv2.INTER_CUBIC)
    ygr, xgr = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, (xgr + f[..., 0]).astype(np.float32),
                     (ygr + f[..., 1]).astype(np.float32),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


print("per-frame residual magnitude captured by each model (px, mean |d|):")
for k, v in variants.items():
    print(f"  {k:9s} {np.linalg.norm(v,axis=3).mean():.4f}   "
          f"(residual left {np.linalg.norm(R-v,axis=3).mean():.4f})")

for i, s in enumerate(stems):
    img = np.asarray(Image.open(srcs[s]).convert("RGB"), dtype=np.float32)
    for k, v in variants.items():
        out = warp(img, (mu + v[i]).astype(np.float32))
        Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(
            os.path.join(a.res, "dof_" + k, s + ".png"))
print("wrote", len(stems), "x", len(variants))
