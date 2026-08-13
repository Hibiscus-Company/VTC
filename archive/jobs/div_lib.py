#!/usr/bin/env python
"""Shared library for the DIVERSITY probe. Hoists every computation that is common to arms.

KEY IDENTITY (verified against the production restore() at startup):
  lap_recon([L0*(1+lam(r-1))] + laps[1:]) == ens + lam*(r-1)*L0(ens)      (pyramid is linear)
  L0(mean_i x_i) == mean_i L0(x_i)                                        (analysis is linear)
  mean_i boxf(sum_c (L0_i - L0_S)^2) == mean_i A_i - Eb,  A_i = boxf(sum_c L0_i^2)
So per stem we compute L0_i and A_i ONCE per variant; every arm is then O(1) elementwise algebra
instead of k+1 fresh Laplacian pyramids. Chain cost per arm is then warp + JPEG + LPIPS only.
"""
import io, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from PIL import Image
from lapfuse import _K, pyr_down, pyr_up, boxf

Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
RD = lambda m: f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png"
GTD = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"

# --- the three families present at HCM0181 -------------------------------------------------
# UT      : gsplat_track --ut, distorted-space unscented transform, native UT render
#           (this is the family EVERY production tower member belongs to)
# UTd     : same UT checkpoint (B11ut60k) re-rendered or 8k-step refit -> degenerate diversity
# AA      : gsplat_track WITHOUT --ut = the antialiased rasteriser, undistort+warp render
# FGS     : the FastGS repo (train.py) on images_undist -- a different codebase entirely
FAM = {
    "gsplatB9ut": "UT", "gsplatB10ut8M": "UT", "gsplatB11ut60k": "UT", "gsplatB12ut8Ms7": "UT",
    "sh0": "UTd", "sh1": "UTd", "sh2": "UTd", "sh3": "UTd",
    "m31b_nolpips": "UTd", "m31b_taillpips": "UTd",
    "gsplatB1": "AA", "gsplatB2": "AA", "gsplatB3": "AA", "gsplatB4warm": "AA",
    "gsplatB5affine": "AA", "gsplatB6bilagrid": "AA", "gsplatB7ppisp2": "AA",
    "gsplatB8pure": "AA",
    "e15ceil95": "FGS", "e16app": "FGS", "e17visnorm": "FGS",
}
UT4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
ALL = UT4 + [m for m in FAM if m not in UT4]


def load_field(gain=1.30, ds="s8", pool="median", how="cubic"):
    from fieldlib import LooPool, upsample
    c = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in c["HW"]]
    f = upsample(LooPool(c[ds]).pooled(pool), H, W, how)
    return (f * gain).astype(np.float32)


def stems():
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    ss = sorted(s for s in gt_by
                if all(os.path.exists(os.path.join(RD(m), s + ".png")) for m in ALL))
    return ss, gt_by


def ld(p, dev):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)


def prep(x, K, win=3):
    """returns L0 [1,3,H,W] and A = boxf(sum_c L0^2) [1,1,H,W]"""
    L0 = x - pyr_up(pyr_down(x, K), x.shape[-2:], K)
    return L0, boxf((L0 ** 2).sum(1, keepdim=True), win)


def combine(X, L0, A, idx, lam, w=None, clamp=4.0):
    """X,L0 dict name->tensor ; A dict name->energy map ; idx list of names.
    Returns the energy-restored ensemble, exactly as production does it."""
    if w is None:
        ens = torch.stack([X[i] for i in idx]).mean(0)
        Ls = torch.stack([L0[i] for i in idx]).mean(0)
        Am = torch.stack([A[i] for i in idx]).mean(0)
    else:
        ww = torch.tensor(w, dtype=torch.float32, device=X[idx[0]].device)
        ww = ww / ww.sum()
        ens = sum(float(a) * X[i] for a, i in zip(ww, idx))
        Ls = sum(float(a) * L0[i] for a, i in zip(ww, idx))
        Am = sum(float(a) * A[i] for a, i in zip(ww, idx))
    # effective member count: (sum w)^2 / sum w^2 -- reduces to len(idx) for uniform weights,
    # which is exactly what production passes as --k
    if w is None:
        k = float(len(idx))
    else:
        a = np.asarray(w, dtype=np.float64)
        k = float(a.sum() ** 2 / (a ** 2).sum())
    if k <= 1.0 or lam == 0.0:
        return ens
    Eb = boxf((Ls ** 2).sum(1, keepdim=True), 3)
    V = (Am - Eb) * (k / (k - 1.0))
    r = torch.sqrt((1.0 + V / (Eb + 1e-10)).clamp(min=0.0)).clamp(max=clamp)
    return ens + lam * (r - 1.0) * Ls


def chain(out, lens, warp):
    """restored tensor -> field warp (lanczos4) -> JPEG q100/ss2 round trip -> numpy HWC"""
    x = out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    x = np.clip(warp(x, lens, "lanczos"), 0, 1)
    b = io.BytesIO()
    Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
    nb = len(b.getvalue())
    j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                   dtype=np.float32) / 255.0
    return np.ascontiguousarray(j), nb


def score(fmt="{:.4f}"):
    pass
