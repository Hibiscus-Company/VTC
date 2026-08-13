#!/usr/bin/env python
"""INDEPENDENT VERIFICATION of the Laplacian-fusion machinery before trusting any number.
 1. analysis/synthesis is exact (recon == input to float eps)
 2. the 'mean' rule at every level reproduces the PIXEL MEAN bit-exactly  => any delta is the rule
 3. the energy rule's output equals an independent numpy re-implementation
 4. the harness GT/renders are the ones claimed (shape, count, and k4 == uint8 mean of 4 members)
"""
import os, sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lapfuse import lap_pyr, lap_recon, fuse_image, _K, boxf
Image.MAX_IMAGE_PIXELS = None

GT = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
MEM = [f"/mnt/d/avv/output/HCM0181_{t}/test_poses_renders_png" for t in
       ("gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7")]
K4 = "/mnt/d/avv/prodharness/k4/png"

dev = "cuda"
k = _K.to(dev)
gtf = sorted(os.listdir(GT))
stems = [os.path.splitext(f)[0] for f in gtf]
print(f"GT files {len(gtf)}  first={gtf[0]}")
s = stems[3]
st = torch.stack([torch.from_numpy(np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                                              dtype=np.float32) / 255.).permute(2, 0, 1)
                  for d in MEM]).to(dev)
g = np.asarray(Image.open(os.path.join(GT, gtf[3])).convert("RGB"))
print(f"member stack {tuple(st.shape)}   GT {g.shape}")

# --- 1. exactness of analysis/synthesis
laps, res, sizes = lap_pyr(st, 5, k)
rec = lap_recon(laps, res, sizes, k)
print(f"[1] recon max|err|            = {(rec - st).abs().max().item():.3e}  (must be ~1e-6)")

# --- 2. mean rule == pixel mean
pm = st.mean(0, keepdim=True)
id0 = fuse_image(st, dict(nalt=0, rule="energy", kw=dict(lam=9.0)), 5, k)
print(f"[2] nalt=0 vs pixel-mean      = {(id0 - pm).abs().max().item():.3e}  (must be ~1e-6)")
u_pm = (pm.clamp(0, 1) * 255).round().to(torch.uint8)
u_id = (id0.clamp(0, 1) * 255).round().to(torch.uint8)
print(f"    uint8 identical           = {bool((u_pm == u_id).all())}")

# --- 3. independent numpy re-implementation of energy rule at level 0
def np_blur(x, kk):
    from scipy.ndimage import correlate1d
    x = correlate1d(x, kk, axis=0, mode="reflect")
    return correlate1d(x, kk, axis=1, mode="reflect")

kn = np.array([1., 4., 6., 4., 1.]) / 16.
A = st.permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)          # [k,H,W,3]
L0 = []
for i in range(A.shape[0]):
    d = np_blur(A[i], kn)[::2, ::2]
    up = np.zeros_like(A[i]); up[::2, ::2] = d
    up = np_blur(up, kn * 2.0)
    L0.append(A[i] - up)
L0 = np.stack(L0)
Lb = L0.mean(0)
def np_box(x, w):
    from scipy.ndimage import uniform_filter
    return uniform_filter(x, size=(w, w), mode="reflect")
w, lam = 3, 1.0
Em = np.mean([np_box((L0[i] ** 2).sum(-1), w) for i in range(L0.shape[0])], axis=0)
Eb = np_box((Lb ** 2).sum(-1), w)
r = np.clip(np.sqrt((Em + 1e-10) / (Eb + 1e-10)), None, 4.0)
delta_np = (Lb * (lam * (r - 1.0))[..., None])                       # what gets ADDED to pixel mean
out_np = A.mean(0) + delta_np
out_t = fuse_image(st, dict(nalt=1, rule="energy", kw=dict(lam=lam, win=w)), 5, k)
out_t = out_t[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
print(f"[3] numpy-vs-torch energy     = {np.abs(out_np - out_t).max():.3e}  "
      f"(reflect-pad conventions differ slightly at borders; interior:"
      f" {np.abs(out_np - out_t)[20:-20, 20:-20].max():.3e})")

# --- 4. k4 == uint8 mean of the 4 members
bad = 0
for s2 in stems[:5]:
    ms = np.mean([np.asarray(Image.open(os.path.join(d, s2 + ".png")).convert("RGB"), dtype=np.float32)
                  for d in MEM], axis=0)
    k4 = np.asarray(Image.open(os.path.join(K4, s2 + ".png")).convert("RGB"), dtype=np.float32)
    bad += int(np.abs(np.round(ms) - k4).max() > 0)
print(f"[4] k4 == round(mean of 4)    = {bad == 0}  (5 images checked)")
print(f"    mean boost r-1 at lam=1   = {float((r - 1).mean()):.4f}   frac(r>1) = {float((r>1).mean()):.3f}")
