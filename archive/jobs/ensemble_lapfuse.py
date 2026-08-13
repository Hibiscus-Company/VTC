#!/usr/bin/env python
"""PRODUCTION DROP-IN for ensemble_renders.py's averaging step.

Identical CLI surface to ensemble_renders.py (--dirs/--weights/--out/--png_dir/--names_from), but
replaces the flat pixel mean with a Laplacian-pyramid fusion in which ONLY the finest level is
recombined by local-energy restoration:

    L0_out = L0_bar * (1 + lam * (r - 1)),
        r  = sqrt( sum_i w_i * E_i / E(L0_bar) ),   E(x) = boxfilter(||x||^2, win)
    every coarser level and the residual: plain weighted mean  (=> identical to today's output)

Because the Burt-Adelson pyramid is exact and linear, --lam 0 reproduces ensemble_renders.py's
pixel mean BIT-EXACTLY (asserted by --selftest). All the behaviour change is in one band.

WHY: averaging k members cancels each member's independent finest-scale texture, so the mean sits
below every member (and far below the GT) in level-0 energy. r is the per-pixel factor that puts
that energy back, and it is LARGE exactly where the members disagree. Measured on the production
harness (HCM0181, 60 real test poses, real test GT, k=4): +0.174 PNG / +0.113 post-JPEG at
lam=1.0/win=5; gain GROWS with k (+0.091/+0.140/+0.174 at k=2/3/4).

RULE 10 / SAFETY: uses no ground truth of any kind, has no fitted parameters, and is a pure
function of the member renders.
"""
import os, csv, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
_K = torch.tensor([1., 4., 6., 4., 1.]) / 16.


def _blur(x, k):
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, 2, 2), mode="reflect"), k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


def _down(x, k):
    return _blur(x, k)[:, :, ::2, ::2]


def _up(x, size, k):
    B, C, H, W = x.shape
    u = x.new_zeros(B, C, H * 2, W * 2)
    u[:, :, ::2, ::2] = x
    return _blur(u, k * 2.0)[:, :, :size[0], :size[1]]


def _boxf(x, w):
    if w <= 1:
        return x
    p = w // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="reflect"), w, stride=1)


def lapfuse(stack, w, lam=1.0, win=5, nlev=5, rmax=4.0):
    """stack [k,3,H,W] float in [0,1]; w [k] normalised weights -> fused [1,3,H,W]."""
    k = _K.to(stack.device)
    wv = w.view(-1, 1, 1, 1)
    laps, sizes, g = [], [], stack
    for _ in range(nlev):
        sizes.append(g.shape[-2:])
        d = _down(g, k)
        laps.append(g - _up(d, g.shape[-2:], k))
        g = d
    out = [(wv * L).sum(0, keepdim=True) for L in laps]
    if lam != 0.0:
        L0 = laps[0]
        bar = out[0]
        Em = (wv * _boxf((L0 ** 2).sum(1, keepdim=True), win)).sum(0, keepdim=True)
        Eb = _boxf((bar ** 2).sum(1, keepdim=True), win)
        r = torch.sqrt((Em + 1e-10) / (Eb + 1e-10)).clamp(max=rmax)
        out[0] = bar * (1.0 + lam * (r - 1.0))
    rec = (wv * g).sum(0, keepdim=True)
    for i in range(nlev - 1, -1, -1):
        rec = out[i] + _up(rec, sizes[i], k)
    return rec


def load_stems(d):
    files = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.JPG"))
                   + glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.PNG")))
    return {os.path.splitext(os.path.basename(f))[0]: f for f in files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--png_dir", default=None)
    p.add_argument("--names_from", default=None)
    p.add_argument("--weights", type=float, nargs="+", default=None)
    p.add_argument("--jpeg_quality", type=int, default=100)
    p.add_argument("--jpeg_subsampling", type=int, default=2)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--win", type=int, default=3)
    p.add_argument("--nlev", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--selftest", action="store_true",
                   help="assert lam=0 reproduces the plain weighted pixel mean, then exit")
    a = p.parse_args()

    members = [load_stems(d) for d in a.dirs]
    stems = set(members[0])
    for d, m in zip(a.dirs, members):
        assert set(m) == stems, f"{d} file stems differ from {a.dirs[0]}"
    w = np.ones(len(members)) if a.weights is None else np.asarray(a.weights, dtype=np.float64)
    assert len(w) == len(members) and (w >= 0).all() and w.sum() > 0
    w = torch.tensor(w / w.sum(), dtype=torch.float32, device=a.device)

    def stack_of(s):
        return torch.stack([torch.from_numpy(
            np.asarray(Image.open(m[s]).convert("RGB"), dtype=np.float32) / 255.0
        ).permute(2, 0, 1) for m in members]).to(a.device)

    if a.selftest:
        s = sorted(stems)[0]
        st = stack_of(s)
        f0 = lapfuse(st, w, lam=0.0, win=a.win, nlev=a.nlev)
        ref = (w.view(-1, 1, 1, 1) * st).sum(0, keepdim=True)
        e = (f0 - ref).abs().max().item()
        print(f"selftest lam=0 vs weighted pixel mean: maxabs {e:.3e}  "
              f"(uint8 LSB = {1/255:.3e})  -> {'PASS' if e < 1e-5 else 'FAIL'}")
        u1 = (f0.clamp(0, 1) * 255).round().to(torch.uint8)
        u2 = (ref.clamp(0, 1) * 255).round().to(torch.uint8)
        print(f"selftest uint8 identical: {bool((u1 == u2).all())}")
        return

    out_names = {s: s + ".JPG" for s in stems}
    if a.names_from:
        with open(a.names_from, newline="") as f:
            rows = list(csv.DictReader(f))
        cn = {os.path.splitext(r["image_name"])[0]: r["image_name"] for r in rows}
        assert set(cn) == stems, "csv image stems differ from render stems"
        out_names = cn
    os.makedirs(a.out, exist_ok=True)
    if a.png_dir:
        os.makedirs(a.png_dir, exist_ok=True)
    for s in sorted(stems):
        f = lapfuse(stack_of(s), w, lam=a.lam, win=a.win, nlev=a.nlev)
        arr = (f.clamp(0, 1) * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(arr)
        im.save(os.path.join(a.out, os.path.splitext(out_names[s])[0] + ".jpg"), "JPEG",
                quality=a.jpeg_quality, subsampling=a.jpeg_subsampling, optimize=True, progressive=True)
        if a.png_dir:
            im.save(os.path.join(a.png_dir, s + ".png"), "PNG")
    print(f"lapfuse: wrote {len(stems)} images (lam={a.lam} win={a.win} nlev={a.nlev}, "
          f"{len(a.dirs)} members, weights {w.tolist()})")


if __name__ == "__main__":
    main()
