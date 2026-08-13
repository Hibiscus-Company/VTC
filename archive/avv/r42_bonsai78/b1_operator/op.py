#!/usr/bin/env python
"""
B1 -- bonsai render-time (post-process) operator.
=================================================

A deterministic, side-effect-free, per-image operator applied to already-rendered frames,
just before the shipping encode.

LEGALITY
--------
Every parameter default in this file was fitted on TRAIN data only:
  renders  : /mnt/d/avv/blurbound/bonsai/train_render   (220 renders at train_sub poses,
                                                         bonsai_ema099 checkpoint)
  photos   : /mnt/d/avv/evalsplit/bonsai/train_sub/images (the 220 held-in TRAIN photos)
No private test GT and no eval-hole GT was used to choose any number here.

WHAT IT DOES (fixed, deterministic order)
-----------------------------------------
    shift -> guided -> unsharp/blur -> chroma blur -> tone -> gamma -> noise -> clip

The only component that is ON by default is a mild isotropic *blur* (negative-alpha unsharp).
The train fit is unambiguous about the direction: the MSE-optimal alpha over the 220 train
pairs is NEGATIVE at every scale (median -0.537 at sigma 0.6, -0.215 at 1.0, -0.107 at 1.6,
-0.062 at 2.5).  The render is marginally over-crunchy relative to the photo, not under-sharp.
Every component is individually switchable so the file doubles as the operator library.

HONEST VALUE: see the B1 report.  On the 28 bonsai eval holes this operator is worth about
+0.07 Score in float and it does NOT survive the shipping JPEG encode.  It is documented here
so the measurement is reproducible, not because it is worth shipping.

USAGE
-----
  python op.py --render_dir DIR --out_dir DIR                 # train-fitted defaults
  python op.py --render_dir DIR --out_dir DIR --params '{"us_sigma":1.6,"us_a":-0.15}'
  python op.py --render_dir DIR --out_dir DIR --preset identity
  python op.py --render_dir DIR --out_dir DIR --encode ship   # write the shipping JPEG instead
"""
import argparse
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

# ---------------------------------------------------------------- presets

PRESETS = {
    # Identity.  The honest recommendation for bonsai (see report).
    "identity": {},
    # The train-fitted operator: mild isotropic blur at the scale the train pairs prefer.
    # sigma 1.6 / a = -0.15 is the train-fit optimum rounded to the measured grid.
    "b1": {"us_sigma": 1.6, "us_a": -0.15},
    # Same, luma-only (chroma left alone) -- survives 4:2:0 slightly better.
    "b1_luma": {"us_sigma": 1.6, "us_a": -0.15, "us_luma_only": 1},
    # Train-fitted global photometric affine (per channel).  Sub-LSB; dies in uint8.
    "tone": {"gain": [0.996517, 0.996330, 0.995058],
             "bias": [0.002407, 0.002762, 0.002960]},
}

DEFAULTS = dict(
    shift_dx=0.0, shift_dy=0.0,               # sub-pixel translation (px, + = content right)
    guided_r=8, guided_eps=4e-4, guided_w=0.0,  # edge-preserving smoothing blend
    us_sigma=1.0, us_a=0.0, us_luma_only=0,   # unsharp/blur blend; a<0 == blur
    chroma_sigma=0.0,                         # blur the colour-difference channels only
    gain=(1.0, 1.0, 1.0), bias=(0.0, 0.0, 0.0), gamma=1.0,
    noise_sd=0.0, noise_seed=1234,            # deterministic seeded grain
)

SHIP_ENCODE = dict(format="JPEG", quality=100, subsampling=2, optimize=True, progressive=True)

_W = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)   # BT.601, JPEG's luma


# ---------------------------------------------------------------- primitives

def _gauss1d(sigma):
    rad = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    return torch.tensor(k, dtype=torch.float32), rad


def gaussian_blur(img, sigma):
    k, rad = _gauss1d(sigma)
    C = img.shape[1]
    kh = k.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()
    kv = k.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()
    x = F.conv2d(F.pad(img, (rad, rad, 0, 0), mode="reflect"), kh, groups=C)
    x = F.conv2d(F.pad(x, (0, 0, rad, rad), mode="reflect"), kv, groups=C)
    return x


def box_blur(img, r):
    C = img.shape[1]
    k = 2 * r + 1
    kh = torch.full((C, 1, 1, k), 1.0 / k)
    kv = torch.full((C, 1, k, 1), 1.0 / k)
    x = F.conv2d(F.pad(img, (r, r, 0, 0), mode="reflect"), kh, groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), kv, groups=C)
    return x


def luma(img):
    return (img * _W).sum(1, keepdim=True)


def guided_filter(img, r=8, eps=4e-4):
    """Self-guided filter (He et al. 2010), per channel, box radius r."""
    mean_I = box_blur(img, r)
    var_I = box_blur(img * img, r) - mean_I * mean_I
    a = var_I / (var_I + eps)
    b = mean_I - a * mean_I
    return box_blur(a, r) * img + box_blur(b, r)


# ---------------------------------------------------------------- components

def op_shift(img, dx=0.0, dy=0.0):
    if dx == 0.0 and dy == 0.0:
        return img
    _, _, H, W = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    grid = torch.stack([(xs - dx) / (W - 1) * 2 - 1,
                        (ys - dy) / (H - 1) * 2 - 1], -1).unsqueeze(0)
    return F.grid_sample(img, grid, mode="bicubic", padding_mode="reflection",
                         align_corners=True)


def op_guided(img, r=8, eps=4e-4, w=0.0):
    if w == 0.0:
        return img
    return img + w * (guided_filter(img, int(r), float(eps)) - img)


def op_unsharp(img, sigma=1.0, a=0.0):
    if a == 0.0:
        return img
    return img + a * (img - gaussian_blur(img, sigma))


def op_luma_unsharp(img, sigma=1.0, a=0.0):
    if a == 0.0:
        return img
    y = luma(img)
    return img + a * (y - gaussian_blur(y, sigma))


def op_chroma_blur(img, sigma=0.0):
    if sigma <= 0:
        return img
    y = luma(img)
    c = img - y
    cb = gaussian_blur(c, sigma)
    return y + (cb - luma(cb))


def op_tone(img, gain=(1.0, 1.0, 1.0), bias=(0.0, 0.0, 0.0)):
    if tuple(gain) == (1.0, 1.0, 1.0) and tuple(bias) == (0.0, 0.0, 0.0):
        return img
    g = torch.tensor(tuple(gain), dtype=torch.float32).view(1, 3, 1, 1)
    b = torch.tensor(tuple(bias), dtype=torch.float32).view(1, 3, 1, 1)
    return img * g + b


def op_gamma(img, g=1.0):
    return img if g == 1.0 else img.clamp(0, 1) ** g


def op_noise(img, sd=0.0, seed=1234):
    if sd <= 0:
        return img
    gen = torch.Generator().manual_seed(int(seed))
    return img + torch.randn(img.shape, generator=gen) * sd


# ---------------------------------------------------------------- pipeline

def apply_op(img, params=None):
    """img: float32 (1,3,H,W) in [0,1].  Returns the same shape, clipped to [0,1].
    Deterministic and side-effect free."""
    q = dict(DEFAULTS)
    q.update(params or {})
    x = img
    x = op_shift(x, float(q["shift_dx"]), float(q["shift_dy"]))
    x = op_guided(x, int(q["guided_r"]), float(q["guided_eps"]), float(q["guided_w"]))
    if int(q["us_luma_only"]):
        x = op_luma_unsharp(x, float(q["us_sigma"]), float(q["us_a"]))
    else:
        x = op_unsharp(x, float(q["us_sigma"]), float(q["us_a"]))
    x = op_chroma_blur(x, float(q["chroma_sigma"]))
    x = op_tone(x, tuple(q["gain"]), tuple(q["bias"]))
    x = op_gamma(x, float(q["gamma"]))
    x = op_noise(x, float(q["noise_sd"]), int(q["noise_seed"]))
    return x.clamp(0, 1)


# ---------------------------------------------------------------- io

def load(path):
    return torch.from_numpy(
        np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)


def to_u8(img):
    return (img.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def ship_encode_bytes(img):
    buf = io.BytesIO()
    Image.fromarray(to_u8(img)).save(buf, **SHIP_ENCODE)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--preset", default="b1", choices=sorted(PRESETS))
    ap.add_argument("--params", default=None,
                    help="JSON dict overriding the preset, e.g. '{\"us_a\": -0.2}'")
    ap.add_argument("--encode", default="png", choices=["png", "ship"],
                    help="png = lossless (default); ship = the exact submission JPEG profile")
    ap.add_argument("--ext", default=None, help="only process files with this extension")
    a = ap.parse_args()

    params = dict(PRESETS[a.preset])
    if a.params:
        params.update(json.loads(a.params))

    names = sorted(f for f in os.listdir(a.render_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))
                   and (a.ext is None or f.lower().endswith(a.ext.lower())))
    if not names:
        raise SystemExit(f"no images in {a.render_dir}")
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[op] preset={a.preset} params={json.dumps(params, sort_keys=True)} "
          f"encode={a.encode} n={len(names)}")

    for n in names:
        y = apply_op(load(os.path.join(a.render_dir, n)), params)
        stem = os.path.splitext(n)[0]
        if a.encode == "png":
            Image.fromarray(to_u8(y)).save(os.path.join(a.out_dir, stem + ".png"))
        else:
            with open(os.path.join(a.out_dir, stem + ".jpg"), "wb") as fh:
                fh.write(ship_encode_bytes(y))
    print(f"[op] wrote {len(names)} -> {a.out_dir}")


if __name__ == "__main__":
    main()
