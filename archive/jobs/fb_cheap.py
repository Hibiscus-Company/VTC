#!/usr/bin/env python
"""FREEBIE probe, stage 1: byte/pixel-level facts that need no GPU and no GT.

1. Do `optimize` / `progressive` change the DECODED PIXELS?  (they must not -- they are
   entropy-coding only.)  If not, they are a pure byte lever.
2. What do q99 / q98 / q97 / ss0 cost in BYTES on real shipped tower pixels?
3. How many uint8 round-trips does the shipped chain actually contain, and what is the
   RMS pixel difference between the float-fused chain and the shipped staged chain?
"""
import io, os, sys, json
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)


def enc(a_u8, **kw):
    b = io.BytesIO()
    Image.fromarray(a_u8).save(b, "JPEG", **kw)
    return b.getvalue()


def dec(buf):
    return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))


def main():
    # real shipped tower pixels
    src = "/mnt/d/avv/r27/tower_ens/HCM0421/png"
    stems = sorted(f for f in os.listdir(src) if f.endswith(".png"))[:6]
    print(f"source {src}  n={len(stems)}\n")

    profiles = {
        "SHIPPED q100 ss2 opt prog": dict(quality=100, subsampling=2, optimize=True, progressive=True),
        "q100 ss2 opt  noprog":      dict(quality=100, subsampling=2, optimize=True, progressive=False),
        "q100 ss2 noopt prog":       dict(quality=100, subsampling=2, optimize=False, progressive=True),
        "q100 ss2 noopt noprog":     dict(quality=100, subsampling=2, optimize=False, progressive=False),
        "q100 ss0 opt prog":         dict(quality=100, subsampling=0, optimize=True, progressive=True),
        "q100 ss1 opt prog":         dict(quality=100, subsampling=1, optimize=True, progressive=True),
        "q99  ss2 opt prog":         dict(quality=99, subsampling=2, optimize=True, progressive=True),
        "q98  ss2 opt prog":         dict(quality=98, subsampling=2, optimize=True, progressive=True),
        "q97  ss2 opt prog":         dict(quality=97, subsampling=2, optimize=True, progressive=True),
        "q95  ss2 opt prog":         dict(quality=95, subsampling=2, optimize=True, progressive=True),
    }
    tot = {k: 0 for k in profiles}
    ident = {k: True for k in profiles}
    ref_pix = None
    for s in stems:
        a = np.asarray(Image.open(os.path.join(src, s)).convert("RGB"))
        ref = dec(enc(a, **profiles["SHIPPED q100 ss2 opt prog"]))
        for k, kw in profiles.items():
            b = enc(a, **kw)
            tot[k] += len(b)
            if not np.array_equal(dec(b), ref):
                ident[k] = False
    n = len(stems)
    base = tot["SHIPPED q100 ss2 opt prog"]
    print(f"{'profile':<28} {'MB/60':>8} {'vs shipped':>11} {'pixels==shipped':>16}")
    for k in profiles:
        mb60 = tot[k] / n * 60 / 1e6
        print(f"{k:<28} {mb60:8.2f} {100*(tot[k]/base-1):+10.2f}% {str(ident[k]):>16}")

    # ---- how many uint8 round trips in the shipped chain
    print("\n--- staged-vs-fused pixel difference (HCM0421, 3 images) ---")
    sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
    print("r22 png_ens -> r25 png_ens -> r28 png_er -> r28 png : each stage writes uint8 PNG")
    for d in ["/mnt/d/avv/r22/tower_ens/HCM0421/png_ens",
              "/mnt/d/avv/r25/tower_ens/HCM0421/png_ens",
              "/mnt/d/avv/r28e_v2/tower_ens/HCM0421/png_er",
              "/mnt/d/avv/r28e_v2/tower_ens/HCM0421/png"]:
        ex = os.path.exists(d)
        nf = len([f for f in os.listdir(d) if f.endswith(".png")]) if ex else 0
        print(f"  {'OK ' if ex else 'MISSING'} {d}  n={nf}")


if __name__ == "__main__":
    main()
