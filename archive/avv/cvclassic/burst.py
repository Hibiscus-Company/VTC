#!/usr/bin/env python
"""GROUPWISE SUB-PIXEL ALIGN -> MERGE  (classical burst-fusion merge).

Baseline : mean_i m_i                    (what we ship)
Treatment: mean_i  Warp(m_i -> consensus geometry)

Each member differs from the consensus by a small (~0.17px) pseudo-random
displacement field. Averaging displaced copies convolves the result with the
displacement distribution => self-inflicted blur. Warping every member onto the
consensus frame first removes that convolution while KEEPING the noise averaging.

Writes both variants so scripts/eval_score.py can score them against real GT.
"""
import os, sys, argparse
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def gray8(a):
    return np.clip(0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2], 0, 255).astype(np.uint8)


def warp(img, fl):
    H, W, _ = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, (xx + fl[..., 0]).astype(np.float32),
                     (yy + fl[..., 1]).astype(np.float32),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out_plain", required=True)
    ap.add_argument("--out_align", required=True)
    ap.add_argument("--iters", type=int, default=2, help="consensus refinement passes")
    ap.add_argument("--clip", type=float, default=1.5, help="max |displacement| px")
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--finest", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_plain, exist_ok=True)
    os.makedirs(args.out_align, exist_ok=True)
    stems = None
    for d in args.dirs:
        s = {os.path.splitext(f)[0] for f in os.listdir(d)
             if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")}
        stems = s if stems is None else (stems & s)
    stems = sorted(stems)
    print(f"{len(stems)} frames x {len(args.dirs)} members")

    preset = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
              "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
              "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[args.preset]

    for s in stems:
        ms = []
        for d in args.dirs:
            for e in (".png", ".jpg", ".JPG", ".jpeg"):
                p = os.path.join(d, s + e)
                if os.path.exists(p):
                    ms.append(load(p)); break
        assert len(ms) == len(args.dirs), s
        M = np.stack(ms)
        plain = M.mean(0)

        ref = plain
        for _ in range(args.iters):
            dis = cv2.DISOpticalFlow_create(preset)
            dis.setFinestScale(args.finest)
            r8 = gray8(ref)
            acc = None
            for m in ms:
                # flow ref->member: sampling the member at (x+fl) lands it on ref's grid
                fl = np.clip(dis.calc(r8, gray8(m), None), -args.clip, args.clip)
                w = warp(m, fl)
                acc = w if acc is None else acc + w
            ref = acc / len(ms)

        Image.fromarray(np.clip(plain + 0.5, 0, 255).astype(np.uint8)).save(
            os.path.join(args.out_plain, s + ".png"))
        Image.fromarray(np.clip(ref + 0.5, 0, 255).astype(np.uint8)).save(
            os.path.join(args.out_align, s + ".png"))
    print("done")


if __name__ == "__main__":
    main()
