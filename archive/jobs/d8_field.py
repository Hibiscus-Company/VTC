#!/usr/bin/env python
"""D8: a SHARED DENSE 2D displacement field (not just a radial curve).

D7 proved the radial component of our render->GT misregistration is a stable,
image-independent property: two disjoint folds agreed to 0.08 px (r=0.9977) and
the held-out gain (+0.788 dB) equalled the in-sample gain. That is a systematic
geometric bias in our rendering pipeline, not noise.

A radial curve is only the rotationally-symmetric part of that bias. The full
bias is a 2D field: lens tangential + thin-prism + principal-point offset + any
systematic pipeline skew all live outside the radial component. So fit the mean
flow field itself, and cross-validate it the same way.

  fit mean field on fold A -> apply to fold B (held out), and vice versa.

Held-out gain is what we can actually bank. The real fix fits this field on
TRAIN renders vs TRAIN photos (no test GT anywhere) and applies it at render
time; this script only establishes the size of the prize.
"""
import argparse, os, sys
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None
DS = 8  # the field is smooth; fit it downsampled, upsample to apply


def psnr_db(mse):
    return 10.0 * np.log10(1.0 / max(float(mse), 1e-12))


def mp(lst):
    return float(np.mean([psnr_db(m) for m in lst]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save_field", default="")
    args = ap.parse_args()

    gt_by_stem = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt)}
    files = sorted(f for f in os.listdir(args.render)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[: args.limit]

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    small = []   # downsampled flow per image
    meta = []
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem not in gt_by_stem:
            continue
        rp = os.path.join(args.render, f)
        gp = os.path.join(args.gt, gt_by_stem[stem])
        r = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0
        if r.shape != g.shape:
            continue
        H, W, _ = r.shape
        rg = (cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        gg = (cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        flow = np.clip(dis.calc(gg, rg, None), -6, 6)
        small.append(cv2.resize(flow, (W // DS, H // DS), interpolation=cv2.INTER_AREA))
        meta.append((rp, gp, H, W))

    n = len(small)
    if n < 4:
        sys.exit("need >=4 images")
    H, W = meta[0][2], meta[0][3]
    print(f"[{n} images @ {W}x{H}, field fit at {W//DS}x{H//DS}]")

    A = list(range(0, n, 2))
    B = list(range(1, n, 2))
    fA = np.mean([small[i] for i in A], axis=0)
    fB = np.mean([small[i] for i in B], axis=0)
    fAll = np.mean(small, axis=0)

    def apply_field(idx, field):
        rp, gp, Hh, Ww = meta[idx]
        r = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0
        fu = cv2.resize(field, (Ww, Hh), interpolation=cv2.INTER_CUBIC)
        yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
        w = cv2.remap(r, (xx + fu[..., 0]).astype(np.float32),
                      (yy + fu[..., 1]).astype(np.float32),
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        return ((w - g) ** 2).mean(), ((r - g) ** 2).mean()

    base, ins, hold = [], [], []
    for i in range(n):
        m_ins, m_base = apply_field(i, fAll)
        m_hold, _ = apply_field(i, fB if i in A else fA)
        base.append(m_base); ins.append(m_ins); hold.append(m_hold)

    b0 = mp(base)
    print(f"\n===== D8 {args.tag} =====")
    print(f"baseline                       {b0:7.4f} dB")
    v = mp(ins)
    print(f"shared 2D field (in-sample)    {v:7.4f} dB   ({v - b0:+.4f} dB)")
    v = mp(hold)
    print(f"shared 2D field (HELD OUT)     {v:7.4f} dB   ({v - b0:+.4f} dB = "
          f"{0.6 * (v - b0):+.4f} score pts)")

    # how well do the two independently-fit fields agree?
    d = np.linalg.norm(fA - fB, axis=2)
    mag = np.linalg.norm(fAll, axis=2)
    ca = np.corrcoef(fA[..., 0].ravel(), fB[..., 0].ravel())[0, 1]
    cb = np.corrcoef(fA[..., 1].ravel(), fB[..., 1].ravel())[0, 1]
    print(f"\nfold agreement: corr dx {ca:.4f}  dy {cb:.4f}   "
          f"mean|fA-fB| {d.mean():.3f} px   mean|field| {mag.mean():.3f} px "
          f"(max {mag.max():.2f})")

    if args.save_field:
        np.save(args.save_field, fAll)
        print(f"saved mean field -> {args.save_field}  {fAll.shape}")


if __name__ == "__main__":
    main()
