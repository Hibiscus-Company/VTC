#!/usr/bin/env python
"""D9: THE decisive one. Fit the displacement field on TRAIN, apply to TEST.

D8 got +0.977 dB held-out, but that field was fit on TEST renders vs TEST GT.
On the private set there IS no test GT -- the field must be fit against TRAIN
photos, which is the only ground truth we are ever allowed to touch there.

The risk is concrete: the model was OPTIMIZED on those train views, so its
gaussians may already have contorted to absorb the lens error there. If so the
train residual is small / differently-shaped, the train-fitted field will not
match the test-fitted one, and the whole idea collapses on private.

  fit field on (train renders, train photos)  -- fully legal, no test GT
  apply it to (test renders)                  -- measure the real, bankable dB

Also compares the train-fitted field against the test-fitted field saved by D8:
if the bias is a genuine camera/pipeline property the two must agree.
"""
import argparse, os, sys
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None
DS = 8


def psnr_db(mse):
    return 10.0 * np.log10(1.0 / max(float(mse), 1e-12))


def mp(lst):
    return float(np.mean([psnr_db(m) for m in lst]))


def pairs(rdir, gdir, limit=0):
    gt = {os.path.splitext(f)[0]: f for f in os.listdir(gdir)}
    fs = sorted(f for f in os.listdir(rdir)
                if f.lower().endswith((".png", ".jpg", ".jpeg")))
    out = [(os.path.join(rdir, f), os.path.join(gdir, gt[os.path.splitext(f)[0]]))
           for f in fs if os.path.splitext(f)[0] in gt]
    return out[:limit] if limit else out


def fit_field(pp, tag):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    acc, n = None, 0
    for rp, gp in pp:
        r = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0
        if r.shape != g.shape:
            continue
        H, W, _ = r.shape
        rg = (cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        gg = (cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        fl = np.clip(dis.calc(gg, rg, None), -6, 6)
        s = cv2.resize(fl, (W // DS, H // DS), interpolation=cv2.INTER_AREA)
        acc = s if acc is None else acc + s
        n += 1
    print(f"  [{tag}] fit on {n} pairs")
    return acc / max(n, 1)


def score(pp, field, tag):
    base, corr = [], []
    for rp, gp in pp:
        r = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0
        if r.shape != g.shape:
            continue
        H, W, _ = r.shape
        base.append(((r - g) ** 2).mean())
        if field is None:
            continue
        fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        w = cv2.remap(r, (xx + fu[..., 0]).astype(np.float32),
                      (yy + fu[..., 1]).astype(np.float32),
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        corr.append(((w - g) ** 2).mean())
    b0 = mp(base)
    if not corr:
        print(f"{tag:34s} {b0:7.4f} dB  (baseline)")
        return b0
    v = mp(corr)
    print(f"{tag:34s} {v:7.4f} dB   base {b0:7.4f}   "
          f"({v - b0:+.4f} dB = {0.6 * (v - b0):+.4f} score pts)")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_render", required=True)
    ap.add_argument("--train_gt", required=True)
    ap.add_argument("--test_render_single", required=True)
    ap.add_argument("--test_render_ens", default="")
    ap.add_argument("--test_gt", required=True)
    ap.add_argument("--test_field", default="", help="D8's test-fitted field, for comparison")
    ap.add_argument("--save_field", default="")
    args = ap.parse_args()

    tr = pairs(args.train_render, args.train_gt)
    te_s = pairs(args.test_render_single, args.test_gt)
    print(f"train pairs {len(tr)}   test pairs {len(te_s)}")

    print("\n--- fitting field on TRAIN (legal: no test GT) ---")
    f_train = fit_field(tr, "train")

    mag = np.linalg.norm(f_train, axis=2)
    print(f"  train field: mean |d| {mag.mean():.3f} px   max {mag.max():.2f} px")

    if args.test_field and os.path.exists(args.test_field):
        f_test = np.load(args.test_field)
        if f_test.shape == f_train.shape:
            mt = np.linalg.norm(f_test, axis=2)
            cx = np.corrcoef(f_train[..., 0].ravel(), f_test[..., 0].ravel())[0, 1]
            cy = np.corrcoef(f_train[..., 1].ravel(), f_test[..., 1].ravel())[0, 1]
            print(f"  test  field: mean |d| {mt.mean():.3f} px   max {mt.max():.2f} px")
            print(f"  TRAIN-vs-TEST field agreement:  corr dx {cx:+.4f}  dy {cy:+.4f}   "
                  f"mean|diff| {np.linalg.norm(f_train - f_test, axis=2).mean():.3f} px")
            print(f"  magnitude ratio train/test: {mag.mean() / max(mt.mean(), 1e-6):.3f}"
                  "   (<<1 => model absorbed the bias on train views; idea is in trouble)")

    print("\n--- applying the TRAIN-fitted field ---")
    score(tr, None, "train renders (baseline)")
    score(tr, f_train, "train renders + train field")
    print()
    score(te_s, f_train, "TEST single + TRAIN field")
    if args.test_render_ens:
        te_e = pairs(args.test_render_ens, args.test_gt)
        score(te_e, f_train, "TEST ensemble + TRAIN field")

    if args.save_field:
        np.save(args.save_field, f_train)
        print(f"\nsaved train-fitted field -> {args.save_field}")


if __name__ == "__main__":
    main()
