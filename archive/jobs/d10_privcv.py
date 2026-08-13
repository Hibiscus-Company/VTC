#!/usr/bin/env python
"""D10: held-out sanity check of each PRIVATE scene's field, using TRAIN data only.

HNI0131's fitted field has max 5.69 px vs 0.99-1.68 px everywhere else -- consistent
with its degenerate k1=-0.115, but far OUTSIDE the regime the 5/5 public validation
covered (public maxed at 1.9 px), and near the +-6 px flow clip where DIS gets
unreliable. No public scene has negative k1, so there is nothing to validate it against.

Applying the field back to the very train renders it was fit on is circular and would
always look good. So: 2-fold CV *within the train views*.
    fit on train fold A -> apply to train fold B (never seen by that fit) -> dB
A field that is real transfers to held-out views. A field that is fitting flow noise
does not. Uses TRAIN photos only -- no test GT anywhere, so it is legal on private.

Reports the clip-saturation rate too: if a large share of pixels hit +-6 px, the field
is truncated and the fit is not trustworthy at any magnitude.
"""
import argparse, os, sys
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None
DS = 8
CLIP = 6.0


def psnr_db(mse):
    return 10.0 * np.log10(1.0 / max(float(mse), 1e-12))


def mp(lst):
    return float(np.mean([psnr_db(m) for m in lst]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True, help="TRAIN renders")
    ap.add_argument("--gt_dir", required=True, help="TRAIN photos")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gt_abs = os.path.abspath(args.gt_dir)
    assert os.sep + "test" + os.sep not in gt_abs + os.sep, "TRAIN photos only"

    gt_by_stem = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir)}
    files = sorted(f for f in os.listdir(args.render_dir) if f.lower().endswith(".png"))

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    small, imgs, sat = [], [], []
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem not in gt_by_stem:
            continue
        r = np.asarray(Image.open(os.path.join(args.render_dir, f)).convert("RGB"),
                       dtype=np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(args.gt_dir, gt_by_stem[stem])).convert("RGB"),
                       dtype=np.float32) / 255.0
        if r.shape != g.shape:
            continue
        H, W, _ = r.shape
        rg = (cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        gg = (cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        raw = dis.calc(gg, rg, None)
        sat.append(float((np.abs(raw) >= CLIP).mean()))
        fl = np.clip(raw, -CLIP, CLIP)
        small.append(cv2.resize(fl, (W // DS, H // DS), interpolation=cv2.INTER_AREA))
        imgs.append((r, g))

    n = len(imgs)
    if n < 4:
        sys.exit("need >=4 train pairs")

    A, B = list(range(0, n, 2)), list(range(1, n, 2))
    fA = np.mean([small[i] for i in A], axis=0)
    fB = np.mean([small[i] for i in B], axis=0)

    def apply_to(i, field):
        r, g = imgs[i]
        H, W, _ = r.shape
        fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        w = cv2.remap(r, (xx + fu[..., 0]).astype(np.float32),
                      (yy + fu[..., 1]).astype(np.float32),
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        return ((w - g) ** 2).mean()

    base = [((r - g) ** 2).mean() for r, g in imgs]
    hold = [apply_to(i, fB) for i in A] + [apply_to(i, fA) for i in B]

    b0, h0 = mp(base), mp(hold)
    magA, magB = np.linalg.norm(fA, axis=2), np.linalg.norm(fB, axis=2)
    cx = np.corrcoef(fA[..., 0].ravel(), fB[..., 0].ravel())[0, 1]
    cy = np.corrcoef(fA[..., 1].ravel(), fB[..., 1].ravel())[0, 1]

    verdict = "PASS" if (h0 - b0) > 0.05 and min(cx, cy) > 0.7 else "*** SUSPECT ***"
    print(f"{args.tag:10s} n={n:3d}  base {b0:7.4f} -> HELD-OUT {h0:7.4f} dB  "
          f"({h0 - b0:+.4f})  foldcorr dx {cx:+.3f} dy {cy:+.3f}  "
          f"max|d| {max(magA.max(), magB.max()):.2f}px  "
          f"clip-sat {100 * np.mean(sat):.2f}%  {verdict}")


if __name__ == "__main__":
    main()
