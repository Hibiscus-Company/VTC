#!/usr/bin/env python
"""D7: is the radial curve a real LENS property, or per-image overfitting?

D6's shared radial warp recovered +0.669 dB -- but that curve was fit on the
same test images it was scored on, with 32 free ring values. That can cheat.

A lens is a property of the CAMERA, so a curve fit on one set of images must
transfer to images it never saw. Test: 2-fold cross-validation.
    fit curve on fold A  -> apply to fold B (held out) -> measure dB
    fit curve on fold B  -> apply to fold A (held out) -> measure dB
If held-out gain ~= in-sample gain, it is a lens and it will transfer to the
private set when fit on TRAIN photos. If held-out gain collapses to ~0, the
curve is noise and the whole idea dies here.

Also fits an OpenCV radial polynomial (k1,k2,k3) to the curve, since that is
what actually has to go into gsplat's radial_coeffs at retrain time -- a curve
that no polynomial can express is not implementable.
"""
import argparse, os, sys
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None
NR = 32


def psnr_db(mse):
    return 10.0 * np.log10(1.0 / max(float(mse), 1e-12))


def mp(lst):
    return float(np.mean([psnr_db(m) for m in lst]))


def build(files, gt_by_stem, rdir, gdir):
    """dense flow + radial binning for each image"""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = []
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem not in gt_by_stem:
            continue
        r = np.asarray(Image.open(os.path.join(rdir, f)).convert("RGB"),
                       dtype=np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(gdir, gt_by_stem[stem])).convert("RGB"),
                       dtype=np.float32) / 255.0
        if r.shape != g.shape:
            continue
        H, W, _ = r.shape
        rg = (cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        gg = (cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        flow = np.clip(dis.calc(gg, rg, None), -6, 6)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        dy, dx = yy - cy, xx - cx
        rr = np.sqrt(dx * dx + dy * dy)
        rmax = float(rr.max())
        ux, uy = dx / (rr + 1e-6), dy / (rr + 1e-6)
        f_rad = flow[..., 0] * ux + flow[..., 1] * uy
        b = np.clip((rr / rmax * NR).astype(np.int32), 0, NR - 1)
        out.append(dict(stem=stem, r=r, g=g, xx=xx, yy=yy, ux=ux, uy=uy,
                        rr=rr, rmax=rmax, b=b, f_rad=f_rad))
    return out


def fit_curve(items):
    num = np.zeros(NR); den = np.zeros(NR)
    for it in items:
        np.add.at(num, it["b"].ravel(), it["f_rad"].ravel())
        np.add.at(den, it["b"].ravel(), 1.0)
    return num / np.maximum(den, 1)


def apply_curve(it, curve):
    c = curve[it["b"]]
    lx = (it["xx"] + c * it["ux"]).astype(np.float32)
    ly = (it["yy"] + c * it["uy"]).astype(np.float32)
    w = cv2.remap(it["r"], lx, ly, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return ((w - it["g"]) ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gt_by_stem = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt)}
    files = sorted(f for f in os.listdir(args.render)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[: args.limit]
    items = build(files, gt_by_stem, args.render, args.gt)
    if len(items) < 4:
        sys.exit("need >=4 images")

    A = items[0::2]   # interleaved so both folds span the flight
    B = items[1::2]
    cA, cB, cAll = fit_curve(A), fit_curve(B), fit_curve(items)

    base = [((it["r"] - it["g"]) ** 2).mean() for it in items]
    b0 = mp(base)

    ins = [apply_curve(it, cAll) for it in items]                 # in-sample (D6)
    hold = [apply_curve(it, cB) for it in A] + \
           [apply_curve(it, cA) for it in B]                      # HELD OUT

    print(f"\n===== D7 {args.tag}  ({len(items)} imgs, folds {len(A)}/{len(B)}) =====")
    print(f"baseline                     {b0:7.4f} dB")
    v = mp(ins)
    print(f"curve fit on ALL (in-sample) {v:7.4f} dB   ({v - b0:+.4f} dB)  <- D6, can cheat")
    v = mp(hold)
    print(f"curve fit on OTHER fold      {v:7.4f} dB   ({v - b0:+.4f} dB = "
          f"{0.6 * (v - b0):+.4f} pts)  <- HELD OUT, honest")

    # agreement between the two independently-fit curves
    rr = np.corrcoef(cA, cB)[0, 1]
    print(f"\nfold-A vs fold-B curve correlation: r = {rr:.4f}  "
          f"(max|A-B| = {np.abs(cA - cB).max():.3f} px)")

    # ---- can an OpenCV radial polynomial express it? ----
    # gsplat applies r_d = r_u*(1 + k1 r^2 + k2 r^4 + k3 r^6); the DISPLACEMENT
    # in px at radius r is then r*(k1 r^2 + k2 r^4 + k3 r^6) in normalized units.
    rn = (np.arange(NR) + 0.5) / NR                 # normalized radius 0..1
    valid = ~np.isnan(cAll)
    Vm = np.stack([rn ** 3, rn ** 5, rn ** 7], 1)[valid]
    coef, *_ = np.linalg.lstsq(Vm, cAll[valid], rcond=None)
    pred = Vm @ coef
    resid = cAll[valid] - pred
    print(f"\npolynomial fit d(r) = a*r^3 + b*r^5 + c*r^7  (px):")
    print(f"   a={coef[0]:+.4f}  b={coef[1]:+.4f}  c={coef[2]:+.4f}   "
          f"rms residual {np.sqrt((resid ** 2).mean()):.4f} px "
          f"(curve rms {np.sqrt((cAll[valid] ** 2).mean()):.4f} px)")
    poly_curve = np.zeros(NR); poly_curve[valid] = pred
    v = mp([apply_curve(it, poly_curve) for it in items])
    print(f"   polynomial-only warp:     {v:7.4f} dB   ({v - b0:+.4f} dB)  "
          f"<- what radial_coeffs CAN express")

    print("\n-- curves (px) --")
    print("   r      fitA     fitB     poly")
    for i in range(0, NR, 3):
        print(f"  {rn[i]:.3f}  {cA[i]:+7.3f}  {cB[i]:+7.3f}  {poly_curve[i]:+7.3f}")


if __name__ == "__main__":
    main()
