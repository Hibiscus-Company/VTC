#!/usr/bin/env python
"""D4: where does the MSE actually live?

The score averages PSNR-in-dB PER IMAGE, so the mean is dragged by the worst
views, not by the average one. And within an image, MSE is dominated by the
worst pixels. Good-LPIPS/bad-PSNR is the signature of concentrated error.
This asks: concentrated in which images, and in which pixels of them?

CPU only (numpy+PIL) so it can run while both GPUs train.
"""
import argparse, os, sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def psnr_db(mse):
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gt_by_stem = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt)}
    rends = sorted(f for f in os.listdir(args.render)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))

    rows = []          # (stem, psnr, mse)
    se_frac_top = np.zeros(4)   # SE fraction from top 1/5/10/25% pixels
    lum_num = np.zeros(4); lum_den = np.zeros(4)   # SE and pixel count by GT luminance quartile band
    border_se = 0.0; border_n = 0; center_se = 0.0; center_n = 0
    sky_se = 0.0; sky_n = 0; nonsky_se = 0.0; nonsky_n = 0

    for f in rends:
        stem = os.path.splitext(f)[0]
        if stem not in gt_by_stem:
            continue
        r = np.asarray(Image.open(os.path.join(args.render, f)).convert("RGB"),
                       dtype=np.float64) / 255.0
        g = np.asarray(Image.open(os.path.join(args.gt, gt_by_stem[stem])).convert("RGB"),
                       dtype=np.float64) / 255.0
        if r.shape != g.shape:
            print(f"  !! size mismatch {stem}: {r.shape} vs {g.shape}", file=sys.stderr)
            continue

        se = ((r - g) ** 2).mean(axis=2)        # H,W  per-pixel mean-over-channel SE
        mse = se.mean()
        rows.append((stem, psnr_db(mse), mse))

        # --- pixel-level concentration
        flat = np.sort(se.ravel())[::-1]
        tot = flat.sum() + 1e-12
        n = flat.size
        for i, q in enumerate((0.01, 0.05, 0.10, 0.25)):
            se_frac_top[i] += flat[: max(1, int(q * n))].sum() / tot

        # --- by GT luminance (sky is bright; structure is dark)
        lum = g.mean(axis=2)
        edges = [0.0, 0.25, 0.5, 0.75, 1.01]
        for i in range(4):
            m = (lum >= edges[i]) & (lum < edges[i + 1])
            lum_num[i] += se[m].sum(); lum_den[i] += m.sum()

        # --- border vs center (distortion / extrapolation artifacts live at edges)
        H, W = se.shape
        bh, bw = int(0.1 * H), int(0.1 * W)
        cmask = np.zeros_like(se, dtype=bool)
        cmask[bh:H - bh, bw:W - bw] = True
        center_se += se[cmask].sum(); center_n += cmask.sum()
        border_se += se[~cmask].sum(); border_n += (~cmask).sum()

        # --- crude sky proxy: bright AND low-saturation
        mx = g.max(axis=2); mn = g.min(axis=2)
        sat = (mx - mn) / (mx + 1e-6)
        sky = (lum > 0.55) & (sat < 0.25)
        sky_se += se[sky].sum(); sky_n += sky.sum()
        nonsky_se += se[~sky].sum(); nonsky_n += (~sky).sum()

    if not rows:
        sys.exit("no matched images")

    rows.sort(key=lambda t: t[1])
    ps = np.array([r[1] for r in rows])
    k = len(rows)
    print(f"\n===== D4 {args.tag}  ({k} images) =====")
    print(f"mean PSNR {ps.mean():.4f} dB   median {np.median(ps):.4f}   "
          f"std {ps.std():.4f}   min {ps.min():.2f}   max {ps.max():.2f}")

    print("\n-- worst 8 images --")
    for stem, p, _ in rows[:8]:
        print(f"   {stem:28s} {p:7.3f} dB   ({p - np.median(ps):+.2f} vs median)")
    print("-- best 3 --")
    for stem, p, _ in rows[-3:]:
        print(f"   {stem:28s} {p:7.3f} dB")

    # how much mean-PSNR is the bad tail costing us?
    med = np.median(ps)
    print("\n-- tail cost: mean PSNR if worst-N images were lifted to the median --")
    for nfix in (1, 3, 5, 10):
        if nfix >= k:
            break
        lifted = ps.copy()
        lifted[:nfix] = np.maximum(lifted[:nfix], med)
        gain = lifted.mean() - ps.mean()
        print(f"   fix worst {nfix:2d}: {lifted.mean():.4f} dB  "
              f"({gain:+.4f} dB = {0.6 * gain:+.4f} score pts)")

    print("\n-- pixel concentration (mean over images: share of an image's SE) --")
    for q, v in zip((1, 5, 10, 25), se_frac_top / k):
        print(f"   top {q:2d}% worst pixels carry {100 * v:5.1f}% of the squared error")

    print("\n-- SE density by GT luminance band (relative to image mean = 1.0) --")
    glob_mse = np.average([r[2] for r in rows])
    for i, lab in enumerate(("dark 0.00-0.25", "mid  0.25-0.50",
                             "lite 0.50-0.75", "brgt 0.75-1.00")):
        if lum_den[i] > 0:
            d = lum_num[i] / lum_den[i]
            print(f"   {lab}: SE/px {d:.6f}  ({d / glob_mse:5.2f}x)  "
                  f"{100 * lum_den[i] / lum_den.sum():5.1f}% of pixels")

    print("\n-- spatial / sky --")
    print(f"   border 10%: SE/px {border_se / border_n:.6f}  "
          f"({(border_se / border_n) / glob_mse:.2f}x)  {100 * border_n / (border_n + center_n):.0f}% of px")
    print(f"   center    : SE/px {center_se / center_n:.6f}  "
          f"({(center_se / center_n) / glob_mse:.2f}x)")
    if sky_n > 0:
        print(f"   sky-proxy : SE/px {sky_se / sky_n:.6f}  "
              f"({(sky_se / sky_n) / glob_mse:.2f}x)  {100 * sky_n / (sky_n + nonsky_n):.1f}% of px  "
              f"-> {100 * sky_se / (sky_se + nonsky_se):.1f}% of ALL squared error")
        print(f"   non-sky   : SE/px {nonsky_se / nonsky_n:.6f}  "
              f"({(nonsky_se / nonsky_n) / glob_mse:.2f}x)")


if __name__ == "__main__":
    main()
