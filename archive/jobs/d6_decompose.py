#!/usr/bin/env python
"""D6: decompose D5's +2.13 dB flow oracle into LENS / POSE / GEOMETRY.

D5 proved the residual is local misregistration (dense flow recovers +2.13 dB;
a global shift recovers nothing). Three sources, in increasing cost-to-fix:

  1. LENS   - a SHARED radial displacement curve c(r), one curve for ALL images.
              A lens is a lens: this is fittable from TRAIN photos alone, so it
              is fully legal, and it is exactly what k2/k3/tangential terms model.
              We currently hand gsplat radial_coeffs=[k1,0,0,0,0,0] -- ONE term,
              from COLMAP SIMPLE_RADIAL, held CONSTANT during training.
  2. POSE   - a per-image homography on top. Bounds what pose refinement buys.
  3. GEOM   - whatever full dense flow recovers beyond 1+2. True 3D error;
              needs better reconstruction (densification / MVS seeding).

Cumulative, so each row's delta is that source's MARGINAL contribution.
Oracle-fit against PUBLIC test GT = diagnosis only (same footing as D1/D5).
"""
import argparse, os, sys
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None
NR = 32  # radial rings for the shared lens curve


def psnr_db(mse):
    return 10.0 * np.log10(1.0 / max(float(mse), 1e-12))


def mp(lst):
    return float(np.mean([psnr_db(m) for m in lst]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gt_by_stem = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt)}
    files = sorted(f for f in os.listdir(args.render)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[: args.limit]

    # ---------- pass 1: dense flow, accumulate the SHARED radial curve ----------
    cache = []
    rad_num = np.zeros(NR); rad_den = np.zeros(NR); tan_num = np.zeros(NR)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    for f in files:
        stem = os.path.splitext(f)[0]
        if stem not in gt_by_stem:
            continue
        r = np.asarray(Image.open(os.path.join(args.render, f)).convert("RGB"),
                       dtype=np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(args.gt, gt_by_stem[stem])).convert("RGB"),
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
        rmax = rr.max()
        ux, uy = dx / (rr + 1e-6), dy / (rr + 1e-6)          # unit radial
        f_rad = flow[..., 0] * ux + flow[..., 1] * uy         # radial component
        f_tan = -flow[..., 0] * uy + flow[..., 1] * ux        # tangential component

        b = np.clip((rr / rmax * NR).astype(np.int32), 0, NR - 1)
        np.add.at(rad_num, b.ravel(), f_rad.ravel())
        np.add.at(tan_num, b.ravel(), np.abs(f_tan).ravel())
        np.add.at(rad_den, b.ravel(), 1.0)

        cache.append((r, g, flow, xx, yy, ux, uy, rr, rmax, b))

    if not cache:
        sys.exit("no images matched")

    curve = rad_num / np.maximum(rad_den, 1)      # px, +ve = outward
    tan_mag = tan_num / np.maximum(rad_den, 1)

    # ---------- pass 2: cumulative oracles ----------
    base, o_lens, o_lens_h, o_flow = [], [], [], []
    for (r, g, flow, xx, yy, ux, uy, rr, rmax, b) in cache:
        Hh, Ww, _ = r.shape
        base.append(((r - g) ** 2).mean())

        # (1) SHARED radial lens warp
        c_pix = curve[b]                              # px displacement at each pixel
        lx = xx + c_pix * ux
        ly = yy + c_pix * uy
        r_lens = cv2.remap(r, lx.astype(np.float32), ly.astype(np.float32),
                           cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        o_lens.append(((r_lens - g) ** 2).mean())

        # (2) + per-image homography on the LENS-CORRECTED render (residual flow)
        rg2 = (cv2.cvtColor(r_lens, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        gg2 = (cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        fl2 = np.clip(dis.calc(gg2, rg2, None), -6, 6)
        step = max(1, min(Hh, Ww) // 60)
        ys, xs = np.mgrid[0:Hh:step, 0:Ww:step]
        src = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
        dst = src + np.stack([fl2[::step, ::step, 0].ravel(),
                              fl2[::step, ::step, 1].ravel()], 1)
        Hm, _ = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
        if Hm is None:
            o_lens_h.append(o_lens[-1])
        else:
            r_h = cv2.warpPerspective(r_lens, np.linalg.inv(Hm), (Ww, Hh),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REFLECT)
            o_lens_h.append(min(((r_h - g) ** 2).mean(), o_lens[-1]))

        # (3) full dense flow (everything)
        mapx = (xx + flow[..., 0]).astype(np.float32)
        mapy = (yy + flow[..., 1]).astype(np.float32)
        r_fl = cv2.remap(r, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        o_flow.append(((r_fl - g) ** 2).mean())

    b0 = mp(base)
    print(f"\n===== D6 {args.tag}  ({len(base)} images) =====")
    print(f"{'baseline':38s} {b0:7.4f} dB")
    prev = b0
    for lab, lst in (("+ SHARED radial lens warp  (LENS)", o_lens),
                     ("+ per-image homography     (POSE)", o_lens_h),
                     ("  full dense flow          (ALL) ", o_flow)):
        v = mp(lst)
        print(f"{lab:38s} {v:7.4f} dB   cum {v - b0:+.4f}   "
              f"marginal {v - prev:+.4f} dB = {0.6 * (v - prev):+.4f} pts")
        prev = v
    print(f"{'  GEOM residual (ALL - POSE)':38s} {'':7s}      "
          f"          {mp(o_flow) - mp(o_lens_h):+.4f} dB")

    print("\n-- SHARED radial displacement curve (px, + = outward) --")
    print("   r/rmax   disp_px   |tangential|   (a smooth monotone curve = "
          "unmodeled lens distortion)")
    for i in range(0, NR, 2):
        rn = (i + 0.5) / NR
        print(f"   {rn:5.3f}   {curve[i]:+7.3f}      {tan_mag[i]:6.3f}")


if __name__ == "__main__":
    main()
