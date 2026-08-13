#!/usr/bin/env python
"""D13: is there a ROLLING-SHUTTER signature in the residual flow?

Our unexplained residual is the +1.23 dB GEOM term: per-image LOCAL misregistration that
is neither the lens (a shared field, already harvested) nor global pose (oracle: -0.02 dB).

These are drone photos from a MOVING platform. A rolling shutter reads the sensor row by
row while the camera translates, which produces a displacement that:
  (1) varies LINEARLY with image row  (top scans at t=0, bottom at t=readout), and
  (2) points along the camera's VELOCITY, so it rotates as the drone orbits.
Property (2) is why the shared lens field cannot absorb it -- the field is fixed, this is
per-image. That makes it a perfect candidate for the GEOM residual.

Test, using flow we can compute for free:
  - per test image, fit flow(y) = a + b*y  -> the slope b IS the RS signature
  - estimate the drone's velocity direction at that view from the flight trajectory
  - correlate b's direction with the velocity direction

Strong correlation => rolling shutter, and gsplat models it natively
(RollingShutterType.ROLLING_TOP_TO_BOTTOM + viewmats_rs). No correlation => not RS.
"""
import argparse, csv, os, sys
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", default=os.path.expanduser("~/data/phase1/public_set"))
    ap.add_argument("--render", required=True, help="test renders (PNG)")
    ap.add_argument("--field", default="", help="shared lens field to remove first")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    S = os.path.join(args.root, args.scene)
    sp = os.path.join(S, "train", "sparse", "0")
    imd = os.path.join(S, "train", "images")
    gtd = os.path.join(S, "test", "images")

    # train trajectory: camera centres, ordered by filename (= capture order for DJI)
    tr = []
    for v in read_extrinsics_binary(os.path.join(sp, "images.bin")).values():
        if not os.path.exists(os.path.join(imd, v.name)):
            continue
        R = qvec2rotmat(v.qvec)
        tr.append((v.name, R, -R.T @ v.tvec))
    tr.sort(key=lambda t: t[0])
    trC = np.stack([c for _, _, c in tr])

    field = np.load(args.field) if args.field and os.path.exists(args.field) else None
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}

    rows = []
    with open(os.path.join(S, "test", "test_poses.csv"), newline="") as fh:
        recs = list(csv.DictReader(fh))[: args.limit]

    for r in recs:
        stem = os.path.splitext(r["image_name"])[0]
        rp = os.path.join(args.render, stem + ".png")
        if stem not in gt_by or not os.path.exists(rp):
            continue
        q = np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])])
        t = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])
        Rt = qvec2rotmat(q); C = -Rt.T @ t

        # drone velocity at this view: finite difference along the flight path,
        # using the two nearest train views IN CAPTURE ORDER
        j = int(np.linalg.norm(trC - C, axis=1).argmin())
        j0, j1 = max(0, j - 1), min(len(tr) - 1, j + 1)
        vel_w = trC[j1] - trC[j0]                      # world-space velocity direction
        if np.linalg.norm(vel_w) < 1e-9:
            continue
        # project the velocity into THIS view's image plane (camera x,y axes)
        vel_c = Rt @ vel_w
        vdir = vel_c[:2]
        if np.linalg.norm(vdir) < 1e-9:
            continue
        vdir = vdir / np.linalg.norm(vdir)

        img = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(os.path.join(gtd, gt_by[stem])).convert("RGB"),
                        dtype=np.float32) / 255.0
        if img.shape != gt.shape:
            continue
        H, W, _ = img.shape
        rg = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        gg = (cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        fl = np.clip(dis.calc(gg, rg, None), -6, 6)
        if field is not None:                       # remove the shared lens field first
            fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
            fl = fl - fu

        # RS signature: flow varies LINEARLY with row y
        ys = np.arange(H, dtype=np.float64)
        fx = fl[..., 0].mean(axis=1)                # mean dx per row
        fy = fl[..., 1].mean(axis=1)
        A = np.stack([ys / H, np.ones(H)], 1)
        bx = np.linalg.lstsq(A, fx, rcond=None)[0][0]   # px of dx across the full height
        by = np.linalg.lstsq(A, fy, rcond=None)[0][0]
        slope = np.array([bx, by])
        if np.linalg.norm(slope) < 1e-9:
            continue
        sdir = slope / np.linalg.norm(slope)
        rows.append((stem, vdir, sdir, float(np.linalg.norm(slope))))

    if len(rows) < 5:
        sys.exit("not enough views")

    V = np.stack([r[1] for r in rows])
    Sd = np.stack([r[2] for r in rows])
    mag = np.array([r[3] for r in rows])
    cos = (V * Sd).sum(axis=1)

    print(f"\n===== D13 rolling-shutter test: {args.scene} ({len(rows)} views) =====")
    print(f"row-linear flow slope magnitude: mean {mag.mean():.3f} px   "
          f"median {np.median(mag):.3f}   max {mag.max():.3f}   (across the full frame height)")
    print(f"\nalignment of the row-slope with the DRONE VELOCITY direction:")
    print(f"  mean cos = {cos.mean():+.4f}   |mean cos| = {abs(cos.mean()):.4f}")
    print(f"  fraction aligned  (cos > +0.5): {100*(cos > 0.5).mean():5.1f}%")
    print(f"  fraction opposed  (cos < -0.5): {100*(cos < -0.5).mean():5.1f}%")
    print(f"  fraction orthogonal (|cos|<0.5): {100*(abs(cos) < 0.5).mean():5.1f}%")
    print("\n  => |mean cos| near 1 (aligned OR opposed -- sign depends on readout")
    print("     direction) with slope of a few px = ROLLING SHUTTER, and gsplat models it.")
    print("     |mean cos| near 0 = the residual is NOT velocity-coupled; RS is dead.")


if __name__ == "__main__":
    main()
