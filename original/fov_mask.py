#!/usr/bin/env python
"""Geometric validity mask: where were the FastGS members actually SUPERVISED?

The FastGS gate members train on UNDISTORTED images built with the SAME K. For the
two negative-k1 scenes (HNI0131/HNI0265, k1 = -0.115) the undistorted preimage of the
frame is LARGER than the canvas, so undistortion CROPS the outer ring: a test pixel
whose undistorted coord lands outside WxH was never covered by any training pixel.

Measured: 11.90% of the frame on those two scenes. 0.00% on every other scene.

The ensemble nonetheless gives the gates 0.4 of the weight EVERYWHERE, including that
ring -- i.e. 40% of the signal in 12% of the image comes from models that are purely
extrapolating there. The UT members (trained in distorted space on the ORIGINAL photos)
have real supervision across the whole frame and should carry that ring alone.

Uses only camera intrinsics. No image content, no test GT.
"""
import argparse, os, sys
import numpy as np

from colmap_loader import read_intrinsics_binary


def valid_mask(W, H, f, cx, cy, k1, iters=50):
    """True where a distorted pixel's undistorted preimage falls inside the WxH canvas.

    GUARDED (audit round 10). For k1 < 0 the cubic rd = ru*(1 + k1*ru^2) is NON-MONOTONE:
    it folds at r_fold = 1/sqrt(-3*k1). Beyond g(r_fold) there is NO undistorted radius
    that maps to the frame corner at all, and Newton then runs off to finite GARBAGE --
    no NaN, no exception (measured: ru_max = 265 at k1=-0.20; 1117 at f=700). The mask
    would come back ~100% "unsupervised" and silently zero an entire member family.
    Our shipped scenes are safe (k1=-0.115, rd_max 0.891 -> ru_max 1.009 << fold 1.704),
    but an unseen dataset is exactly where this bites. Fail loudly instead.
    """
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    xd = (xs - cx) / f
    yd = (ys - cy) / f
    rd = np.sqrt(xd ** 2 + yd ** 2)

    if k1 < 0:
        r_fold = 1.0 / np.sqrt(-3.0 * k1)
        g_fold = r_fold * (1 + k1 * r_fold ** 2)      # max of the cubic on [0, r_fold]
        assert rd.max() < g_fold, (
            f"fov_mask: NO ROOT EXISTS. rd_max={rd.max():.4f} >= g(r_fold)={g_fold:.4f} "
            f"at k1={k1:+.5f} (fold at r_u={r_fold:.4f}). Newton would return finite "
            "garbage and silently zero a member family. Widen the guard only if you have "
            "re-derived the inversion.")

    ru = rd.copy()
    for _ in range(iters):
        ru = ru - (ru * (1 + k1 * ru ** 2) - rd) / (1 + 3 * k1 * ru ** 2)

    # converged, and on the correct (monotone) branch?
    resid = np.abs(ru * (1 + k1 * ru ** 2) - rd).max()
    assert resid < 1e-6, f"fov_mask: Newton did not converge (residual {resid:.3e})"
    if k1 < 0:
        r_fold = 1.0 / np.sqrt(-3.0 * k1)
        assert ru.max() < r_fold, (
            f"fov_mask: Newton landed past the fold (ru_max={ru.max():.4f} >= "
            f"r_fold={r_fold:.4f}) -- wrong branch")

    scale = np.divide(ru, rd, out=np.ones_like(rd), where=rd > 1e-9)
    xu = xd * scale * f + cx
    yu = yd * scale * f + cy
    return ((xu >= 0) & (xu <= W - 1) & (yu >= 0) & (yu <= H - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True, help="scene train/sparse/0")
    ap.add_argument("--out", required=True, help=".npy mask (uint8, 1 = supervised)")
    ap.add_argument("--feather", type=int, default=8,
                    help="px of cosine feather at the boundary (avoids a hard seam)")
    args = ap.parse_args()

    cam = list(read_intrinsics_binary(os.path.join(args.sparse, "cameras.bin")).values())[0]
    p = list(cam.params)
    f, cx, cy = float(p[0]), float(p[1]), float(p[2])
    k1 = float(p[3]) if cam.model in ("SIMPLE_RADIAL", "RADIAL") else 0.0
    W, H = cam.width, cam.height

    m = valid_mask(W, H, f, cx, cy, k1).astype(np.float32)
    if args.feather > 0 and m.min() < 1.0:
        import cv2
        # distance-to-invalid, ramped: a hard 0/1 seam at 40% weight would be visible
        d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        m = np.clip(d / float(args.feather), 0.0, 1.0)
        m = 0.5 - 0.5 * np.cos(np.pi * m)   # smoothstep

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, m.astype(np.float32))
    print(f"k1={k1:+.5f}  {W}x{H}  unsupervised {100 * (m < 0.5).mean():.2f}% "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
