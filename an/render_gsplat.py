#
# Track B renderer: render competition test_poses.csv from a train_gsplat.py
# checkpoint, with the SAME distortion warp-back + dual-write path as FastGS
# renders (imports DistortionWarp/save_image from cameras.py).
#
# Because train_gsplat.py keeps raw COLMAP world coordinates, CSV poses are
# used as-is (w2c straight into gsplat's viewmats). Padding for the warp is
# applied by shifting the principal point (cx,cy)+pad — exact, no FoV algebra.
# rasterize_mode MUST match training ("antialiased": opacity-compensated
# dilation; a model trained antialiased renders dim in classic mode).
#
import os
import re
import sys
import argparse
import numpy as np
import torch


def seq_num(name):
    # DJI_20241229103156_0001_V.JPG -> 1 (flight sequence; test frames
    # interleave the same numbering as training frames)
    m = re.search(r"_(\d{4})_", name)
    return int(m.group(1)) if m else None


def interp_affine(app, row, w2c, k=4):
    """Inverse-distance-weighted 3x4 affine from the k nearest training frames,
    by flight sequence number when parseable, else by camera-center distance."""
    names, M = app["names"], app["M"]  # [n], [n,3,4] cpu
    s_test = seq_num(row["image_name"])
    seqs = [seq_num(n) for n in names]
    if s_test is not None and all(s is not None for s in seqs):
        d = np.abs(np.array(seqs, dtype=np.float64) - s_test)
    else:
        c_test = -w2c[:3, :3].T @ w2c[:3, 3]
        d = np.linalg.norm(app["centers"] - c_test, axis=1)
    idx = np.argsort(d)[:k]
    w = 1.0 / (d[idx] + 1e-6)
    w = w / w.sum()
    return (M[idx] * torch.tensor(w, dtype=torch.float32).view(-1, 1, 1)).sum(0)

from cameras import DistortionWarp, save_image, load_csv, read_k_from_sparse
from colmap_loader import qvec2rotmat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="train_gsplat.py output ckpt.pt")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--distort", default="none", help="'none', 'auto' (needs --sparse), or numeric k")
    p.add_argument("--sparse", default=None)
    p.add_argument("--jpeg_quality", type=int, default=100)
    p.add_argument("--jpeg_subsampling", type=int, default=0)
    p.add_argument("--png_dir", default=None)
    p.add_argument("--app_affine", default=None,
                   help="app_affine.pt from --app_affine training: interpolate each test "
                        "pose's 3x4 color transform from trajectory-neighbor training "
                        "frames (DJI sequence numbers; falls back to camera-center KNN)")
    p.add_argument("--app_k", type=int, default=4, help="neighbors for --app_affine interpolation")
    p.add_argument("--ppisp", default=None,
                   help="ppisp.pt from --ppisp training: controller predicts exposure/WB "
                        "per test view from the rendered radiance (frame_idx=None)")
    p.add_argument("--ut_render", choices=("native", "warp"), default="native",
                   help="for a --ut-trained ckpt (auto-detected): 'native' renders "
                        "distorted-space directly (classic + with_ut + with_eval3d + "
                        "radial_coeffs, no warp); 'warp' renders pinhole in classic "
                        "mode and applies DistortionWarp — only for the UT-vs-warp "
                        "A/B smoke, both paths must agree sub-pixel away from corners")
    p.add_argument("--sh_degree", type=int, default=None,
                   help="clamp SH below the ckpt's degree at render time (P4 probe)")
    p.add_argument("--radial", action="store_true",
                   help="CLOSED LEVER, do not use for antialiased-trained models: "
                        "radial_coeffs needs with_ut=True, and gsplat 1.5.3 raises on "
                        "with_ut + rasterize_mode!=classic (rendering.py:170) — classic "
                        "renders an antialiased-trained model dim. Kept for future "
                        "classic-mode experiments only")
    args = p.parse_args()

    from gsplat import rasterization

    k = None
    if args.distort == "auto":
        assert args.sparse, "--distort auto requires --sparse"
        k = read_k_from_sparse(args.sparse)
        print(f"Distortion warp enabled, k={k:+.6f}")
    elif args.distort != "none":
        k = float(args.distort)

    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    splats = {n: t.cuda() for n, t in ckpt["splats"].items()}
    sh_degree = ckpt["sh_degree"]
    if args.sh_degree is not None:
        # P4: at 11.8 deg of extrapolation from the nearest train view, view-dependent
        # colour is a candidate silent PSNR tax -- higher SH orders can hallucinate
        # view-dependent shading the test pose never justified. Clamping costs nothing
        # to try and has never been measured. gsplat uses the first (d+1)^2 coeffs.
        assert 0 <= args.sh_degree <= sh_degree, f"can only clamp below {sh_degree}"
        sh_degree = args.sh_degree
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    # a --ut-trained model was optimized in classic mode; rendering it
    # antialiased (or vice versa) shifts opacity compensation -> dim/bright
    ut = bool(ckpt.get("ut", False))
    mode = "classic" if ut else "antialiased"
    ut_native = ut and args.ut_render == "native"
    # audit r16 B1: render must use the SAME eps2d the ckpt trained with, else opacity/size
    # response shifts between train and test (>=0 means it was set; <0 = gsplat default)
    eps2d = float(ckpt.get("eps2d", -1.0))
    eps_extra = {"eps2d": eps2d} if eps2d >= 0 else {}
    print(f"{len(splats['means'])} gaussians (mode={mode}, ut={ut}, eps2d={eps2d})")

    rows = load_csv(args.csv)
    r0 = rows[0]
    for r in rows:
        assert (r["fx"], r["fy"], r["width"], r["height"]) == \
               (r0["fx"], r0["fy"], r0["width"], r0["height"]), "csv must share one camera"
    W, H = int(r0["width"]), int(r0["height"])
    fx, fy = float(r0["fx"]), float(r0["fy"])

    warp, pad, radial_coeffs = None, 0, None
    if ut_native:
        kk = float(ckpt.get("k1", k if k is not None else 0.0))
        radial_coeffs = torch.tensor([[kk, 0, 0, 0, 0, 0]], dtype=torch.float32, device="cuda")
        print(f"3DGUT native distorted render, k1={kk:+.6f} (no warp)")
    elif k is not None and args.radial:
        # COLMAP SIMPLE_RADIAL x_d = x_u(1 + k r^2) == OpenCV k1 term
        radial_coeffs = torch.tensor([[k, 0, 0, 0, 0, 0]], dtype=torch.float32, device="cuda")
        print("Native radial rendering (no warp)")
    elif k is not None:
        warp = DistortionWarp(W, H, fx, fy, k)
        pad = warp.pad
        print(f"Render canvas padding: {pad}px")
    # principal point from CSV when present (equals W/2,H/2 on this data)
    cx = float(r0.get("cx", W / 2.0)) + pad
    cy = float(r0.get("cy", H / 2.0)) + pad
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device="cuda")

    app = torch.load(args.app_affine, weights_only=False) if args.app_affine else None

    pp = None
    if args.ppisp:
        from ppisp import PPISP
        pp = PPISP.from_state_dict(torch.load(args.ppisp, weights_only=False))
        pp.eval()
        # applied on the padded pinhole render, BEFORE the warp: vignetting was
        # learned in undistorted pixel coords (pad shifts the field ≤4px/0.3%)
        pw, ph = W + 2 * pad, H + 2 * pad
        pys, pxs = torch.meshgrid(torch.arange(ph, device="cuda", dtype=torch.float32),
                                  torch.arange(pw, device="cuda", dtype=torch.float32),
                                  indexing="ij")
        pp_xy = torch.stack([pxs, pys], dim=-1)
        print("PPISP controller inference enabled")

    os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        for row in rows:
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = qvec2rotmat(np.array([float(row["qw"]), float(row["qx"]),
                                                float(row["qy"]), float(row["qz"])]))
            w2c[:3, 3] = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
            render, _, _ = rasterization(
                means=splats["means"], quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors,
                viewmats=torch.from_numpy(w2c).cuda()[None], Ks=K[None],
                width=W + 2 * pad, height=H + 2 * pad,
                sh_degree=sh_degree, rasterize_mode=mode,
                near_plane=0.01, packed=False, radial_coeffs=radial_coeffs,
                # a UT-trained ckpt must keep with_ut/with_eval3d even in warp
                # mode (pinhole UT is valid and fold-free): dropping eval3d
                # changes the gaussian response model between train and render
                with_ut=ut, with_eval3d=ut, **eps_extra)
            img = render[0]
            if pp is not None:
                img = pp(rgb=img, pixel_coords=pp_xy,
                         resolution=(W + 2 * pad, H + 2 * pad),
                         camera_idx=0, frame_idx=None)
            if app is not None:
                M = interp_affine(app, row, w2c, k=args.app_k).cuda()
                img = img @ M[:, :3].T + M[:, 3]
            arr = img.clamp(0.0, 1.0).cpu().numpy()
            if warp is not None:
                arr = warp.apply(arr)
            save_image(arr, os.path.join(args.out, row["image_name"]),
                       jpeg_quality=args.jpeg_quality,
                       jpeg_subsampling=args.jpeg_subsampling, png_dir=args.png_dir)
    print(f"Wrote {len(rows)} images to {args.out}")


if __name__ == "__main__":
    main()
