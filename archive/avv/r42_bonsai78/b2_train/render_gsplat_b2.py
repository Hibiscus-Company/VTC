#
# Track B renderer: render competition test_poses.csv from a train_gsplat.py
# checkpoint, with the SAME distortion warp-back + dual-write path as FastGS
# renders (imports DistortionWarp/save_image from render_test_poses.py).
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


# ===================== B2 (3): test-time counterpart of the per-view blur =====================
# The trainer explained each TRAIN photo's own blur with a per-view Gaussian sigma_i and saved
# them in blur.pt. A test view has no photo, so its sigma must be PREDICTED. A2 measured that a
# frame's blur is predictable from its TRAIN NEIGHBOURS alone at R2 = 0.62 on log_lapvar --
# legal under Rule 10, since nothing but train-frame pixels and the test frame's INDEX is used.
#
# We regress the LEARNED sigma_i directly onto the frame-index axis rather than going
# photo-sharpness -> sigma, because (a) it compounds one regression instead of two and (b)
# sigma_i is what the model was actually trained against, including whatever common-mode
# offset the optimisation settled on.
#
# IMPORTANT, and it is why this costs almost no GPU: blurring is a per-image operator applied
# AFTER rasterization, so the ENTIRE sigma_test policy can be swept on already-written PNGs.
# Render ONCE with --blur_mode none, then sweep gamma offline against the 28 eval holes. Only
# the final production render needs this path at all.
FRAME_RE = re.compile(r"(\d{4,})")


def frame_index(name):
    m = FRAME_RE.search(os.path.splitext(os.path.basename(name))[0])
    return int(m.group(1)) if m else None


def nw_predict(fx, fy, x0, h):
    """Nadaraya-Watson kernel average over the frame-index axis (A2 BONSAI_FIT: h=13 frames
    for log_lapvar). Test frames are simply absent from (fx, fy) -- exactly the 'my neighbour
    is also a test frame' condition A2 measured at R2 0.56-0.58."""
    d = np.asarray(fx, dtype=np.float64) - float(x0)
    w = np.exp(-0.5 * (d / h) ** 2)
    if w.sum() < 1e-8:
        return float(np.asarray(fy)[int(np.argmin(np.abs(d)))])
    return float((w * np.asarray(fy)).sum() / w.sum())


def build_sigma_lookup(blur, rows, mode, gamma, h, sigma_max, const):
    """{image_name -> sigma_test px}. Falls back to the capture-wide median for captures
    whose file names carry no frame index."""
    sig = np.asarray(blur["sigma"], dtype=np.float64)
    tf = [frame_index(n) for n in blur["names"]]
    if mode == "const":
        return {r["image_name"]: float(np.clip(gamma * const, 0.0, sigma_max)) for r in rows}, "const"
    usable = all(x is not None for x in tf)
    tf = np.array([x if x is not None else 0 for x in tf], dtype=np.float64)
    out, how = {}, ("nw_index" if usable else "median_fallback")
    for r in rows:
        f = frame_index(r["image_name"])
        s = nw_predict(tf, sig, f, h) if (usable and f is not None) else float(np.median(sig))
        out[r["image_name"]] = float(np.clip(gamma * s, 0.0, sigma_max))
    return out, how


def _sep1(arr, k, radius, axis):
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (radius, radius)
    a = np.pad(arr, pad, mode="reflect")
    out = np.zeros_like(arr)
    for i, kv in enumerate(k):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(i, i + arr.shape[axis])
        out += kv * a[tuple(sl)]
    return out


def gauss_blur_np(arr, sigma, radius):
    """Separable Gaussian on an [H,W,3] float array, reflect-padded. Same kernel and same
    padding as the trainer's gauss_blur1, so train and test see the identical operator."""
    if sigma <= 1e-3:
        return arr
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return _sep1(_sep1(arr.astype(np.float64), k, radius, 1), k, radius, 0)


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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from render_test_poses import DistortionWarp, save_image, load_csv, read_k_from_sparse
from scene.colmap_loader import qvec2rotmat


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
    # ---------------- B2 (3): per-view blur at test time ----------------
    p.add_argument("--blur_pt", default=None,
                   help="blur.pt written by train_gsplat_b2.py --blur_view. Holds the "
                        "per-TRAIN-view sigma the gaussians were deblurred against.")
    p.add_argument("--blur_mode", default="none", choices=("none", "predict", "const"),
                   help="'none' = render the SHARP model as-is (what the training fix is FOR; "
                        "also the right choice for an offline sigma sweep, since blurring is "
                        "a post-rasterization per-image operator). 'predict' = Nadaraya-Watson "
                        "regression of the train-view sigmas onto this frame's index (A2 "
                        "h=13). 'const' = one --blur_const for every view.")
    p.add_argument("--blur_gamma", type=float, default=1.0,
                   help="shrinkage on the predicted sigma. gamma=0 reduces to 'none'. The "
                        "predictor's residual sd is ~0.5 px, and A1 measured that the metric's "
                        "preferred global direction on bonsai renders is MILD BLUR (best "
                        "global unsharp alpha = -0.15 at sigma 1.6, dScore +0.0735), not "
                        "sharpening -- so gamma is the knob that decides whether this whole "
                        "branch pays. SWEEP IT OFFLINE on saved PNGs; never re-render per gamma.")
    p.add_argument("--blur_h", type=float, default=13.0,
                   help="Nadaraya-Watson bandwidth in FRAMES (A2 BONSAI_FIT log_lapvar h=13)")
    p.add_argument("--blur_const", type=float, default=0.0, help="sigma px for --blur_mode const")
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

    # ---------------- B2 (3): sigma_test lookup ----------------
    sig_lut, blur_radius = None, 0
    if args.blur_mode != "none":
        assert args.blur_pt, "--blur_mode needs --blur_pt (blur.pt from --blur_view training)"
        blur = torch.load(args.blur_pt, map_location="cpu", weights_only=False)
        blur_radius = int(blur.get("radius", 8))
        sig_lut, how = build_sigma_lookup(blur, rows, args.blur_mode, args.blur_gamma,
                                          args.blur_h, float(blur.get("sigma_max", 2.5)),
                                          args.blur_const)
        v = np.array(list(sig_lut.values()))
        tr = np.asarray(blur["sigma"], dtype=np.float64)
        print(f"B2 blur_mode={args.blur_mode} ({how}) gamma={args.blur_gamma} radius={blur_radius}"
              f" | train sigma med {np.median(tr):.3f} [{tr.min():.3f},{tr.max():.3f}]"
              f" | test sigma med {np.median(v):.3f} [{v.min():.3f},{v.max():.3f}] px")
        if args.blur_mode == "predict":
            # the prediction is a kernel AVERAGE, so it is shrunk toward the mean by
            # construction; report by how much so a silent collapse to a constant is visible
            print(f"   predicted-sigma sd {v.std():.4f} vs train-sigma sd {tr.std():.4f} "
                  f"(shrink {v.std()/max(tr.std(),1e-9):.2f}x) -- if this is near 0 the "
                  f"'predict' mode has degenerated to 'const' and you should ship const")

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
            if sig_lut is not None:
                # BEFORE the distortion warp: the photo's PSF lives in sensor space and the
                # warp maps pinhole -> sensor, so blurring first is the physically consistent
                # order (the two differ only by the local warp Jacobian, <1% away from the
                # corners, and bonsai renders with --distort none where warp is None anyway).
                # AFTER the clamp, so the operator sees exactly the pixels that get encoded.
                arr = np.clip(gauss_blur_np(arr, sig_lut[row["image_name"]], blur_radius),
                              0.0, 1.0).astype(np.float32)
            if warp is not None:
                arr = warp.apply(arr)
            save_image(arr, os.path.join(args.out, row["image_name"]),
                       jpeg_quality=args.jpeg_quality,
                       jpeg_subsampling=args.jpeg_subsampling, png_dir=args.png_dir)
    print(f"Wrote {len(rows)} images to {args.out}")


if __name__ == "__main__":
    main()
