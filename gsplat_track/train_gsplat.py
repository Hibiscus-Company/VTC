#
# Track B trainer: gsplat MCMC + antialiased rasterization on our competition
# scene layout (train/sparse/0 + train/images_undist).
#
# Deliberately does NOT use gsplat's Parser/Dataset (audit round-3 pitfalls):
#   - no scene-normalization transform -> raw COLMAP world coords everywhere,
#     so CSV test poses need no transform at render time
#   - no double-undistortion: images_undist are already pinhole; we take K
#     from cameras.bin (SIMPLE_RADIAL [f,cx,cy,k]) and ignore k
#   - no data_factor resizing: native resolution always
# scene_scale (for lr/noise scaling only, world coords untouched) follows the
# simple_trainer recipe: 1.1 * max camera-center distance from their mean.
#
# Smoke tests (audit round 3):
#   --geom_check  project points3D through our K/w2c vs COLMAP's stored 2D
#                 observations (median px error; <1px = conventions correct)
#   --iters 1000  mini-train for pipeline validation before a 30k run
#
import os
import sys
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                 read_points3D_binary, qvec2rotmat)


def load_scene(source, images_dirname):
    sparse = os.path.join(source, "sparse", "0")
    cams = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    imgs = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    xyz, rgb, _ = read_points3D_binary(os.path.join(sparse, "points3D.bin"))

    assert len(cams) == 1, f"expected single camera, got {len(cams)}"
    cam = list(cams.values())[0]
    assert cam.model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_PINHOLE", "PINHOLE"), cam.model
    if cam.model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_PINHOLE"):
        f, cx, cy = float(cam.params[0]), float(cam.params[1]), float(cam.params[2])
        fx = fy = f
    else:
        fx, fy, cx, cy = map(float, cam.params[:4])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    # SIMPLE_RADIAL/RADIAL k1 == OpenCV k1 (x_d = x_u(1+k r^2)); 0 for pinhole
    k1 = float(cam.params[3]) if cam.model in ("SIMPLE_RADIAL", "RADIAL") else 0.0

    views = []
    img_dir = os.path.join(source, images_dirname)
    for img in imgs.values():
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = qvec2rotmat(img.qvec)
        w2c[:3, 3] = img.tvec
        views.append({"name": img.name, "w2c": w2c,
                      "path": os.path.join(img_dir, img.name),
                      "xys": img.xys, "point3D_ids": img.point3D_ids})
    # the competition sparse holds 388 poses (train + test + dropped) but only
    # the train images exist on disk — keep only those, BEFORE scene_scale is
    # derived from camera centers (audit round 5 finding 1)
    n_all = len(views)
    views = [v for v in views if os.path.exists(v["path"])]
    print(f"{len(views)}/{n_all} images.bin entries have files in {img_dir}")
    views.sort(key=lambda v: v["name"])
    return (views, K, (cam.width, cam.height), xyz.astype(np.float32),
            rgb.astype(np.float32), k1)


def geom_check(views, K, xyz_all, pt_index, n=3):
    # audit round-3 smoke (a'): reproject each view's COLMAP 2D observations.
    # The competition sparse models store images.bin xys in ORIGINAL-resolution
    # (5280x3956) coords while cameras.bin was rescaled to the delivered
    # 1320x989 — so fit the integer downscale factor s per view and compare
    # obs/s to our projection. Residual includes the (unmodeled) few-px radial
    # distortion of the original observations, so the pass bar is 2px, not 1.
    errs, scales = [], []
    for v in views[:: max(1, len(views) // n)][:n]:
        valid = v["point3D_ids"] >= 0
        ids, xys = v["point3D_ids"][valid], v["xys"][valid]
        rows = np.array([pt_index.get(i, -1) for i in ids])
        keep = rows >= 0
        pts = xyz_all[rows[keep]]
        cam_pts = (v["w2c"][:3, :3] @ pts.T + v["w2c"][:3, 3:4]).T
        infront = cam_pts[:, 2] > 1e-3
        proj = (K @ cam_pts[infront].T).T
        proj = proj[:, :2] / proj[:, 2:3]
        obs = xys[keep][infront]
        s, err = min(((c, float(np.median(np.linalg.norm(obs / c - proj, axis=1))))
                      for c in (1, 2, 4, 8)), key=lambda t: t[1])
        errs.append(err); scales.append(s)
        print(f"geom_check {v['name']}: {infront.sum()} pts, obs-scale 1/{s}, "
              f"median reproj err {err:.3f}px")
    med = float(np.median(errs))
    print(f"geom_check OVERALL median: {med:.3f}px at obs-scale 1/{scales[0]} "
          f"({'OK' if med < 2.0 else 'FAIL — convention bug!'})")
    return med


def knn_mean_dist(pts, k=3):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    return d[:, 1:].mean(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="scene train/ dir (holds sparse/0 and images)")
    p.add_argument("--images", default="images_undist")
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=30_000)
    p.add_argument("--cap_max", type=int, default=5_000_000)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--means_lr", type=float, default=1.6e-4)
    p.add_argument("--opacity_reg", type=float, default=0.01)
    p.add_argument("--scale_reg", type=float, default=0.01)
    p.add_argument("--ssim_lambda", type=float, default=0.2)
    p.add_argument("--lambda_lpips", type=float, default=0.1)
    p.add_argument("--lpips_from", type=int, default=25_000)
    p.add_argument("--refine_stop", type=int, default=25_000)
    p.add_argument("--noise_stop", type=int, default=-1,
                   help="stop MCMC noise injection at this iter (-1 = never; "
                        "set = lpips_from so the perceptual phase polishes a noise-free model)")
    p.add_argument("--app_affine", action="store_true",
                   help="idea 1c: per-image 3x4 affine color transform in the loss, "
                        "L2-regularized to identity; params saved to app_affine.pt for "
                        "trajectory interpolation at test render time")
    p.add_argument("--bilagrid", action="store_true",
                   help="idea 1b: per-view bilateral grid (16x16x8) in the loss, "
                        "10x TV reg; identity at test time")
    p.add_argument("--ppisp_activation", type=float, default=29 / 30,
                   help="controller_activation_ratio; keep the post-activation "
                        "distillation window (splats frozen w.r.t. photometric "
                        "loss) to ~1k steps at the very end")
    p.add_argument("--ppisp", action="store_true",
                   help="idea 1a: PPISP per-frame ISP + controller (all trained per-scene); "
                        "controller predicts exposure/WB for novel views from rendered "
                        "radiance; module saved to ppisp.pt")
    p.add_argument("--app_lr", type=float, default=1e-3)
    p.add_argument("--lambda_appreg", type=float, default=0.1,
                   help="L2-to-identity weight for --app_affine")
    p.add_argument("--ut", action="store_true",
                   help="idea 2 (3DGUT): train in DISTORTED space — classic mode + "
                        "with_ut + with_eval3d + radial_coeffs k1 from cameras.bin. "
                        "Use with --images images (original distorted GT); "
                        "gsplat rejects UT with antialiased mode, so renders of a "
                        "--ut model must also go through the UT path")
    p.add_argument("--geom_check", action="store_true", help="reprojection smoke test, no training")
    p.add_argument("--init_ply", default=None,
                   help="warm-start from a FastGS/3DGS point_cloud.ply instead of SfM points "
                        "(log-scales/logit-opacities carried raw; f_rest reshaped channel-major)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    views, K_np, (W, H), xyz, rgb, k1 = load_scene(args.source, args.images)
    print(f"{len(views)} views, {len(xyz)} init points, K=[[{K_np[0,0]:.1f},0,{K_np[0,2]:.1f}],"
          f"[0,{K_np[1,1]:.1f},{K_np[1,2]:.1f}]], size {W}x{H}")
    if args.ut:
        if args.images == "images_undist":
            print("WARNING: --ut expects DISTORTED GT (--images images); "
                  "images_undist would double-model the distortion")
        print(f"3DGUT mode: classic + with_ut + with_eval3d, k1={k1:+.6f}")

    if args.geom_check:
        # points3D ids -> row index (read_points3D_binary returns arrays in file order;
        # rebuild the id map the same way the loader saw them)
        import struct
        pt_ids = []
        with open(os.path.join(args.source, "sparse", "0", "points3D.bin"), "rb") as f:
            n_pts = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_pts):
                data = struct.unpack("<QdddBBBd", f.read(43))
                pt_ids.append(data[0])
                tl = struct.unpack("<Q", f.read(8))[0]
                f.read(8 * tl)
        idx = {pid: i for i, pid in enumerate(pt_ids)}
        ok = geom_check(views, K_np, xyz, idx)
        sys.exit(0 if ok < 2.0 else 1)

    import gsplat
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy
    from fused_ssim import fused_ssim
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"

    if args.ut:
        # gsplat 1.5.3 rejects UT/eval3d with non-classic mode (rendering.py:170)
        rast_extra = {"rasterize_mode": "classic", "with_ut": True, "with_eval3d": True,
                      "radial_coeffs": torch.tensor([[k1, 0.0, 0.0, 0.0, 0.0, 0.0]],
                                                    dtype=torch.float32, device=dev)}
    else:
        rast_extra = {"rasterize_mode": "antialiased"}

    # scene scale for lr/noise (world coords untouched)
    centers = np.stack([-v["w2c"][:3, :3].T @ v["w2c"][:3, 3] for v in views])
    scene_scale = 1.1 * float(np.max(np.linalg.norm(centers - centers.mean(0), axis=1)))
    print(f"scene_scale {scene_scale:.3f}")

    if args.init_ply:
        # warm-start from a trained FastGS/3DGS model (audit round 6 rank-2):
        # ply stores raw log-scales and logit-opacities — carry over directly;
        # f_rest is CHANNEL-MAJOR (N,3,15) — transpose to gsplat's [N,15,3]
        from plyfile import PlyData
        v = PlyData.read(args.init_ply)["vertex"]
        n = v["x"].shape[0]
        get = lambda names: np.stack([np.asarray(v[nm], dtype=np.float32) for nm in names], axis=1)
        means = torch.tensor(get(["x", "y", "z"]), device=dev)
        scales = torch.tensor(get([f"scale_{i}" for i in range(3)]), device=dev)
        quats = torch.tensor(get([f"rot_{i}" for i in range(4)]), device=dev)  # wxyz both
        opacities = torch.tensor(np.asarray(v["opacity"], dtype=np.float32), device=dev)
        sh0 = torch.tensor(get([f"f_dc_{i}" for i in range(3)]), device=dev).unsqueeze(1)
        n_rest = (args.sh_degree + 1) ** 2 - 1
        rest = get([f"f_rest_{i}" for i in range(3 * n_rest)])          # [N, 3*15] channel-major
        shN = torch.tensor(rest, device=dev).reshape(n, 3, n_rest).transpose(1, 2).contiguous()
        print(f"Warm-start: {n} gaussians from {args.init_ply}")
        assert args.cap_max >= n, f"cap_max {args.cap_max} < init N {n}"
    else:
        # splat init from SfM points
        means = torch.tensor(xyz, device=dev)
        scales = torch.log(torch.tensor(knn_mean_dist(xyz), device=dev, dtype=torch.float32)
                           .clamp(min=1e-7)).unsqueeze(-1).repeat(1, 3)
        quats = torch.zeros(len(xyz), 4, device=dev); quats[:, 0] = 1.0
        # 0.5 = the MCMC-config reference init (relocation samples ∝ opacity;
        # a low start + opacity_reg slows early mass formation)
        opacities = torch.logit(torch.full((len(xyz),), 0.5, device=dev))
        from utils.sh_utils import RGB2SH
        sh0 = RGB2SH(torch.tensor(rgb / 255.0, device=dev)).unsqueeze(1)          # [N,1,3]
        shN = torch.zeros(len(xyz), (args.sh_degree + 1) ** 2 - 1, 3, device=dev)  # [N,15,3]

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means), "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats), "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0), "shN": torch.nn.Parameter(shN),
    }).to(dev)
    lrs = {"means": args.means_lr * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {n: torch.optim.Adam([{"params": params[n], "lr": lrs[n], "name": n}],
                                      eps=1e-15) for n in params.keys()}

    strategy = MCMCStrategy(cap_max=args.cap_max, refine_stop_iter=args.refine_stop,
                            noise_injection_stop_iter=args.noise_stop, verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    K = torch.tensor(K_np, device=dev)
    w2cs = torch.tensor(np.stack([v["w2c"] for v in views]), device=dev)
    # GT cache on CPU (uint8), like FastGS --data_device cpu
    from PIL import Image
    gt_cache = [None] * len(views)

    # idea 1c: per-image 3x4 affine (M @ rgb + b) applied in the LOSS only —
    # the splat model stays canonical; drone AE/AWB drift is soaked per-frame.
    # Unlike FastGS e16 (which regressed −0.91: no test-time params), these are
    # interpolated onto test poses along the flight trajectory at render time.
    app_M = None
    if args.app_affine:
        eye = torch.eye(3, device=dev).unsqueeze(0).repeat(len(views), 1, 1)
        app_M = torch.nn.Parameter(torch.cat([eye, torch.zeros(len(views), 3, 1, device=dev)], dim=2))
        app_opt = torch.optim.Adam([app_M], lr=args.app_lr)
        app_eye = torch.eye(3, device=dev)

    bil = None
    if args.bilagrid:
        from lib_bilagrid import BilateralGrid, slice as bil_slice, total_variation_loss
        bil = BilateralGrid(len(views), grid_X=16, grid_Y=16, grid_W=8).to(dev)
        bil_opt = torch.optim.Adam(bil.parameters(), lr=2e-3)
        ys, xs = torch.meshgrid(torch.linspace(0, 1, H, device=dev),
                                torch.linspace(0, 1, W, device=dev), indexing="ij")
        bil_xy = torch.stack([xs, ys], dim=-1).unsqueeze(0)  # [1,H,W,2]

    pp = None
    pp_act_step = args.iters + 1
    if args.ppisp:
        from ppisp import PPISP, PPISPConfig
        pp = PPISP(num_cameras=1, num_frames=len(views),
                   config=PPISPConfig(scheduler_decay_max_steps=args.iters,
                                      controller_activation_ratio=args.ppisp_activation))
        # from activation the controller distills with rgb.detach(): splats get
        # ZERO photometric gradient, so (1) keep this window short and after
        # lpips_from, (2) reg terms must switch off with it (no counterweight)
        pp_act_step = int(args.ppisp_activation * args.iters)
        pp_opts = pp.create_optimizers()
        # must be called before forward(): controller activation (at 0.8*iters,
        # then main ISP params freeze + input detach for distillation) reads
        # the scheduler's step counter
        pp_scheds = pp.create_schedulers(pp_opts, max_optimization_iters=args.iters)
        pys, pxs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                  torch.arange(W, device=dev, dtype=torch.float32),
                                  indexing="ij")
        pp_xy = torch.stack([pxs, pys], dim=-1)  # [H,W,2] raw (x,y)

    lpips_net = None
    order = np.random.permutation(len(views))
    oi = 0
    for step in range(args.iters):
        if oi >= len(order):
            order = np.random.permutation(len(views)); oi = 0
        vi = int(order[oi]); oi += 1
        if gt_cache[vi] is None:
            gt_cache[vi] = torch.from_numpy(
                np.asarray(Image.open(views[vi]["path"]).convert("RGB"), dtype=np.uint8))
        gt = gt_cache[vi].to(dev).float().div_(255.0)  # [H,W,3]

        # ramp only from SfM init — a warm-started model must render at full
        # degree immediately or its trained high-band SH gets damaged while
        # the image supervises low bands only
        sh_deg = args.sh_degree if args.init_ply else min(step // 1000, args.sh_degree)
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        render, alpha, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors, viewmats=w2cs[vi:vi + 1], Ks=K[None], width=W, height=H,
            sh_degree=sh_deg, near_plane=0.01, packed=False, absgrad=False,
            **rast_extra)
        # loss on UNCLAMPED colors (upstream convention: clamping zeroes
        # gradients at saturated pixels, exactly where early training
        # overshoots); clamp only for the LPIPS term (parity with scoring)
        img = render[0]  # [H,W,3]

        strategy.step_pre_backward(params, optimizers, state, step, info)

        if app_M is not None:
            img = img @ app_M[vi, :, :3].T + app_M[vi, :, 3]
        if bil is not None:
            img = bil_slice(bil, bil_xy, img.unsqueeze(0),
                            torch.tensor([[vi]], device=dev))["rgb"].squeeze(0)
        if pp is not None:
            img = pp(rgb=img, pixel_coords=pp_xy, resolution=(W, H),
                     camera_idx=0, frame_idx=vi)

        l1 = (img - gt).abs().mean()
        ssim = fused_ssim(img.permute(2, 0, 1).unsqueeze(0),
                          gt.permute(2, 0, 1).unsqueeze(0))
        loss = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * (1 - ssim)
        if step < pp_act_step:
            loss += args.opacity_reg * torch.sigmoid(params["opacities"]).mean()
            loss += args.scale_reg * torch.exp(params["scales"]).mean()
        if app_M is not None:
            loss += args.lambda_appreg * (
                (app_M[vi, :, :3] - app_eye).pow(2).sum() + app_M[vi, :, 3].pow(2).sum())
        if bil is not None:
            loss += 10 * total_variation_loss(bil.grids)
        if pp is not None:
            loss += pp.get_regularization_loss()
        if args.lambda_lpips > 0 and step > args.lpips_from:
            if lpips_net is None:
                import lpips
                lpips_net = lpips.LPIPS(net="vgg").to(dev)
                for q in lpips_net.parameters():
                    q.requires_grad_(False)
            loss += args.lambda_lpips * lpips_net(
                img.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0) * 2 - 1,
                gt.permute(2, 0, 1).unsqueeze(0) * 2 - 1).squeeze()

        loss.backward()

        # 3DGS-style exponential means-lr decay to 1% over the run
        cur_means_lr = lrs["means"] * (0.01 ** (step / args.iters))
        optimizers["means"].param_groups[0]["lr"] = cur_means_lr

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if app_M is not None:
            app_opt.step()
            app_opt.zero_grad(set_to_none=True)
        if bil is not None:
            bil_opt.step()
            bil_opt.zero_grad(set_to_none=True)
        if pp is not None:
            for o in pp_opts:
                o.step()
                o.zero_grad(set_to_none=True)
            for s in pp_scheds:
                s.step()

        # AFTER opt.step (audit round 5 finding 2): MCMC relocation rebuilds
        # params with grad=None — calling this before step() silently discards
        # the step's gradients on every refine event
        strategy.step_post_backward(params, optimizers, state, step, info, lr=cur_means_lr)

        if step % 1000 == 0 or step == args.iters - 1:
            print(f"[{step}] loss {loss.item():.4f} l1 {l1.item():.4f} "
                  f"ssim {ssim.item():.4f} N {len(params['means'])}")

    os.makedirs(args.out, exist_ok=True)
    torch.save({"splats": {n: params[n].detach().cpu() for n in params.keys()},
                "sh_degree": args.sh_degree, "K": K_np, "wh": (W, H),
                "ut": args.ut, "k1": k1},
               os.path.join(args.out, "ckpt.pt"))
    print(f"Saved {len(params['means'])} gaussians to {args.out}/ckpt.pt")
    if pp is not None:
        torch.save(pp.state_dict(), os.path.join(args.out, "ppisp.pt"))
        print(f"Saved ppisp.pt (exposure std {pp.exposure_params.std().item():.4f})")
    if app_M is not None:
        centers = np.stack([-v["w2c"][:3, :3].T @ v["w2c"][:3, 3] for v in views])
        torch.save({"M": app_M.detach().cpu(), "names": [v["name"] for v in views],
                    "centers": centers},
                   os.path.join(args.out, "app_affine.pt"))
        dev_id = (app_M.detach() - torch.cat(
            [torch.eye(3, device=dev), torch.zeros(3, 1, device=dev)], 1)).abs()
        print(f"Saved app_affine.pt  |M-I| mean {dev_id.mean():.4f} max {dev_id.max():.4f}")


if __name__ == "__main__":
    main()
