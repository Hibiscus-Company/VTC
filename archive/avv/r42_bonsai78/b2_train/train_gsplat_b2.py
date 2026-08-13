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


@torch.no_grad()
def compute_mip3d_filter(means, w2cs, K, W, H, factor):
    """Mip-Splatting 3D smoothing filter (Yu et al. 2024), computed from the TRAIN cameras.

    Per gaussian: the world-space size below which NO training view can constrain it --
    dist_to_nearest_train_cam / focal. A gaussian smaller than this sits under the Nyquist
    limit of every view that sees it, so its detail is pure aliasing, unconstrained by the
    data, and it renders as noise from any novel pose. Our post-hoc Mip probe measured the
    median gaussian at FAR below one pixel, i.e. this population dominates our models --
    and it is exactly the reconstruction noise that pixel-mean ensembling was cancelling
    (+1.82). Band-limiting DURING training stops the spikes forming at all.

    Returns [N,1] world-space sigma to be added in quadrature to the gaussian scales.
    """
    N = means.shape[0]
    dev = means.device
    dist = torch.full((N,), 1e10, device=dev)
    seen = torch.zeros(N, dtype=torch.bool, device=dev)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    focal = max(fx, fy)
    for c in range(w2cs.shape[0]):
        R, t = w2cs[c, :3, :3], w2cs[c, :3, 3]
        xyz_cam = means @ R.T + t
        z = xyz_cam[:, 2]
        zc = z.clamp(min=1e-3)
        x = xyz_cam[:, 0] / zc * fx + cx
        y = xyz_cam[:, 1] / zc * fy + cy
        # 15% frustum margin (upstream): a gaussian just outside the frame still
        # contributes to in-frame pixels through its tail
        ok = ((z > 0.2) & (x >= -0.15 * W) & (x <= 1.15 * W)
              & (y >= -0.15 * H) & (y <= 1.15 * H))
        d = xyz_cam.norm(dim=1)
        dist = torch.where(ok & (d < dist), d, dist)
        seen |= ok
    if bool(seen.any()):
        # never-seen gaussians get the most permissive (largest) band limit
        dist[~seen] = dist[seen].max()
    else:
        dist.fill_(1.0)
    return (dist / focal * (factor ** 0.5)).unsqueeze(-1)


def mip3d_apply(scales_lin, opac, filt):
    """Add the band limit in quadrature and compensate opacity so total energy is preserved.
    scales_lin [N,3] linear scales, opac [N] in (0,1), filt [N,1]. Returns (scales, opac)."""
    s2 = scales_lin.square()
    d2 = s2 + filt.square()
    coef = torch.sqrt(s2.prod(-1) / d2.prod(-1).clamp_min(1e-30))
    return torch.sqrt(d2), opac * coef


# ======================= B2 helpers: sharpness weighting + per-view blur =======================
def load_sharp_csv(path, stat):
    """{image_name -> float} from the B2 sidecar CSV. Keyed by the image FILE NAME so it
    survives any view reordering; also accepts the stem."""
    import csv as _csv
    out = {}
    with open(path) as fh:
        for r in _csv.DictReader(fh):
            assert stat in r, f"{path} has no column {stat!r} (has {list(r)})"
            v = float(r[stat])
            out[r["name"]] = v
            out[os.path.splitext(r["name"])[0]] = v
    return out


def build_view_weights(args, names, sharp):
    """Per-view photometric weight, mean 1 over the views actually trained on.

    The trainer processes exactly ONE view per step, so w_i is a per-step scalar on the
    loss -- there is no batching obstacle to weighting the SSIM or LPIPS terms.

    Under Adam this is NOT a plain gradient rescale: for a parameter whose gradient comes
    from views that all share the same w (e.g. a gaussian visible in one view only) Adam's
    second-moment normalisation makes the weight EXACTLY cancel -- a literal no-op. The
    weighting only bites on parameters that receive CONFLICTING gradients from views of
    differing sharpness, which is precisely the blur-average population, but it also means
    the realised effect is much weaker than the nominal weight ratio. Expect the sqrt-damped
    version of whatever the ESS table predicts.
    """
    s = np.array([sharp[n] for n in names], dtype=np.float64)
    if args.sw_mode == "power":
        lin = s if args.sharp_stat == "reblur" else np.exp(s)
        w = (lin / np.median(lin)) ** args.sw_power
    elif args.sw_mode == "tilt":
        w = np.exp(args.sw_beta * (s - s.mean()) / (s.std() + 1e-12))
    elif args.sw_mode == "topk":
        w = np.zeros_like(s)
        if args.sw_topk_stratify > 0:
            B = args.sw_topk_stratify
            keep = max(1, int(round(args.sw_topk * B)))
            for i in range(0, len(s), B):
                blk = np.arange(i, min(i + B, len(s)))
                k = min(keep, len(blk))
                w[blk[np.argsort(-s[blk])[:k]]] = 1.0
        else:
            k = max(1, int(round(args.sw_topk * len(s))))
            w[np.argsort(-s)[:k]] = 1.0
        assert w.sum() > 0
    else:
        raise ValueError(args.sw_mode)
    if args.sw_mode != "topk":
        w = np.clip(w, args.sw_wmin, args.sw_wmax)
    w = w / w.mean()
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    print(f"sharpness weights [{args.sw_mode}/{args.sharp_stat}] over {len(w)} views: "
          f"min {w.min():.3f} p10 {np.percentile(w,10):.3f} med {np.median(w):.3f} "
          f"p90 {np.percentile(w,90):.3f} max {w.max():.3f}  ESS {ess:.1f}/{len(w)} "
          f"({100*ess/len(w):.0f}%)  weighted-mean sharpness shift "
          f"{float((w*s).mean()-s.mean()):+.4f}")
    return w


def blur_sigma_init(args, names, sharp):
    """sigma_i (px) implied by each photo's own measured sharpness.

    The gaussians are asked to reach the sharpness of the --blur_ref_q quantile photo;
    every softer photo is then explained by convolving that consensus with sigma_i. A2's
    calibration on bonsai is 1.00 nats of log_lapvar per px of added Gaussian sigma.
    """
    s = np.array([sharp[n] for n in names], dtype=np.float64)
    if args.sharp_stat != "log_lapvar":
        # rescale an arbitrary statistic onto the log_lapvar scale by matching sd, so
        # --blur_gain keeps its px-per-nat meaning
        s = (s - s.mean()) / (s.std() + 1e-12) * 0.727 + s.mean()
    ref = float(np.quantile(s, args.blur_ref_q))
    sig = np.clip(args.blur_gain * (ref - s), 0.0, args.blur_sigma_max)
    print(f"blur sigma init (ref q{args.blur_ref_q} = {ref:+.4f}, gain {args.blur_gain}): "
          f"zero-frac {100*(sig<=1e-6).mean():.1f}%  med {np.median(sig):.3f} "
          f"p90 {np.percentile(sig,90):.3f} max {sig.max():.3f} mean {sig.mean():.3f} px")
    return sig


def gauss_blur1(img_hw3, sigma, radius):
    """Separable depthwise Gaussian blur of an [H,W,3] render, differentiable w.r.t. sigma.

    The kernel is rebuilt from sigma every call (a 2R+1 vector -- free) so sigma receives
    gradient. Radius is FIXED from --blur_sigma_max, so the conv shapes never change and
    cudnn does not re-plan. Reflect padding keeps the frame border unbiased.
    """
    x = torch.arange(-radius, radius + 1, device=img_hw3.device, dtype=img_hw3.dtype)
    k = torch.exp(-0.5 * (x / sigma.clamp(min=1e-3)) ** 2)
    k = k / k.sum()
    t = img_hw3.permute(2, 0, 1).unsqueeze(0)                    # [1,3,H,W]
    kx = k.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    t = F.pad(t, (radius, radius, 0, 0), mode="reflect")
    t = F.conv2d(t, kx, groups=3)
    t = F.pad(t, (0, 0, radius, radius), mode="reflect")
    t = F.conv2d(t, ky, groups=3)
    return t.squeeze(0).permute(1, 2, 0)                         # [H,W,3]


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
    p.add_argument("--metric_loss", action="store_true",
                   help="idea 3: EXACT metric-matched loss, derived from the "
                        "competition score S = 0.4(1-LPIPS) + 0.3 SSIM + 0.3 PSNR/50. "
                        "Maximizing S == minimizing 0.4*LPIPS + 0.3*(1-SSIM) + "
                        "0.02606*ln(MSE)  [0.06/ln10]. The log-MSE term is "
                        "self-scaling (grad 1/mse) and is the ONLY term that "
                        "optimizes PSNR — the stock L1+DSSIM loss never does "
                        "(L1 = median estimator; PSNR needs the mean)")
    p.add_argument("--pure_l2", action="store_true",
                   help="DIAGNOSTIC (audit r12): loss = MSE only, no SSIM. Isolates "
                        "whether the ~27dB train fit is a loss/reg cap or a real "
                        "capacity/content wall — the one thing the registration oracle "
                        "cannot see. Run from-scratch with densification ON.")
    p.add_argument("--init_ckpt", default=None,
                   help="warm-start from our own train_gsplat ckpt.pt (UT-aware)")
    p.add_argument("--init_ply", default=None,
                   help="warm-start from a FastGS/3DGS point_cloud.ply instead of SfM points "
                        "(log-scales/logit-opacities carried raw; f_rest reshaped channel-major)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="EMA of gaussian params over the POST-refine_stop window (topology "
                        "frozen there, so a plain tensor EMA is valid -- no clone/split "
                        "bookkeeping). Denoises late SGLD/LPIPS-phase jitter that made several "
                        "runs oscillate. Saved in place of raw params. 0 = off. Try 0.99.")
    p.add_argument("--depth_prior", type=float, default=0.0,
                   help="MICCAI endoscopic-GS technique (SurgicalGaussian/Endo-4DGS): weight of a "
                        "monocular-depth-prior loss. Renders expected depth (RGB+ED) and matches it "
                        "to a FROZEN Depth-Anything-V2 prediction via a scale-and-shift-invariant "
                        "PEARSON correlation (mono depth has unknown scale). Regularizes geometry "
                        "where SfM is sparse (glass table / low texture). 0 = off. Try 0.05.")
    p.add_argument("--depth_dir", default=None,
                   help="dir of precomputed Depth-Anything .npy depth maps (one per train image, "
                        "keyed by image stem); required when --depth_prior > 0")
    p.add_argument("--sky_dome", type=int, default=0,
                   help="WildGaussians-inspired (trick-hunt, tower scenes): seed N low-opacity "
                        "points on a sphere enclosing the scene before training. Open-sky "
                        "background gets ~0 SfM points -> MCMC spawns cloud-like floaters there; "
                        "dome points give it real geometry to relocate into instead. 0 = off. "
                        "Try ~50000. SfM-init only (ignored on warm-start).")
    p.add_argument("--absgrad", action="store_true",
                   help="AbsGS-inspired (arXiv 2404.10484, found via trick-hunt 21/07): gsplat "
                        "natively supports absgrad (confirmed via inspect.signature on our "
                        "installed 1.5.3). Mechanism: opposite-signed positional gradients from "
                        "different views can CANCEL in high-frequency/repetitive regions (tower "
                        "lattice braces, bolts) under the default signed-sum, silently hiding "
                        "the densification signal there; absgrad sums |gradient| instead, so "
                        "collision doesn't mask the need for more capacity. One-flag experiment, "
                        "no other code path touched. Default False = unmodified behavior.")
    p.add_argument("--init_clip", type=float, default=0.0,
                   help="drop SfM init points farther than camera-hull-radius x this "
                        "(0 = off; consult#3 M2: mirror-world reflection points)")
    p.add_argument("--mip3d", type=float, default=0.0,
                   help="Mip-Splatting 3D smoothing-filter factor (0 = off; paper default 0.2). "
                        "Band-limits every gaussian to the sampling rate of the nearest TRAIN "
                        "camera DURING training, so sub-pixel aliasing spikes never form. Baked "
                        "into the saved ckpt, so the renderer needs no change. Distinct from the "
                        "post-hoc filter probe (which broke models already full of such spikes).")
    p.add_argument("--mip3d_every", type=int, default=100,
                   help="recompute the 3D filter every N steps (also recomputed whenever "
                        "densification changes the gaussian count)")
    p.add_argument("--eps2d", type=float, default=-1.0,
                   help="2D-covariance low-pass (gsplat default 0.3; <0 = leave default). "
                        "Lower = sharper; audit r16 B1 for interpolated video test poses")
    p.add_argument("--texture_weight", type=float, default=0.0,
                   help="LEGS-inspired (arXiv 2606.07932, simplified to first-order Sobel): "
                        "weight the L1 term by local GT gradient magnitude, floor 1x, cap 6x. "
                        "0 = off. Plain mean-L1 is diluted by large flat regions (walls, "
                        "furniture), starving fine-texture regions (carpet fiber) of gradient "
                        "signal -- this rebalances toward the GT photo's own high-frequency "
                        "content, deriving the weight only from the training photo (Rule-10 "
                        "clean, no external data/net).")
    p.add_argument("--pose_opt", action="store_true",
                   help="Robust-GS-inspired (arXiv 2404.04211) TRAIN-VIEW pose refinement: a "
                        "learnable per-image SE3 residual (axis-angle + translation, "
                        "composed in the camera's own frame: R_new=dR@R0, t_new=dR@t0+dt) "
                        "corrects imperfect SfM/COLMAP poses. Applied ONLY to training-view "
                        "poses during optimization -- test poses (CSV) are never touched, so "
                        "this only improves the fitted Gaussians, not the render-time camera. "
                        "Fixes systematic geometric misalignment (manifests as blur/ghosting) "
                        "as opposed to up-weighting high-frequency pixels (texture_weight, "
                        "killed 20/07) -- architecturally cannot overfit sensor noise the same "
                        "way since it corrects ALL pixels via a rigid transform, not per-pixel.")
    p.add_argument("--pose_lr", type=float, default=1e-4,
                   help="Adam lr for the pose residual (both axis-angle and translation "
                        "components share this rate; deltas start at 0 = identity). v1/v2 "
                        "BOTH crashed (chair eval 69.37->45.96 / 47.36) at this default -- "
                        "nerfstudio issue #1635 (found 21/07) reports the identical symptom "
                        "(default camera-opt LR blurrier on low-texture-strong-edge images) "
                        "fixed by a 10x reduction (6e-4->6e-5); v3 should try 1e-5.")
    p.add_argument("--pose_warmup", type=int, default=0,
                   help="v3 FIX (22/07): v1 (independent per-view SE3, no constraint) and v2 "
                        "(+temporal-smoothness reg) BOTH crashed catastrophically on both "
                        "scenes despite pose deltas staying individually tiny -- divergence "
                        "traced to LATE training (final pose_t magnitudes were 2x+ larger "
                        "than an 8000-iter snapshot suggested), not the early cross-view-"
                        "consistency issue smoothness targeted. nerfstudio issue #1635's own "
                        "hypothesis: pose-opt at full strength from step 0, before gaussian "
                        "geometry has stabilized, compounds noise into a runaway spiral. "
                        "Delay pose_r/pose_t from receiving gradient (and the optimizer step) "
                        "until this iter -- same warmup-gate pattern as --uncertainty_warmup. "
                        "0 = off (reproduces v1/v2 behavior for regression testing).")
    p.add_argument("--min_opacity", type=float, default=0.005,
                   help="MCMCStrategy relocation threshold (gsplat default 0.005). Trick-hunt "
                        "22/07 + verified on our ckpts: 27-47%% of gaussians end at opacity<0.05 "
                        "(dead), wasting the cap budget. Raising this makes MCMC relocate "
                        "low-opacity gaussians more aggressively -> reclaim budget for hard "
                        "regions. Test 0.01, 0.02.")
    p.add_argument("--reg_stop", type=int, default=0,
                   help="stop applying opacity_reg/scale_reg after this step (0 = never stop, "
                        "the historical behaviour). DBS/UBS gate both to the densify window.")
    p.add_argument("--aniso_reg", type=float, default=0.0,
                   help="scale-anisotropy regularizer (trick-hunt: distinct from the killed "
                        "opacity/scale-reg-lowering). Penalizes log-scale spread beyond a ~10x "
                        "max-ratio cap -> discourages needle/sheet gaussians that cause floor "
                        "'sheet' artifacts and under-represent thin structures (chair legs, "
                        "tower braces). 0 = off.")
    p.add_argument("--lambda_pose_reg", type=float, default=0.01,
                   help="L2-to-identity weight on the pose residual, scaled by scene_scale "
                        "for the translation term -- keeps corrections small/plausible, "
                        "same role as --lambda_appreg for --app_affine")
    p.add_argument("--uncertainty_weight", action="store_true",
                   help="Fable-consult-2-inspired (Kendall&Gal aleatoric NLL, generalized "
                        "Laplace form): learned per-view 16x16 confidence field b(x)>0 "
                        "reweights the L1 term as |img-gt|/b + log(b) (reduces to plain L1 "
                        "at b=1). Unlike texture_weight (killed 20/07, chases HF and overfits "
                        "blur) this REMOVES attention from unreliable/blurry regions instead "
                        "of adding it, and the log(b) term makes escaping large residuals "
                        "costly -- a real equilibrium, not a free/hard mask (RobustNeRF-style "
                        "masking rejected on mechanism grounds: our blur is PERSISTENT across "
                        "nearly every frame, not transient/minority, so a hard mask would "
                        "starve those regions of gradient and could starve them of gaussians "
                        "via MCMC relocation too). Self-contained, no pretrained net, Rule-10 "
                        "trivially clear. 16x16 (not per-pixel) forces spatial smoothness.")
    p.add_argument("--uncertainty_lr", type=float, default=1e-3)
    p.add_argument("--uncertainty_warmup", type=int, default=3000,
                   help="b(x) stays fixed at 1 (plain L1) until this step -- early renders "
                        "are noise, don't let b chase them")
    p.add_argument("--lambda_pose_smooth", type=float, default=1.0,
                   help="FIX (21/07): --pose_opt v1 crashed (chair eval 69.37->45.96) because "
                        "147 independent per-view corrections broke multi-view triangulation "
                        "consistency -- MCMC read the inconsistency as 'needs more geometry' "
                        "and exploded to 3M gaussians in 8k/60k iters with oscillating SSIM, "
                        "even though each correction individually stayed tiny (verified: max "
                        "0.3deg rotation, max 0.24% scene_scale translation -- NOT a magnitude/ "
                        "drift bug). Views are video frames, sorted chronologically (frame_NNN "
                        "naming, confirmed) -- penalize squared difference between TEMPORALLY "
                        "ADJACENT frames' pose deltas so the trajectory correction is smooth/ "
                        "shared rather than independently fit per frame, matching what the "
                        "source literature (Robust-GS) actually recommends. 0 = off (v1 bug "
                        "reproducible for regression testing).")

    # ===================== B2: training-side attack on the blur floor =====================
    # bonsai's train photos vary 7.7x in sharpness (log_lapvar sd 0.727 nats ~= 0.73 px of
    # equivalent Gaussian sigma) and 84% of the held-out LPIPS error is already present on
    # views the model FULLY supervises. The model converges to a BLUR AVERAGE that matches no
    # individual photo. Two interventions, both changing WHICH photo the gaussians are asked
    # to match rather than filtering the rendered pixels afterwards (render-time frequency
    # operators are measured dead on this scene: GT-peeking ceiling +0.24, honest +0.05).
    p.add_argument("--sharp_csv", default=None,
                   help="B2 sidecar CSV of per-train-photo sharpness. Header: "
                        "name,frame,log_lapvar,reblur,log_hf025,sigma_init . Produced by "
                        "r42_bonsai78/b2_train/design_probe.py from the TRAIN photos only "
                        "(Rule-10 clean: no test image is ever read). Required by "
                        "--sw_mode != off and by --blur_view != off.")
    p.add_argument("--sharp_stat", default="log_lapvar",
                   choices=("log_lapvar", "reblur", "log_hf025"),
                   help="which column of --sharp_csv is 'sharpness'. log_lapvar and reblur "
                        "agree (r=+0.90 on the 220 train_sub frames); log_hf025 is CONTENT-"
                        "driven here (r=+0.26 with log_lapvar, opposite sign against score) "
                        "-- do not use it as a blur proxy.")
    # ---- (1) sharpness-weighted photometric loss ----
    p.add_argument("--sw_mode", default="off", choices=("off", "power", "tilt", "topk"),
                   help="per-training-view loss weight w_i = f(sharpness_i), normalised to "
                        "mean 1 over the views actually used. 'power': w=(s_i/median s)^p in "
                        "the LINEAR domain of the statistic (--sw_power). 'tilt': "
                        "w=exp(beta*z_i) with z the z-score of the statistic (--sw_beta) -- "
                        "scale-free, the correct form for a log-domain statistic. 'topk': "
                        "hard mask, w=1/F on the top --sw_topk fraction and 0 elsewhere. "
                        "NOTE this file trains ONE view per step, so a per-view weight is "
                        "just a per-step scalar on the loss and EVERY term (L1, SSIM, LPIPS) "
                        "can be weighted -- see --sw_scope.")
    p.add_argument("--sw_power", type=float, default=2.0,
                   help="exponent for --sw_mode power. On bonsai/reblur: p=1 -> effective "
                        "sample size 92%% of 220 views, p=2 -> 74%%, p=4 -> 46%%.")
    p.add_argument("--sw_beta", type=float, default=0.5,
                   help="tilt strength for --sw_mode tilt. On bonsai/log_lapvar: beta=0.5 "
                        "shifts the weighted-mean sharpness +0.366 nats at ESS 78%%; beta=1.0 "
                        "shifts +0.698 nats at ESS 44%%.")
    p.add_argument("--sw_topk", type=float, default=0.5,
                   help="keep-fraction for --sw_mode topk. WARNING (measured, probe Q2): an "
                        "UNSTRATIFIED top-50%% sharpness mask on bonsai opens a 390-frame "
                        "trajectory hole (vs max gap 30 for the full set) because sharpness "
                        "is autocorrelated at lag 10 -- the sharp frames come in RUNS. Always "
                        "pair with --sw_topk_stratify.")
    p.add_argument("--sw_topk_stratify", type=int, default=0,
                   help="if >0, apply --sw_topk WITHIN consecutive blocks of this many views "
                        "(name order), keeping round(F*B) per block. Preserves temporal "
                        "coverage: block=6 keep=3 holds the max gap to 60 frames while still "
                        "separating kept from dropped by 0.62 nats.")
    p.add_argument("--sw_wmin", type=float, default=0.05)
    p.add_argument("--sw_wmax", type=float, default=5.0,
                   help="clamp on w BEFORE the mean-1 renormalisation, so one freak frame "
                        "cannot dominate (log_lapvar p=2 unclamped reaches w=17).")
    p.add_argument("--sw_scope", default="photo", choices=("l1", "photo", "all"),
                   help="which terms w multiplies. 'l1' = the L1 term only; 'photo' = L1 + "
                        "(1-SSIM) + LPIPS (default -- the three terms the competition score "
                        "is made of); 'all' = also the opacity/scale/aniso regularisers, "
                        "which holds the photometric:regulariser RATIO fixed per step instead "
                        "of letting blurry views feel relatively more scale_reg.")
    p.add_argument("--sw_from", type=int, default=0,
                   help="apply the weighting only from this step (0 = whole run). Weighting "
                        "during densification changes WHERE gaussians are spawned as well as "
                        "how they are fit; --sw_from 15000 (== refine_stop) isolates the fit.")
    p.add_argument("--view_subset", default=None,
                   help="path to a newline-separated list of image file names; train on those "
                        "views ONLY. Used for the coverage-matched sharp-vs-blurry gate arm "
                        "(r42_bonsai78/b2_train/subsets/b6m3_{sharp,blurry,rand}.txt): every "
                        "block of 6 consecutive frames contributes exactly 3 views to every "
                        "arm, so view count and trajectory coverage are identical across arms "
                        "and the ONLY difference is a 0.62-nat sharpness shift.")
    # ---- (2) per-view learnable blur (BAD-Gaussians / Deblur-GS, cheapest form) ----
    p.add_argument("--blur_view", default="off", choices=("off", "frozen", "learn"),
                   help="explain each training photo's own blur with a PER-VIEW ISOTROPIC "
                        "GAUSSIAN applied to the RENDER before the loss, so the gaussians are "
                        "supervised toward the sharp consensus instead of the blur average. "
                        "One scalar per training view (220 params on the eval split). "
                        "'frozen' = sigma fixed at the value implied by the measured photo "
                        "sharpness (no degeneracy possible -- the cheapest mechanism test); "
                        "'learn' = that value is the INIT and sigma is optimised.")
    p.add_argument("--blur_ref_q", type=float, default=0.75,
                   help="quantile of the sharpness statistic taken as the 'sharp consensus' "
                        "the gaussians should reach: sigma_i = clip(gain*(q_ref - s_i), 0, "
                        "cap). q=0.75 leaves 25%% of views at sigma=0 -- those views are the "
                        "ONLY thing preventing the degenerate solution (model sharpens without "
                        "bound while every kernel widens to hide it), because invented detail "
                        "is not hidden on a zero-sigma view. Do NOT set q above ~0.9.")
    p.add_argument("--blur_gain", type=float, default=1.0,
                   help="px of Gaussian sigma per nat of log_lapvar deficit. A2 measured this "
                        "calibration at 1.00 nats per px on bonsai (3 frames spanning the "
                        "range). Only meaningful with --sharp_stat log_lapvar.")
    p.add_argument("--blur_sigma_max", type=float, default=2.5,
                   help="hard cap; sigma = 0.05 + (max-0.05)*sigmoid(raw). Sets the fixed "
                        "kernel radius ceil(3*max), so the conv shape never changes.")
    p.add_argument("--blur_lr", type=float, default=1e-2,
                   help="Adam lr on the raw per-view blur parameter. Each view is revisited "
                        "~iters/n_views times (136x at 30k/220) and Adam steps are ~lr in "
                        "size, so 1e-2 gives each sigma ~1.4 px of total travel over the run "
                        "-- matched to the 0-2.5 px range. 1e-3 would move it 0.14 px, i.e. "
                        "a no-op arm dressed up as a treatment.")
    p.add_argument("--blur_warmup", type=int, default=3000,
                   help="sigma is held at its init until this step (same gate as "
                        "--uncertainty_warmup): early renders are noise and sigma would "
                        "collapse chasing them.")
    p.add_argument("--blur_from", type=int, default=0,
                   help="step from which the kernel is APPLIED at all (0 = always). Applying "
                        "it only after densification (--blur_from 15000) is the conservative "
                        "variant: the gaussian population is decided by the ordinary loss and "
                        "only the final fit is deblurred.")
    p.add_argument("--lambda_blur", type=float, default=0.02,
                   help="L2 pull on sigma, toward --blur_reg. The soft half of the "
                        "anti-degeneracy guard: the runaway direction is (model sharper, all "
                        "sigma larger), so any downward pull on sigma opposes it. NB "
                        "--scale_reg 0.1 pushes gaussians SMALLER i.e. sharper, so it pushes "
                        "ALONG the degenerate direction -- if sigma runs to the cap, lower "
                        "scale_reg before raising lambda_blur.")
    p.add_argument("--blur_reg", default="init", choices=("init", "zero"),
                   help="'init' anchors sigma to the measured-sharpness init (keeps the "
                        "relative pattern, allows a common-mode shift); 'zero' pulls the "
                        "common mode all the way down (stronger anti-degeneracy, but fights "
                        "the intended per-view spread too).")
    p.add_argument("--lambda_blur_smooth", type=float, default=0.0,
                   help="squared-difference penalty between temporally adjacent views' sigma. "
                        "Justified: frame sharpness is autocorrelated r=+0.70 at lag 10. Also "
                        "makes the learned sigma directly predictable at test time by the A2 "
                        "neighbour regressor. 0 = off; try 0.1.")
    args = p.parse_args()

    views, K_np, (W, H), xyz, rgb, k1 = load_scene(args.source, args.images)
    if args.view_subset:
        # BEFORE scene_scale is derived from the camera centers -- same ordering
        # requirement as the on-disk filter in load_scene (audit round 5 finding 1).
        want = set(l.strip() for l in open(args.view_subset) if l.strip())
        want |= set(os.path.splitext(x)[0] for x in list(want))
        kept = [v for v in views if v["name"] in want
                or os.path.splitext(v["name"])[0] in want]
        assert kept, f"--view_subset {args.view_subset} matched 0 of {len(views)} views"
        print(f"view_subset {os.path.basename(args.view_subset)}: training on "
              f"{len(kept)}/{len(views)} views")
        views = kept
    print(f"{len(views)} views, {len(xyz)} init points, K=[[{K_np[0,0]:.1f},0,{K_np[0,2]:.1f}],"
          f"[0,{K_np[1,1]:.1f},{K_np[1,2]:.1f}]], size {W}x{H}")
    if args.init_clip > 0:
        # consult #3 / M2: glossy surfaces make COLMAP triangulate REFLECTED features into
        # mirror-world points far outside the capture volume (bonsai: p100/p95 = 9.9 vs 3.5-4.5
        # on honest scenes). Those gaussians float mid-air from opposing views and may seed the
        # MCMC relocation collapse. Drop init points beyond camera-hull-radius x clip.
        import numpy as _np
        w2cs_ = _np.stack([v["w2c"] for v in views])
        Cs = -_np.einsum("nij,nj->ni", w2cs_[:, :3, :3].transpose(0, 2, 1), w2cs_[:, :3, 3])
        ctr = Cs.mean(0)
        hull = _np.linalg.norm(Cs - ctr, axis=1).max()
        keep = _np.linalg.norm(xyz - ctr, axis=1) <= hull * args.init_clip
        print(f"init_clip {args.init_clip}: hull R={hull:.2f}, keeping {keep.sum()}/{len(xyz)} "
              f"points (dropped {(~keep).sum()} beyond {hull * args.init_clip:.2f})")
        xyz, rgb = xyz[keep], rgb[keep]
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
    # audit r16 B1: eps2d is gsplat's hardcoded 0.3 2D-covariance low-pass (anti-alias margin).
    # Our video test poses are 3-4deg interpolations -> little aliasing risk -> 0.3 over-smooths
    # the LPIPS-deficit scenes. Expose it; train AND render must use the same value (saved in ckpt).
    if args.eps2d >= 0:
        rast_extra["eps2d"] = args.eps2d
    # depth-prior: render expected depth alongside RGB (RGB+ED -> 4th channel = expected depth)
    if args.depth_prior > 0:
        rast_extra["render_mode"] = "RGB+ED"

    # scene scale for lr/noise (world coords untouched)
    centers = np.stack([-v["w2c"][:3, :3].T @ v["w2c"][:3, 3] for v in views])
    scene_scale = 1.1 * float(np.max(np.linalg.norm(centers - centers.mean(0), axis=1)))
    print(f"scene_scale {scene_scale:.3f}")

    if args.init_ckpt:
        # warm-start from OUR OWN ckpt.pt (same param layout; UT flag/k1 carried)
        ick = torch.load(args.init_ckpt, map_location=dev, weights_only=False)
        means, scales = ick["splats"]["means"].to(dev), ick["splats"]["scales"].to(dev)
        quats, opacities = ick["splats"]["quats"].to(dev), ick["splats"]["opacities"].to(dev)
        sh0, shN = ick["splats"]["sh0"].to(dev), ick["splats"]["shN"].to(dev)
        print(f"Warm-start: {len(means)} gaussians from {args.init_ckpt} (ut={ick.get('ut')})")
        assert args.cap_max >= len(means), f"cap_max {args.cap_max} < init N {len(means)}"
    elif args.init_ply:
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

        if args.sky_dome > 0:
            # enclosing sphere at 3x the camera-hull radius, points on a Fibonacci sphere
            # (uniform), large initial scale, low opacity, neutral-grey color. Gives MCMC a
            # geometric target for the low-parallax open-sky background instead of floaters.
            cen = torch.tensor(centers.mean(0), device=dev, dtype=torch.float32)
            hullR = float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
            R = 3.0 * hullR
            m = args.sky_dome
            gi = torch.arange(m, device=dev, dtype=torch.float32)
            phi = torch.acos(1.0 - 2.0 * (gi + 0.5) / m)
            gold = math.pi * (3.0 - math.sqrt(5.0))
            theta = gold * gi
            dome = torch.stack([torch.sin(phi) * torch.cos(theta),
                                torch.sin(phi) * torch.sin(theta),
                                torch.cos(phi)], dim=1) * R + cen
            means = torch.cat([means, dome], dim=0)
            dome_scale = math.log(2.0 * R / math.sqrt(m))  # ~point-spacing, full sphere coverage
            scales = torch.cat([scales, torch.full((m, 3), dome_scale, device=dev)], dim=0)
            dq = torch.zeros(m, 4, device=dev); dq[:, 0] = 1.0
            quats = torch.cat([quats, dq], dim=0)
            opacities = torch.cat([opacities, torch.logit(torch.full((m,), 0.05, device=dev))], dim=0)
            sh0 = torch.cat([sh0, RGB2SH(torch.full((m, 3), 0.5, device=dev)).unsqueeze(1)], dim=0)
            shN = torch.cat([shN, torch.zeros(m, (args.sh_degree + 1) ** 2 - 1, 3, device=dev)], dim=0)
            print(f"sky_dome: +{m} points on sphere R={R:.1f} about scene center")

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means), "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats), "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0), "shN": torch.nn.Parameter(shN),
    }).to(dev)
    lrs = {"means": args.means_lr * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {n: torch.optim.Adam([{"params": params[n], "lr": lrs[n], "name": n}],
                                      eps=1e-15) for n in params.keys()}

    # audit 23/07: absgrad only populates a buffer that DefaultStrategy consumes; MCMCStrategy
    # never reads it (relocation is opacity-driven) -- the flag is a placebo here.
    assert not args.absgrad, "--absgrad is a silent no-op under MCMCStrategy; refusing to run a placebo A/B"
    # B2 compatibility guards
    assert not (args.blur_view != "off" and (args.bilagrid or args.ppisp)), \
        "--blur_view with --bilagrid/--ppisp: those are spatially varying and do NOT commute " \
        "with the blur kernel; the composition would be ill-defined"
    assert not (args.blur_view != "off" and args.depth_prior > 0), \
        "--blur_view + --depth_prior untested: the depth channel is not blurred, so the two " \
        "supervise inconsistent versions of the same render"
    if args.blur_view == "learn" and args.scale_reg >= 0.1:
        print(f"WARNING: --blur_view learn with --scale_reg {args.scale_reg}: scale_reg pushes "
              f"gaussians SMALLER (sharper), i.e. ALONG the kernel-absorbs-the-model runaway "
              f"direction. Watch the sigma[mean] diagnostic every 1000 steps.")
    strategy = MCMCStrategy(cap_max=args.cap_max, min_opacity=args.min_opacity,
                            refine_stop_iter=args.refine_stop,
                            noise_injection_stop_iter=args.noise_stop, verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    K = torch.tensor(K_np, device=dev)
    w2cs = torch.tensor(np.stack([v["w2c"] for v in views]), device=dev)

    # idea: TRAIN-VIEW pose refinement (Robust-GS-inspired). One SE3 residual per training
    # image, composed in the camera's own frame; test poses (CSV, render_gsplat.py) are never
    # touched -- only the photometric supervision signal improves, hence only the fitted
    # Gaussians. delta_r is axis-angle (Rodrigues), starts at 0 -> R_delta = I at init.
    pose_r = pose_t = pose_opt_ = None
    if args.pose_opt:
        pose_r = torch.nn.Parameter(torch.zeros(len(views), 3, device=dev))
        pose_t = torch.nn.Parameter(torch.zeros(len(views), 3, device=dev))
        pose_opt_ = torch.optim.Adam([pose_r, pose_t], lr=args.pose_lr, eps=1e-15)

    def _rodrigues(phi):  # [3] axis-angle -> [3,3] rotation, safe at phi->0
        theta = phi.norm().clamp(min=1e-8)
        axis = phi / theta
        K_ = torch.zeros(3, 3, device=phi.device, dtype=phi.dtype)
        K_[0, 1], K_[0, 2] = -axis[2], axis[1]
        K_[1, 0], K_[1, 2] = axis[2], -axis[0]
        K_[2, 0], K_[2, 1] = -axis[1], axis[0]
        eye3 = torch.eye(3, device=phi.device, dtype=phi.dtype)
        return eye3 + torch.sin(theta) * K_ + (1 - torch.cos(theta)) * (K_ @ K_)

    def corrected_w2c(vi):
        R0, t0 = w2cs[vi, :3, :3], w2cs[vi, :3, 3]
        dR = _rodrigues(pose_r[vi])
        R_new, t_new = dR @ R0, dR @ t0 + pose_t[vi]
        w2c_new = torch.eye(4, device=dev, dtype=w2cs.dtype)
        w2c_new = w2c_new.clone()
        w2c_new[:3, :3], w2c_new[:3, 3] = R_new, t_new
        return w2c_new

    # Fable-consult-2 aleatoric uncertainty: b(x)=softplus(raw), init so softplus(init)=1.0
    unc_b = unc_opt = None
    if args.uncertainty_weight:
        init_val = math.log(math.e - 1.0)  # softplus^-1(1.0) (math imported module-level)
        unc_b = torch.nn.Parameter(torch.full((len(views), 16, 16), init_val, device=dev))
        unc_opt = torch.optim.Adam([unc_b], lr=args.uncertainty_lr)

    # ==================== B2 (1): per-view sharpness weights ====================
    view_names = [v["name"] for v in views]
    sharp = None
    if args.sw_mode != "off" or args.blur_view != "off":
        assert args.sharp_csv, "--sw_mode/--blur_view need --sharp_csv"
        sharp = load_sharp_csv(args.sharp_csv, args.sharp_stat)
        missing = [n for n in view_names if n not in sharp]
        assert not missing, f"{len(missing)} views missing from {args.sharp_csv}: {missing[:3]}"
    sw = None
    if args.sw_mode != "off":
        sw = torch.tensor(build_view_weights(args, view_names, sharp),
                          device=dev, dtype=torch.float32)

    # ==================== B2 (2): per-view blur kernel ====================
    # Render sharp, convolve with THIS view's own kernel, then compare to THIS view's photo.
    # The photo's blur is then explained by the kernel instead of being baked into the
    # gaussians, so the gaussians converge to the sharp consensus rather than the blur
    # average. At test time the kernel is dropped (or set from the A2 neighbour predictor).
    blur_raw = blur_opt = None
    blur_sig0 = None
    blur_radius = int(math.ceil(3.0 * args.blur_sigma_max))
    BLUR_FLOOR = 0.05
    if args.blur_view != "off":
        sig0 = blur_sigma_init(args, view_names, sharp)
        blur_sig0 = torch.tensor(sig0, device=dev, dtype=torch.float32)
        if args.blur_view == "learn":
            # sigma = FLOOR + (max-FLOOR)*sigmoid(raw): bounded above by the cap (hard
            # anti-runaway guard) and strictly positive below (the kernel stays well
            # conditioned and sigma keeps receiving gradient at the bottom of the range).
            u = ((blur_sig0 - BLUR_FLOOR) / (args.blur_sigma_max - BLUR_FLOOR)).clamp(1e-4, 1 - 1e-4)
            blur_raw = torch.nn.Parameter(torch.logit(u))
            blur_opt = torch.optim.Adam([blur_raw], lr=args.blur_lr)
        print(f"blur_view={args.blur_view} radius {blur_radius}px ({2*blur_radius+1}-tap "
              f"separable) applied from step {args.blur_from}, sigma learnable from "
              f"{args.blur_warmup if args.blur_view == 'learn' else 'never'}")

    def blur_sigmas():
        """[n_views] current sigma in px."""
        if args.blur_view == "frozen":
            return blur_sig0
        return BLUR_FLOOR + (args.blur_sigma_max - BLUR_FLOOR) * torch.sigmoid(blur_raw)

    # monocular depth-prior targets (Depth-Anything, precomputed .npy per train image); the loss
    # is a scale-and-shift-invariant Pearson correlation, so raw mono depth (unknown scale) is fine
    depth_cache = [None] * len(views)
    if args.depth_prior > 0:
        assert args.depth_dir, "--depth_prior needs --depth_dir"
        for vi, v in enumerate(views):
            stem = os.path.splitext(v["name"])[0]
            dp = os.path.join(args.depth_dir, stem + ".npy")
            assert os.path.exists(dp), f"missing depth prior {dp}"
            depth_cache[vi] = torch.from_numpy(np.load(dp)).float()

    # GT cache on CPU (uint8), like FastGS --data_device cpu
    from PIL import Image
    gt_cache = [None] * len(views)
    wmap_cache = [None] * len(views)  # texture_weight: static per view (derived from GT only)
    _sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                            device=dev).view(1, 1, 3, 3)
    _sobel_y = _sobel_x.transpose(2, 3)

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
    ema_params = None
    mip_filt = None
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
        sh_deg = args.sh_degree if (args.init_ply or args.init_ckpt) \
            else min(step // 1000, args.sh_degree)
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        pose_active = args.pose_opt and step >= args.pose_warmup
        viewmat = corrected_w2c(vi)[None] if pose_active else w2cs[vi:vi + 1]
        rast_scales = torch.exp(params["scales"])
        rast_opac = torch.sigmoid(params["opacities"])
        if args.mip3d > 0:
            # recompute on schedule AND whenever densification changed N (the filter is
            # per-gaussian, so a stale one would be silently misaligned after a refine)
            if (mip_filt is None or mip_filt.shape[0] != rast_scales.shape[0]
                    or step % args.mip3d_every == 0):
                mip_filt = compute_mip3d_filter(params["means"].detach(), w2cs, K, W, H,
                                                args.mip3d)
            rast_scales, rast_opac = mip3d_apply(rast_scales, rast_opac, mip_filt)
        render, alpha, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=rast_scales,
            opacities=rast_opac,
            colors=colors, viewmats=viewmat, Ks=K[None], width=W, height=H,
            sh_degree=sh_deg, near_plane=0.01, packed=False, absgrad=args.absgrad,
            **rast_extra)
        # loss on UNCLAMPED colors (upstream convention: clamping zeroes
        # gradients at saturated pixels, exactly where early training
        # overshoots); clamp only for the LPIPS term (parity with scoring)
        if args.depth_prior > 0:
            img = render[0][..., :3]       # [H,W,3]
            depth_r = render[0][..., 3]    # [H,W] expected depth
        else:
            img = render[0]  # [H,W,3]

        strategy.step_pre_backward(params, optimizers, state, step, info)

        # ---- B2 (2): convolve the SHARP render with THIS view's own blur kernel ----
        # Placed immediately after rasterization, before the appearance transforms: the PSF
        # physically acts on radiance before the ISP. (app_M is a per-pixel affine and the
        # kernel sums to 1, so the two commute exactly anyway; --bilagrid/--ppisp are
        # spatially varying and do NOT commute -- do not combine them with --blur_view.)
        # The depth channel is never blurred.
        cur_sigma = None
        if args.blur_view != "off" and step >= args.blur_from:
            sigs = blur_sigmas()
            if args.blur_view == "learn" and step < args.blur_warmup:
                sigs = sigs.detach()
            cur_sigma = sigs[vi]
            if float(cur_sigma) > BLUR_FLOOR + 1e-4 or args.blur_view == "learn":
                img = gauss_blur1(img, cur_sigma, blur_radius)

        if app_M is not None:
            img = img @ app_M[vi, :, :3].T + app_M[vi, :, 3]
        if bil is not None:
            img = bil_slice(bil, bil_xy, img.unsqueeze(0),
                            torch.tensor([[vi]], device=dev))["rgb"].squeeze(0)
        if pp is not None:
            img = pp(rgb=img, pixel_coords=pp_xy, resolution=(W, H),
                     camera_idx=0, frame_idx=vi)

        if args.uncertainty_weight and step >= args.uncertainty_warmup:
            b = F.softplus(unc_b[vi])[None, None]  # [1,1,16,16]
            b_map = F.interpolate(b, size=(H, W), mode="bilinear",
                                  align_corners=False).squeeze(0).squeeze(0).unsqueeze(-1)
            l1 = ((img - gt).abs() / b_map + torch.log(b_map)).mean()
        elif args.texture_weight > 0:
            if wmap_cache[vi] is None:
                gray = gt.mean(-1, keepdim=True).permute(2, 0, 1).unsqueeze(0)  # [1,1,H,W]
                gx = F.conv2d(gray, _sobel_x, padding=1)
                gy = F.conv2d(gray, _sobel_y, padding=1)
                gmag = (gx.pow(2) + gy.pow(2)).sqrt().squeeze(0).squeeze(0)  # [H,W]
                w = (gmag / (gmag.mean() + 1e-6)).clamp(max=6.0)
                wmap_cache[vi] = (1.0 + args.texture_weight * w).detach().unsqueeze(-1)  # [H,W,1]
            l1 = (wmap_cache[vi] * (img - gt).abs()).mean()
        else:
            l1 = (img - gt).abs().mean()
        # ---- B2 (1): this view's sharpness weight (batch is ONE view per step, so a
        # per-view weight is simply a per-step scalar and every term can carry it) ----
        wv = sw[vi] if (sw is not None and step >= args.sw_from) else None
        if wv is not None and args.sw_scope == "l1":
            l1 = wv * l1
        w_photo = wv if (wv is not None and args.sw_scope in ("photo", "all")) else None
        w_reg = wv if (wv is not None and args.sw_scope == "all") else 1.0
        ssim = fused_ssim(img.permute(2, 0, 1).unsqueeze(0),
                          gt.permute(2, 0, 1).unsqueeze(0))
        if args.pure_l2:
            # DIAGNOSTIC (audit r12): ask the model for NOTHING but PSNR. This isolates
            # whether the ~27dB train fit is a loss/reg artifact or a real capacity/content
            # wall — the one lever the D5 registration oracle cannot see. Pure MSE, no SSIM.
            loss = (img.clamp(0.0, 1.0) - gt).pow(2).mean()
        elif args.metric_loss:
            # exact score-matched loss: minimizing this maximizes the
            # competition score (see --metric_loss help). MSE on the CLAMPED
            # render — the scorer measures clamped uint8 pixels, and an
            # unclamped MSE would chase gradients outside the display range.
            mse = (img.clamp(0.0, 1.0) - gt).pow(2).mean()
            loss = 0.3 * (1 - ssim) + 0.02606 * torch.log(mse.clamp(min=1e-8))
        else:
            loss = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * (1 - ssim)
        if w_photo is not None:
            loss = w_photo * loss
        # --reg_stop: DBS/UBS gate opacity_reg and scale_reg to the DENSIFICATION WINDOW only.
        # gsplat MCMC (and we) apply them for the whole run, so on bonsai the scale penalty keeps
        # shaping a frozen primitive set for 15k steps after relocation has ended. Default keeps
        # the historical behaviour; --reg_stop 15000 reproduces the DBS/UBS schedule.
        if step < pp_act_step and (args.reg_stop <= 0 or step < args.reg_stop):
            loss += w_reg * args.opacity_reg * torch.sigmoid(params["opacities"]).mean()
            loss += w_reg * args.scale_reg * torch.exp(params["scales"]).mean()
        if args.aniso_reg > 0:
            # penalize per-gaussian log-scale spread beyond a ~10x max/min ratio (ln10=2.30);
            # scales are stored in log space so (max-min) over the 3 axes == log(ratio)
            sp = params["scales"].max(dim=1).values - params["scales"].min(dim=1).values
            loss += w_reg * args.aniso_reg * (sp - 2.302585).clamp(min=0.0).mean()
        # ---- B2 (2): keep the per-view sigma from running away ----
        # The degeneracy is (gaussians arbitrarily sharp) x (every kernel arbitrarily wide):
        # it leaves the blurred render unchanged, so the photometric loss cannot see it. Four
        # guards, in order of strength:
        #   (a) HARD  sigma <= --blur_sigma_max via the sigmoid parameterisation;
        #   (b) HARD  the --blur_ref_q quantile pins ~25% of views at sigma ~ 0 -- on those
        #             views nothing is hidden, so invented detail is paid for immediately;
        #   (c) SOFT  the L2 pull below (toward the init, or toward zero);
        #   (d) SOFT  finite gaussian capacity under cap_max.
        # --scale_reg pushes gaussians SMALLER, i.e. ALONG the runaway direction: watch
        # mean(sigma) in the step log and cut scale_reg before raising lambda_blur.
        if args.blur_view == "learn" and step >= args.blur_warmup:
            sg_all = blur_sigmas()
            if args.lambda_blur > 0:
                tgt = blur_sig0 if args.blur_reg == "init" else torch.zeros_like(sg_all)
                loss = loss + args.lambda_blur * (sg_all - tgt).pow(2).mean()
            if args.lambda_blur_smooth > 0:
                loss = loss + args.lambda_blur_smooth * (sg_all[1:] - sg_all[:-1]).pow(2).mean()
        if args.depth_prior > 0:
            # scale-and-shift-invariant monocular depth prior (SurgicalGaussian/Endo-4DGS class):
            # maximize Pearson corr between rendered expected depth and the frozen Depth-Anything
            # prediction. Mono depth is DISPARITY-like (near=large); gsplat depth is metric
            # (near=small), so a NEGATIVE correlation is the geometric match -> loss = 1 + corr.
            dt = depth_cache[vi].to(dev)
            if dt.shape != depth_r.shape:
                dt = F.interpolate(dt[None, None], size=depth_r.shape, mode="bilinear",
                                   align_corners=False)[0, 0]
            dr = depth_r.reshape(-1)
            dt = dt.reshape(-1)
            dr = dr - dr.mean(); dt = dt - dt.mean()
            corr = (dr * dt).sum() / (dr.norm() * dt.norm() + 1e-8)
            loss = loss + args.depth_prior * (1.0 + corr)
        if app_M is not None:
            loss += args.lambda_appreg * (
                (app_M[vi, :, :3] - app_eye).pow(2).sum() + app_M[vi, :, 3].pow(2).sum())
        if pose_active:
            loss += args.lambda_pose_reg * (
                pose_r[vi].pow(2).sum() + (pose_t[vi] / scene_scale).pow(2).sum())
            if args.lambda_pose_smooth > 0:
                # global term (all views, not just vi): identity-reg alone leaves neighboring
                # frames free to disagree, which is exactly what broke v1. views are sorted
                # chronologically (frame_NNNNNN naming) so consecutive array index == temporal
                # adjacency; train_sub eval-holes create occasional gaps but order is preserved.
                loss += args.lambda_pose_smooth * (
                    (pose_r[1:] - pose_r[:-1]).pow(2).sum()
                    + ((pose_t[1:] - pose_t[:-1]) / scene_scale).pow(2).sum())
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
            lp = args.lambda_lpips * lpips_net(
                img.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0) * 2 - 1,
                gt.permute(2, 0, 1).unsqueeze(0) * 2 - 1).squeeze()
            loss += (w_photo * lp) if w_photo is not None else lp

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
        if pose_active:
            pose_opt_.step()
            pose_opt_.zero_grad(set_to_none=True)
        if args.uncertainty_weight and step >= args.uncertainty_warmup:
            unc_opt.step()
            unc_opt.zero_grad(set_to_none=True)
        if blur_opt is not None:
            if step >= args.blur_warmup:
                blur_opt.step()
            blur_opt.zero_grad(set_to_none=True)
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

        # EMA of gaussian params (trick-hunt, AsymGS-inspired but simplified): only AFTER
        # refine_stop, when MCMC topology is frozen (no relocation/densification changing
        # gaussian identity/count) -> a plain tensor EMA is valid with zero clone/split
        # bookkeeping. Denoises the late-training SGLD/LPIPS-phase jitter. Saved in place of raw.
        if args.ema_decay > 0 and step >= args.refine_stop:
            if ema_params is None:
                ema_params = {n: params[n].detach().clone() for n in params.keys()}
            else:
                for n in params.keys():
                    ema_params[n].mul_(args.ema_decay).add_(params[n].detach(), alpha=1 - args.ema_decay)

        if step % 1000 == 0 or step == args.iters - 1:
            diag = ""
            if args.pose_opt:
                rmag = pose_r.norm(dim=-1)
                tmag = pose_t.norm(dim=-1)
                diag = (f" pose_r[mean/max] {rmag.mean():.5f}/{rmag.max():.5f} "
                        f"pose_t[mean/max] {tmag.mean():.5f}/{tmag.max():.5f}")
            if args.blur_view != "off":
                with torch.no_grad():
                    sg = blur_sigmas()
                    d = (sg - blur_sig0)
                # THE early-warning line. If sigma_mean climbs monotonically and
                # frac@cap grows, the kernel is absorbing the model and the run is
                # heading for the degenerate solution -- kill it, do not wait for 30k.
                diag += (f" sigma[min/mean/max] {sg.min():.3f}/{sg.mean():.3f}/{sg.max():.3f}"
                         f" d_init[mean] {d.mean():+.3f} frac@cap "
                         f"{(sg > args.blur_sigma_max - 0.02).float().mean():.3f}")
            print(f"[{step}] loss {loss.item():.4f} l1 {l1.item():.4f} "
                  f"ssim {ssim.item():.4f} N {len(params['means'])}{diag}")

    os.makedirs(args.out, exist_ok=True)
    save_splats = (ema_params if ema_params is not None
                   else {n: params[n].detach() for n in params.keys()})
    if ema_params is not None:
        print(f"Saving EMA-averaged params (decay {args.ema_decay}, from refine_stop {args.refine_stop})")
    if args.mip3d > 0:
        # BAKE the filter into the saved scales/opacities using the FINAL means, so the
        # stock renderer reproduces training-time appearance with no renderer change.
        filt = compute_mip3d_filter(save_splats["means"].detach(), w2cs, K, W, H, args.mip3d)
        s_new, o_new = mip3d_apply(torch.exp(save_splats["scales"].detach()),
                                   torch.sigmoid(save_splats["opacities"].detach()), filt)
        save_splats = dict(save_splats)
        save_splats["scales"] = torch.log(s_new.clamp_min(1e-12))
        save_splats["opacities"] = torch.logit(o_new.clamp(1e-6, 1 - 1e-6))
        print(f"Baked mip3d filter (factor {args.mip3d}) into ckpt: "
              f"median sigma {filt.median().item():.6g}, mean opacity "
              f"{torch.sigmoid(save_splats['opacities']).mean().item():.4f}")
    torch.save({"splats": {n: save_splats[n].detach().cpu() for n in save_splats.keys()},
                "sh_degree": args.sh_degree, "K": K_np, "wh": (W, H),
                "ut": args.ut, "k1": k1, "eps2d": args.eps2d},
               os.path.join(args.out, "ckpt.pt"))
    print(f"Saved {len(save_splats['means'])} gaussians to {args.out}/ckpt.pt")
    if args.blur_view != "off":
        # B2 (3) hand-off: the renderer regresses sigma onto a TEST frame index from these
        # train-view sigmas (A2 Nadaraya-Watson, h=13 frames). Frame indices are parsed from
        # the image names so the regression axis needs nothing else.
        with torch.no_grad():
            sg = blur_sigmas().detach().cpu()
        torch.save({"sigma": sg, "sigma_init": blur_sig0.detach().cpu(),
                    "names": view_names, "mode": args.blur_view,
                    "sigma_max": args.blur_sigma_max, "floor": BLUR_FLOOR,
                    "radius": blur_radius, "sharp_stat": args.sharp_stat,
                    "ref_q": args.blur_ref_q, "gain": args.blur_gain},
                   os.path.join(args.out, "blur.pt"))
        d = (sg - blur_sig0.detach().cpu())
        print(f"Saved blur.pt: sigma min {sg.min():.3f} med {sg.median():.3f} "
              f"max {sg.max():.3f} mean {sg.mean():.3f} px | moved from init by mean "
              f"{d.mean():+.3f} (abs {d.abs().mean():.3f}) px | corr(sigma, sigma_init) "
              f"{float(np.corrcoef(sg.numpy(), blur_sig0.detach().cpu().numpy())[0,1]):+.3f}")
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
