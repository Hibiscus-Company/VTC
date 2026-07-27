#!/usr/bin/env python
"""Standalone 2DGS (2D Gaussian Splatting) trainer -- a MAXIMALLY-DECORRELATED ensemble member.

Our production trainer (train_gsplat.py) uses 3D Gaussians + MCMC densification. gsplat's
rasterization_2dgs uses 2D surfels + adaptive density control (DefaultStrategy -- MCMC does NOT
support 2dgs). This is a SEPARATE trainer so the production path is untouched. 2dgs typically
trades a little PSNR for much better geometry/normals; the bet is that as a different-PRIMITIVE
member it ADDS to the pixel-mean ensemble (composition transfers >=1x -- the one proven lever).

Self-contained: trains, renders eval poses (or --csv test poses), saves ckpt + PNGs. Reuses
load_scene/knn_mean_dist from train_gsplat; no changes to the production trainer.
"""
import argparse, os, sys, random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from gsplat_track.train_gsplat import load_scene, knn_mean_dist
from utils.sh_utils import RGB2SH
from fused_ssim import fused_ssim
from gsplat import rasterization_2dgs
from gsplat.strategy import DefaultStrategy
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--images", default="images")
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--means_lr", type=float, default=1.6e-4)
    p.add_argument("--ssim_lambda", type=float, default=0.2)
    p.add_argument("--lambda_lpips", type=float, default=0.1)
    p.add_argument("--lpips_from", type=int, default=15000)
    p.add_argument("--lambda_normal", type=float, default=0.05,
                   help="2dgs normal-consistency (align splat normals to depth-gradient normals)")
    p.add_argument("--lambda_dist", type=float, default=100.0,
                   help="2dgs depth-distortion (concentrate surfels on surfaces); paper ~100-1000")
    p.add_argument("--dist_from", type=int, default=3000)
    p.add_argument("--normal_from", type=int, default=7000)
    p.add_argument("--refine_start", type=int, default=500)
    p.add_argument("--refine_stop", type=int, default=15000)
    p.add_argument("--reset_every", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", default=None, help="render these poses at the end instead of nothing")
    p.add_argument("--png_dir", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    views, K_np, (W, H), xyz, rgb, k1 = load_scene(args.source, args.images)
    centers = np.stack([-v["w2c"][:3, :3].T @ v["w2c"][:3, 3] for v in views])
    scene_scale = 1.1 * float(np.max(np.linalg.norm(centers - centers.mean(0), axis=1)))
    print(f"2dgs: {len(views)} views, {len(xyz)} init pts, scene_scale {scene_scale:.3f}")

    means = torch.tensor(xyz, device=dev)
    scales = torch.log(torch.tensor(knn_mean_dist(xyz), device=dev, dtype=torch.float32)
                       .clamp(min=1e-7)).unsqueeze(-1).repeat(1, 3)
    quats = torch.zeros(len(xyz), 4, device=dev); quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((len(xyz),), 0.1, device=dev))
    sh0 = RGB2SH(torch.tensor(rgb / 255.0, device=dev)).unsqueeze(1)
    shN = torch.zeros(len(xyz), (args.sh_degree + 1) ** 2 - 1, 3, device=dev)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means), "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats), "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0), "shN": torch.nn.Parameter(shN),
    }).to(dev)
    lrs = {"means": args.means_lr * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {n: torch.optim.Adam([{"params": params[n], "lr": lrs[n], "name": n}], eps=1e-15)
                  for n in params.keys()}

    strategy = DefaultStrategy(refine_start_iter=args.refine_start, refine_stop_iter=args.refine_stop,
                               reset_every=args.reset_every, refine_every=100,
                               key_for_gradient="gradient_2dgs", verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    K = torch.tensor(K_np, device=dev)
    w2cs = torch.tensor(np.stack([v["w2c"] for v in views]), device=dev)
    gt_cache = [None] * len(views)
    lpips_net = None

    def render(vi_w2c, step, sh_deg):
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        return rasterization_2dgs(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=colors,
            viewmats=vi_w2c, Ks=K[None], width=W, height=H, sh_degree=sh_deg,
            near_plane=0.01, packed=False, render_mode="RGB")

    order = list(range(len(views)))
    for step in range(args.iters):
        vi = order[step % len(order)]
        if step % len(order) == 0:
            random.shuffle(order)
        if gt_cache[vi] is None:
            gt_cache[vi] = torch.from_numpy(
                np.asarray(Image.open(views[vi]["path"]).convert("RGB"), dtype=np.uint8))
        gt = gt_cache[vi].to(dev).float().div_(255.0)
        sh_deg = min(step // 1000, args.sh_degree)

        (rc, ra, rn, rn_depth, rdist, rmed, meta) = render(w2cs[vi][None], step, sh_deg)
        strategy.step_pre_backward(params, optimizers, state, step, meta)
        img = rc[0][..., :3]

        l1 = (img - gt).abs().mean()
        ssim = fused_ssim(img.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0))
        loss = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * (1 - ssim)
        # 2dgs regularizers (ramped in, like the paper): normal consistency + depth distortion
        if args.lambda_dist > 0 and step >= args.dist_from:
            loss = loss + args.lambda_dist * rdist.mean()
        if args.lambda_normal > 0 and step >= args.normal_from \
                and rn is not None and rn_depth is not None and rn.shape == rn_depth.shape:
            # rn, rn_depth: [C,H,W,3]; consistency = 1 - dot(splat normal, depth-normal)
            ncons = (1.0 - (rn * rn_depth).sum(dim=-1)).mean()
            loss = loss + args.lambda_normal * ncons
        if args.lambda_lpips > 0 and step > args.lpips_from:
            if lpips_net is None:
                import lpips
                lpips_net = lpips.LPIPS(net="vgg").to(dev)
                for q in lpips_net.parameters():
                    q.requires_grad_(False)
            loss = loss + args.lambda_lpips * lpips_net(
                img.clamp(0, 1).permute(2, 0, 1).unsqueeze(0) * 2 - 1,
                gt.permute(2, 0, 1).unsqueeze(0) * 2 - 1).mean()

        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, meta, packed=False)
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        if step % 1000 == 0:
            print(f"[{step}] loss {loss.item():.4f} l1 {l1.item():.4f} ssim {ssim.item():.4f} "
                  f"N {params['means'].shape[0]}")

    save = {n: params[n].detach().cpu() for n in params.keys()}
    torch.save({"splats": save, "ut": False, "k1": 0.0, "eps2d": -1.0,
                "sh_degree": args.sh_degree, "twodgs": True}, os.path.join(args.out, "ckpt.pt"))
    print(f"saved {len(save['means'])} 2d-gaussians -> {args.out}/ckpt.pt")

    # render eval/test poses if requested (2dgs render path, RGB only)
    if args.csv and args.png_dir:
        import csv as _csv
        os.makedirs(args.png_dir, exist_ok=True)
        with torch.no_grad():
            with open(args.csv) as f:
                rows = list(_csv.DictReader(f))
            for r in rows:
                q = np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])])
                t = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])
                w = q / (np.linalg.norm(q) + 1e-12)
                wq, xq, yq, zq = w
                R = np.array([[1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - wq * zq), 2 * (xq * zq + wq * yq)],
                              [2 * (xq * yq + wq * zq), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - wq * xq)],
                              [2 * (xq * zq - wq * yq), 2 * (yq * zq + wq * xq), 1 - 2 * (xq * xq + yq * yq)]])
                w2c = np.eye(4, dtype=np.float32); w2c[:3, :3] = R; w2c[:3, 3] = t
                wm = torch.tensor(w2c, device=dev)[None]
                fx, fy = float(r["fx"]), float(r["fy"]); cx, cy = float(r["cx"]), float(r["cy"])
                Kc = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=dev, dtype=torch.float32)
                ww, hh = int(r["width"]), int(r["height"])
                colors = torch.cat([params["sh0"], params["shN"]], dim=1)
                rc = rasterization_2dgs(means=params["means"], quats=params["quats"],
                                        scales=torch.exp(params["scales"]),
                                        opacities=torch.sigmoid(params["opacities"]), colors=colors,
                                        viewmats=wm, Ks=Kc[None], width=ww, height=hh,
                                        sh_degree=args.sh_degree, near_plane=0.01, packed=False,
                                        render_mode="RGB")[0]
                arr = (rc[0][..., :3].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(args.png_dir, os.path.splitext(r["image_name"])[0] + ".png"))
        print(f"rendered {len(rows)} poses -> {args.png_dir}")


if __name__ == "__main__":
    main()
