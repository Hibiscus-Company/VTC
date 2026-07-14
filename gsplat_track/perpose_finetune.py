#
# Per-test-pose local finetune (audit round-8 rank-2 game changer).
#
# For each test pose: clone the scene model, finetune a few hundred steps on
# only the K flight-nearest train frames, render the test pose, discard the
# clone. Specializes appearance/exposure/geometry to the exact flight segment
# with no seams (everything stays in-model). Composes with IBR (better depth
# + better fallback).
#
# UT ckpts only for now (the adopted base): supervision renders are UT-native
# distorted vs the ORIGINAL distorted train photos -- byte-identical regime to
# --ut training -- and the test render is UT-native too, no warp.
# Rule-10 clean: only this scene's train imagery + our own model.
#
import os
import re
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from render_test_poses import save_image, load_csv
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from PIL import Image


def seq_num(name):
    m = re.search(r"_(\d{4})_", name)
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="UT ckpt.pt from train_gsplat.py --ut")
    p.add_argument("--source", required=True)
    p.add_argument("--images", default="images")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--png_dir", default=None)
    p.add_argument("--k", type=int, default=5, help="nearest train frames per test pose")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr_scale", type=float, default=0.3,
                   help="finetune lr = base trainer lr x this")
    p.add_argument("--lambda_lpips", type=float, default=0.1, help="0 disables")
    p.add_argument("--ssim_lambda", type=float, default=0.2)
    p.add_argument("--limit", type=int, default=0, help="only N poses (A/B smoke)")
    p.add_argument("--stride", type=int, default=1,
                   help="take every Nth CSV row before --limit (unbiased smoke "
                        "sample across the flight instead of one segment)")
    args = p.parse_args()

    from gsplat import rasterization
    from fused_ssim import fused_ssim
    torch.manual_seed(42)
    dev = "cuda"

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    assert ckpt.get("ut", False), "per-pose finetune currently supports UT ckpts only"
    base = {n: t.to(dev) for n, t in ckpt["splats"].items()}
    sh_degree = ckpt["sh_degree"]
    k1 = float(ckpt["k1"])
    radial = torch.tensor([[k1, 0, 0, 0, 0, 0]], dtype=torch.float32, device=dev)
    print(f"{len(base['means'])} gaussians, k1={k1:+.6f}")

    rows = load_csv(args.csv)
    r0 = rows[0]
    W, H = int(r0["width"]), int(r0["height"])
    fx, fy = float(r0["fx"]), float(r0["fy"])
    cx, cy = float(r0.get("cx", W / 2.0)), float(r0.get("cy", H / 2.0))
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device=dev)
    rows = rows[:: args.stride]
    if args.limit:
        rows = rows[: args.limit]

    imgs = read_extrinsics_binary(os.path.join(args.source, "sparse", "0", "images.bin"))
    img_dir = os.path.join(args.source, args.images)
    train = []
    for im in imgs.values():
        path = os.path.join(img_dir, im.name)
        if not os.path.exists(path):
            continue
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = qvec2rotmat(im.qvec)
        w2c[:3, 3] = im.tvec
        train.append({"name": im.name, "w2c": torch.from_numpy(w2c).to(dev),
                      "path": path, "seq": seq_num(im.name)})
    train.sort(key=lambda v: v["name"])
    centers = np.stack([(-v["w2c"][:3, :3].T @ v["w2c"][:3, 3]).cpu().numpy() for v in train])
    have_seq = all(v["seq"] is not None for v in train)
    print(f"{len(train)} train frames")

    lpips_net = None
    if args.lambda_lpips > 0:
        import lpips
        lpips_net = lpips.LPIPS(net="vgg").to(dev).eval()
        for q in lpips_net.parameters():
            q.requires_grad_(False)

    # base trainer lrs (scene_scale-independent parts); means lr uses the same
    # scene_scale recipe as train_gsplat.py
    scene_centers = centers
    scene_scale = 1.1 * float(np.max(np.linalg.norm(
        scene_centers - scene_centers.mean(0), axis=1)))
    lrs = {"means": 1.6e-4 * scene_scale * args.lr_scale,
           "scales": 5e-3 * args.lr_scale, "quats": 1e-3 * args.lr_scale,
           "opacities": 5e-2 * args.lr_scale,
           "sh0": 2.5e-3 * args.lr_scale, "shN": 2.5e-3 / 20 * args.lr_scale}

    gt_cache = {}

    def gt_of(idx):
        if idx not in gt_cache:
            arr = np.asarray(Image.open(train[idx]["path"]).convert("RGB"), dtype=np.uint8)
            gt_cache[idx] = torch.from_numpy(arr).to(dev).float().div_(255.0)
            if len(gt_cache) > 30:
                gt_cache.pop(next(iter(gt_cache)))
        return gt_cache[idx]

    def render(params, w2c_t):
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        rend, _, _ = rasterization(
            means=params["means"], quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors, viewmats=w2c_t[None], Ks=K[None], width=W, height=H,
            sh_degree=sh_degree, rasterize_mode="classic",
            near_plane=0.01, packed=False,
            with_ut=True, with_eval3d=True, radial_coeffs=radial)
        return rend[0]

    os.makedirs(args.out, exist_ok=True)
    for row in rows:
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = qvec2rotmat(np.array([float(row["qw"]), float(row["qx"]),
                                            float(row["qy"]), float(row["qz"])]))
        w2c[:3, 3] = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
        w2c_t = torch.from_numpy(w2c).to(dev)

        s_test = seq_num(row["image_name"])
        if s_test is not None and have_seq:
            d_n = np.array([abs(v["seq"] - s_test) for v in train], dtype=np.float64)
        else:
            c_test = -w2c[:3, :3].T @ w2c[:3, 3]
            d_n = np.linalg.norm(centers - c_test, axis=1)
        near_idx = [int(i) for i in np.argsort(d_n)[: args.k]]

        params = {n: t.clone().requires_grad_(True) for n, t in base.items()}
        opts = {n: torch.optim.Adam([params[n]], lr=lrs[n], eps=1e-15)
                for n in params}
        order = np.tile(near_idx, args.steps // len(near_idx) + 1)[: args.steps]
        np.random.shuffle(order)
        for step, vi in enumerate(order):
            img = render(params, train[vi]["w2c"])
            gt = gt_of(vi)
            l1 = (img - gt).abs().mean()
            ssim = fused_ssim(img.permute(2, 0, 1).unsqueeze(0),
                              gt.permute(2, 0, 1).unsqueeze(0))
            loss = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * (1 - ssim)
            if lpips_net is not None:
                loss = loss + args.lambda_lpips * lpips_net(
                    img.clamp(0, 1).permute(2, 0, 1).unsqueeze(0),
                    gt.permute(2, 0, 1).unsqueeze(0), normalize=True).mean()
            for o in opts.values():
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in opts.values():
                o.step()
        with torch.no_grad():
            out = render(params, w2c_t).clamp(0, 1).cpu().numpy()
        save_image(out, os.path.join(args.out, row["image_name"]),
                   jpeg_quality=100, jpeg_subsampling=0, png_dir=args.png_dir)
        del params, opts
        torch.cuda.empty_cache()
        print(f"{row['image_name']}: k-neighbors {[train[i]['name'][-12:-6] for i in near_idx]} done")
    print(f"Wrote {len(rows)} images to {args.out}")


if __name__ == "__main__":
    main()
