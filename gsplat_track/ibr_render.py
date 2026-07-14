#
# Depth-guided image-based rendering (audit round-8 rank-1 game changer).
#
# For each test pose: render splat RGB + expected depth in PADDED PINHOLE
# space from a Track-B ckpt, backproject every pixel to 3D, project into the
# 2 flight-adjacent train frames (closed-form SIMPLE_RADIAL forward
# distortion), and sample the ORIGINAL distorted train photos. Photo texture
# is accepted per-pixel behind three gates:
#   geometry  |z_reproj - neighbor_depth| / z < tau_geo   (occlusion)
#   photometry|lowpass(photo*gain) - lowpass(splat)| < tau_pho
#             (gate on LOW-PASSED images so the photo's high-frequency
#              detail -- the whole point -- is not rejected)
#   bounds    projected pixel inside the neighbor frame
# plus a per-neighbor scalar exposure gain fit on agreeing pixels. The splat
# render fills every rejected/disoccluded pixel; the accept mask is feathered
# to hide seams. Composite is warped into distorted GT geometry at the end
# (same DistortionWarp as every shipped render).
#
# Rule-10 clean: uses only this scene's train imagery + our own model.
#
import os
import re
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from render_test_poses import DistortionWarp, save_image, load_csv, read_k_from_sparse
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from PIL import Image


def seq_num(name):
    m = re.search(r"_(\d{4})_", name)
    return int(m.group(1)) if m else None


def gauss_blur(img_chw, sigma):
    # separable gaussian, reflect padding; img [C,H,W]
    r = max(1, int(3 * sigma))
    xs = torch.arange(-r, r + 1, device=img_chw.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    c = img_chw.shape[0]
    x = img_chw.unsqueeze(0)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"),
                 k.unsqueeze(2).expand(c, 1, 1, 2 * r + 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"),
                 k.unsqueeze(3).expand(c, 1, 2 * r + 1, 1), groups=c)
    return x.squeeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="train_gsplat.py ckpt.pt (UT or classic)")
    p.add_argument("--source", required=True, help="scene train/ dir (sparse/0 + images)")
    p.add_argument("--images", default="images", help="distorted originals to sample")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--png_dir", default=None)
    p.add_argument("--n_neighbors", type=int, default=2)
    p.add_argument("--tau_geo", type=float, default=0.02, help="relative depth agreement")
    p.add_argument("--tau_pho", type=float, default=0.06, help="low-pass residual gate")
    p.add_argument("--sigma_lp", type=float, default=4.0, help="low-pass blur sigma (px)")
    p.add_argument("--feather", type=float, default=2.0, help="accept-mask feather sigma")
    p.add_argument("--gain_clip", type=float, default=0.12, help="exposure gain clip +/-")
    p.add_argument("--splat_only", action="store_true", help="skip IBR (baseline sanity)")
    p.add_argument("--flow_correct", action="store_true",
                   help="snap each warped photo onto the splat render with cv2 DIS "
                        "optical flow before gating (fixes the measured stochastic "
                        "depth-noise misregistration; splat render = alignment "
                        "target, so no external data). Classical flow, Rule-10 clean")
    p.add_argument("--flow_max", type=float, default=3.0,
                   help="reject flow vectors longer than this (px): large flow = "
                        "flow failure or true disocclusion, not misregistration")
    args = p.parse_args()

    from gsplat import rasterization
    dev = "cuda"

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    splats = {n: t.to(dev) for n, t in ckpt["splats"].items()}
    sh_degree = ckpt["sh_degree"]
    ut = bool(ckpt.get("ut", False))
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    mode = "classic" if ut else "antialiased"
    print(f"{len(splats['means'])} gaussians, ut={ut}")

    # scene geometry
    k1 = read_k_from_sparse(os.path.join(args.source, "sparse", "0"))
    rows = load_csv(args.csv)
    r0 = rows[0]
    W, H = int(r0["width"]), int(r0["height"])
    fx, fy = float(r0["fx"]), float(r0["fy"])
    cx0, cy0 = float(r0.get("cx", W / 2.0)), float(r0.get("cy", H / 2.0))
    warp = DistortionWarp(W, H, fx, fy, k1)
    pad = warp.pad
    pw, ph = W + 2 * pad, H + 2 * pad
    Kpad = torch.tensor([[fx, 0, cx0 + pad], [0, fy, cy0 + pad], [0, 0, 1]],
                        dtype=torch.float32, device=dev)
    print(f"k1={k1:+.6f} pad={pad}px")

    # train frames: poses from images.bin, photos = distorted originals
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
        train.append({"name": im.name, "w2c": w2c, "path": path, "seq": seq_num(im.name)})
    train.sort(key=lambda v: v["name"])
    centers = np.stack([-v["w2c"][:3, :3].T @ v["w2c"][:3, 3] for v in train])
    have_seq = all(v["seq"] is not None for v in train)
    print(f"{len(train)} train frames (seq numbering: {have_seq})")

    photo_cache, depth_cache = {}, {}

    def render_splat(w2c_t, need_depth=True):
        rend, _, _ = rasterization(
            means=splats["means"], quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors, viewmats=w2c_t[None], Ks=Kpad[None],
            width=pw, height=ph, sh_degree=sh_degree, rasterize_mode=mode,
            near_plane=0.01, packed=False,
            with_ut=ut, with_eval3d=ut,
            render_mode="RGB+ED" if need_depth else "RGB")
        rgb = rend[0, ..., :3].clamp(0, 1)
        depth = rend[0, ..., 3] if need_depth else None
        return rgb, depth

    def neighbor_depth(idx):
        if idx not in depth_cache:
            w2c_t = torch.from_numpy(train[idx]["w2c"]).to(dev)
            _, d = render_splat(w2c_t, need_depth=True)
            depth_cache[idx] = d
            if len(depth_cache) > 40:
                depth_cache.pop(next(iter(depth_cache)))
        return depth_cache[idx]

    def neighbor_photo(idx):
        if idx not in photo_cache:
            arr = np.asarray(Image.open(train[idx]["path"]).convert("RGB"),
                             dtype=np.float32) / 255.0
            photo_cache[idx] = torch.from_numpy(arr).to(dev)
            if len(photo_cache) > 40:
                photo_cache.pop(next(iter(photo_cache)))
        return photo_cache[idx]

    # padded-pinhole pixel grid (undistorted camera coords)
    vs, us = torch.meshgrid(torch.arange(ph, device=dev, dtype=torch.float32),
                            torch.arange(pw, device=dev, dtype=torch.float32),
                            indexing="ij")
    dirs = torch.stack([(us - (cx0 + pad)) / fx, (vs - (cy0 + pad)) / fy,
                        torch.ones_like(us)], dim=-1)  # [ph,pw,3]

    os.makedirs(args.out, exist_ok=True)
    cov_all = []
    with torch.no_grad():
        for row in rows:
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = qvec2rotmat(np.array([float(row["qw"]), float(row["qx"]),
                                                float(row["qy"]), float(row["qz"])]))
            w2c[:3, 3] = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
            w2c_t = torch.from_numpy(w2c).to(dev)
            splat_rgb, depth = render_splat(w2c_t)

            if args.splat_only:
                arr = splat_rgb.cpu().numpy()
                save_image(warp.apply(arr), os.path.join(args.out, row["image_name"]),
                           jpeg_quality=100, jpeg_subsampling=0, png_dir=args.png_dir)
                continue

            # backproject to world
            R = w2c_t[:3, :3]
            t = w2c_t[:3, 3]
            Xc = dirs * depth.unsqueeze(-1)               # [ph,pw,3] camera
            Xw = (Xc - t) @ R                             # R^T (Xc - t)

            # neighbors by flight sequence (fallback: camera-center distance)
            s_test = seq_num(row["image_name"])
            if s_test is not None and have_seq:
                d_n = np.array([abs(v["seq"] - s_test) for v in train], dtype=np.float64)
            else:
                c_test = (-w2c[:3, :3].T @ w2c[:3, 3])
                d_n = np.linalg.norm(centers - c_test, axis=1)
            near_idx = np.argsort(d_n)[: args.n_neighbors]

            splat_lp = gauss_blur(splat_rgb.permute(2, 0, 1), args.sigma_lp)
            acc_photo = torch.zeros_like(splat_rgb)
            acc_w = torch.zeros_like(depth)
            for ni in near_idx:
                nb = train[ni]
                w2c_n = torch.from_numpy(nb["w2c"]).to(dev)
                Xn = Xw @ w2c_n[:3, :3].T + w2c_n[:3, 3]  # neighbor camera coords
                zn = Xn[..., 2].clamp(min=1e-6)
                xu, yu = Xn[..., 0] / zn, Xn[..., 1] / zn
                r2 = xu * xu + yu * yu
                dfac = 1.0 + k1 * r2                       # forward SIMPLE_RADIAL
                ud = fx * (xu * dfac) + cx0                # distorted photo px
                vd = fy * (yu * dfac) + cy0
                # geometry gate: neighbor splat depth at the PINHOLE projection
                up = fx * xu + (cx0 + pad)
                vp = fy * yu + (cy0 + pad)
                dn = neighbor_depth(int(ni))
                gx = (up / (pw - 1) * 2 - 1)
                gy = (vp / (ph - 1) * 2 - 1)
                grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
                dn_s = F.grid_sample(dn[None, None], grid, mode="bilinear",
                                     align_corners=True)[0, 0]
                ok = (Xn[..., 2] > 0.01) & (ud >= 0) & (ud <= W - 1) & \
                     (vd >= 0) & (vd <= H - 1) & \
                     ((zn - dn_s).abs() / zn < args.tau_geo)
                if ok.sum() < 1000:
                    continue
                photo = neighbor_photo(int(ni))
                pgx = (ud / (W - 1) * 2 - 1)
                pgy = (vd / (H - 1) * 2 - 1)
                pgrid = torch.stack([pgx, pgy], dim=-1).unsqueeze(0)
                warped = F.grid_sample(photo.permute(2, 0, 1)[None], pgrid,
                                       mode="bilinear", align_corners=True)[0] \
                    .permute(1, 2, 0)
                if args.flow_correct:
                    import cv2
                    a8 = (warped.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    b8 = (splat_rgb.cpu().numpy() * 255).astype(np.uint8)
                    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                    flow = dis.calc(cv2.cvtColor(a8, cv2.COLOR_RGB2GRAY),
                                    cv2.cvtColor(b8, cv2.COLOR_RGB2GRAY), None)
                    fmag = np.linalg.norm(flow, axis=-1)
                    # warp photo BY the flow: sample warped at p + flow(p)
                    fys, fxs = np.mgrid[0:ph, 0:pw].astype(np.float32)
                    snapped = cv2.remap(a8, fxs + flow[..., 0], fys + flow[..., 1],
                                        interpolation=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
                    warped = torch.from_numpy(snapped.astype(np.float32) / 255).to(dev)
                    # long flow = flow failure or true structural mismatch: reject
                    ok = ok & (torch.from_numpy(fmag).to(dev) < args.flow_max)
                # scalar exposure gain on geometry-accepted pixels
                w_lp = gauss_blur(warped.permute(2, 0, 1), args.sigma_lp)
                num = splat_lp.permute(1, 2, 0)[ok].mean()
                den = w_lp.permute(1, 2, 0)[ok].mean().clamp(min=1e-4)
                gain = (num / den).clamp(1 - args.gain_clip, 1 + args.gain_clip)
                warped = (warped * gain).clamp(0, 1)
                w_lp = w_lp * gain
                # photometric gate on LOW-PASSED difference
                resid = (w_lp - splat_lp).abs().mean(dim=0)
                ok = ok & (resid < args.tau_pho)
                wgt = ok.float() / (d_n[ni] + 0.5)
                acc_photo += warped * wgt.unsqueeze(-1)
                acc_w += wgt

            covered = acc_w > 0
            cov = covered.float().mean().item()
            cov_all.append(cov)
            photo_rgb = torch.where(covered.unsqueeze(-1),
                                    acc_photo / acc_w.clamp(min=1e-6).unsqueeze(-1),
                                    splat_rgb)
            # feathered blend hides seam between photo and splat regions
            m = gauss_blur(covered.float()[None], args.feather)[0].clamp(0, 1)
            out = photo_rgb * m.unsqueeze(-1) + splat_rgb * (1 - m).unsqueeze(-1)
            arr = out.clamp(0, 1).cpu().numpy()
            save_image(warp.apply(arr), os.path.join(args.out, row["image_name"]),
                       jpeg_quality=100, jpeg_subsampling=0, png_dir=args.png_dir)
    if cov_all:
        print(f"photo coverage: mean {np.mean(cov_all):.3f} min {np.min(cov_all):.3f} "
              f"max {np.max(cov_all):.3f}")
    print(f"Wrote {len(rows)} images to {args.out}")


if __name__ == "__main__":
    main()
