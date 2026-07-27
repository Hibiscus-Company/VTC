#!/usr/bin/env python
"""Mip-Splatting 3D smoothing filter (Yu et al. 2023), applied POST-HOC to a trained ckpt.

Band-limits each Gaussian to the finest scale actually resolvable by the training cameras, so
sub-pixel Gaussians (which alias / produce high-frequency artifacts under novel views) are
low-pass filtered. Per Gaussian:
  tau  = filter_scale * min_over_cams( depth / focal_px )      # world-units-per-pixel, finest
  scale_i' = sqrt(scale_i^2 + tau^2)                            # inflate covariance isotropically
  opacity' = opacity * sqrt(prod scale_i^2 / prod scale_i'^2)  # preserve integrated energy
Only cameras that actually SEE the Gaussian (in front, inside the image frustum) contribute to
the min. World-space filter -> mode-agnostic (works for classic / antialiased / UT ckpts).

RULE 10: uses only the scene's own training cameras (poses+intrinsics from sparse/0). No GT
pixels, no external data. This is a rendering-correctness fix, not content injection.
"""
import argparse, os, struct
import numpy as np
import torch


def read_cam(cameras_bin):
    MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5)}
    with open(cameras_bin, "rb") as f:
        struct.unpack("<Q", f.read(8))
        cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
        model, npar = MODELS[mid]
        par = struct.unpack("<" + "d" * npar, f.read(8 * npar))
    fx = par[0]
    return float(fx), int(w), int(h)


def read_images(images_bin):
    """-> list of (qvec[4], tvec[3]) world->cam for every registered train frame."""
    out = []
    with open(images_bin, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<i", f.read(4))
            q = struct.unpack("<4d", f.read(32))
            t = struct.unpack("<3d", f.read(24))
            struct.unpack("<i", f.read(4))
            while f.read(1) != b"\x00":
                pass
            npt = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * npt, 1)
            out.append((np.array(q), np.array(t)))
    return out


def qt_to_w2c(q, t):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sparse", required=True, help="scene train/sparse/0 dir")
    ap.add_argument("--out", required=True, help="output ckpt path")
    ap.add_argument("--filter_scale", type=float, default=0.2,
                    help="Mip filter size in pixels (paper uses ~0.2)")
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sp = ck["splats"]
    means = sp["means"].to(dev).float()                       # [N,3]
    log_scales = sp["scales"].to(dev).float()                 # [N,3] (log)
    logit_op = sp["opacities"].to(dev).float()                # [N] or [N,1]
    scales = torch.exp(log_scales)                            # world-unit std devs
    N = means.shape[0]

    fx, W, H = read_cam(os.path.join(args.sparse, "cameras.bin"))
    frames = read_images(os.path.join(args.sparse, "images.bin"))
    w2cs = torch.tensor(np.stack([qt_to_w2c(q, t) for q, t in frames]),
                        device=dev, dtype=torch.float32)       # [C,4,4]

    # per-Gaussian finest world-units-per-pixel over cameras that SEE it
    ones = torch.ones(N, 1, device=dev)
    homog = torch.cat([means, ones], 1)                        # [N,4]
    best_wupp = torch.full((N,), float("inf"), device=dev)
    for c in range(w2cs.shape[0]):
        cam = (w2cs[c] @ homog.T).T                            # [N,4]
        z = cam[:, 2]
        x = cam[:, 0] / z.clamp(min=1e-6)
        y = cam[:, 1] / z.clamp(min=1e-6)
        u = fx * x + W / 2.0
        v = fx * y + H / 2.0
        vis = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        wupp = z / fx                                          # world units per pixel at depth z
        wupp = torch.where(vis, wupp, torch.full_like(wupp, float("inf")))
        best_wupp = torch.minimum(best_wupp, wupp)
    # gaussians seen by no camera: leave untouched
    seen = torch.isfinite(best_wupp)
    tau = torch.where(seen, args.filter_scale * best_wupp, torch.zeros_like(best_wupp))  # [N]

    tau2 = (tau ** 2).unsqueeze(1)                             # [N,1]
    new_scales2 = scales ** 2 + tau2                           # [N,3]
    new_scales = torch.sqrt(new_scales2)
    # opacity energy renorm: alpha * sqrt(prod s^2 / prod s'^2)
    ratio = torch.sqrt((scales ** 2).prod(1) / new_scales2.prod(1))   # [N]
    op = torch.sigmoid(logit_op.squeeze(-1) if logit_op.dim() > 1 else logit_op)
    new_op = (op * ratio).clamp(1e-6, 1 - 1e-6)
    new_logit = torch.log(new_op / (1 - new_op))

    sp["scales"] = torch.log(new_scales).cpu()
    sp["opacities"] = (new_logit.unsqueeze(-1) if logit_op.dim() > 1 else new_logit).cpu()
    ck["splats"] = sp
    ck["mip3d_filter_scale"] = args.filter_scale
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ck, args.out)
    nmod = int(seen.sum())
    print(f"mip3d: {nmod}/{N} gaussians filtered (tau med "
          f"{tau[seen].median().item():.5f} wu), scale +{(new_scales/scales).median().item():.4f}x med, "
          f"opacity x{ratio[seen].median().item():.4f} med -> {args.out}")


if __name__ == "__main__":
    main()
