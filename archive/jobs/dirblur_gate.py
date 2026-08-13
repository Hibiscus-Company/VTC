#!/usr/bin/env python
"""CHEAP GATE for the motion-blur forward model -- costs minutes, decides an 8-12 GPU-h build.

Our two prior blur tests do NOT rule this out:
  - 24/07 Farneback gate tested CORRELATION (does motion explain error *differences*). It found
    ~15px inter-frame motion in BOTH best- and worst-fit frames -- motion is uniformly LARGE, so a
    correlation test is structurally blind to a uniformly-present blur. Null there != no blur.
  - M4 sweep tested ISOTROPIC gaussian blur (net ~0). Motion blur is ANISOTROPIC and per-frame.

This tests the actual hypothesis: blur each render along ITS OWN camera-motion direction, with
magnitude from the pose finite difference, and see if the score moves. Rule 10 clean: uses only
the GIVEN poses (finite differences of pose sequence), never test GT, never external imagery.

If no alpha > 0 beats alpha = 0, the blur forward model is dead for this scene and we skip the build.
"""
import argparse, os, sys, csv
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None


def qvec2R(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_poses(csv_path):
    out = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            q = np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])])
            t = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])
            R = qvec2R(q)
            out[os.path.splitext(r["image_name"])[0]] = {
                "R": R, "t": t, "C": -R.T @ t, "fx": float(r["fx"]), "fy": float(r["fy"])}
    return out


def line_kernel(angle_rad, length):
    """normalized 1D line (motion) kernel rendered into a 2D patch"""
    L = max(int(round(length)), 1)
    if L <= 1:
        return None
    size = L if L % 2 == 1 else L + 1
    k = np.zeros((size, size), np.float32)
    c = size // 2
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    for i in range(L):
        s = (i - (L - 1) / 2.0)
        x = int(round(c + s * dx)); y = int(round(c + s * dy))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] += 1.0
    return k / k.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--eval_csv", required=True, help="poses of the rendered (held-out) frames")
    ap.add_argument("--all_csv", nargs="+", required=True,
                    help="pose csv(s) covering the FULL temporal sequence (train+held-out)")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.15, 0.3, 0.5, 0.75, 1.0])
    ap.add_argument("--tag", default="dirblur")
    args = ap.parse_args()

    import cv2
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    poses = {}
    for c in args.all_csv:
        poses.update(read_poses(c))
    ev = read_poses(args.eval_csv)
    order = sorted(poses)                      # frame_NNNNNN -> temporal order
    pos = {s: i for i, s in enumerate(order)}
    print(f"{len(order)} poses in full sequence, {len(ev)} held-out frames scored")

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.render_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir)}

    # per-frame image-space motion from pose finite differences (Rule 10: poses only)
    motion = {}
    for s in stems:
        i = pos[s]
        a, b = order[max(i - 1, 0)], order[min(i + 1, len(order) - 1)]
        if a == b:
            motion[s] = (0.0, 0.0); continue
        pa, pb, pe = poses[a], poses[b], ev[s]
        # camera translation between neighbours, expressed in the eval frame's camera axes,
        # projected to pixels at a representative scene depth of 1 unit
        d = pe["R"] @ (pb["C"] - pa["C"])
        du, dv = d[0] * pe["fx"], d[1] * pe["fy"]
        # add the rotational component (dominant for handheld pan): image shift ~ f * dtheta
        rel = pb["R"] @ pa["R"].T
        rvec, _ = cv2.Rodrigues(rel)
        du += -rvec[1, 0] * pe["fx"]
        dv += rvec[0, 0] * pe["fy"]
        motion[s] = (float(du), float(dv))
    mags = np.array([np.hypot(*motion[s]) for s in stems])
    print(f"per-frame motion magnitude (px @ depth 1): median {np.median(mags):.2f} "
          f"p10 {np.percentile(mags,10):.2f} p90 {np.percentile(mags,90):.2f}")

    R = {s: np.asarray(Image.open(os.path.join(args.render_dir, s + ".png")).convert("RGB"),
                       dtype=np.float32) / 255.0 for s in stems}
    G = {s: np.asarray(Image.open(os.path.join(args.gt_dir, gt_by[s])).convert("RGB"),
                       dtype=np.float32) / 255.0 for s in stems}

    for a in args.alphas:
        P = S = L = 0.0
        for s in stems:
            img = R[s]
            du, dv = motion[s]
            mag = np.hypot(du, dv) * a
            if mag >= 1.5:
                k = line_kernel(np.arctan2(dv, du), mag)
                if k is not None:
                    img = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)
            r = torch.from_numpy(np.clip(img, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(dev)
            g = torch.from_numpy(G[s]).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                S += float(repo_ssim(r, g))
                L += float(vgg(r * 2 - 1, g * 2 - 1).item())
        n = len(stems)
        P, S, L = P / n, S / n, L / n
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        print(f"{args.tag} alpha {a:5.2f}: PSNR {P:7.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {sc:.4f}")


if __name__ == "__main__":
    main()
