#!/usr/bin/env python
"""DOES PHOTO-REUSE HAVE ANY HEADROOM ON THE *VIDEO* SCENES?

Diagnostic D3 killed image-based rendering with one number -- but it measured a TOWER, where the
nearest train pose is 11.8 deg of view angle away, and "paste the nearest train photo" scored
9.4 dB against our render's 24.5. That verdict was then applied to all 7 scenes.

The two video scenes are a different capture. Measured today:
    HCM0421 / HCM0674 towers   nearest train pose  8.5 / 9.9 deg,  16.5% of scene radius
    chair / bonsai             nearest train pose  3.6 / 4.0 deg,   7.2% of scene radius
2.4x closer in angle, 2.3x closer in baseline. The reasoning that killed IBR does not automatically
carry over, and the video scenes are 2/7 of the score.

THE PROBE, entirely on TRAIN data (Rule 10 clean, no test imagery anywhere):
hold out every 4th train frame; its nearest remaining train frame is then ~5 video frames away,
which is the same spacing a real test frame sees. For each held-out frame compare
    (a) paste the nearest remaining train PHOTO, unwarped
    (b) our production render of that frame
If (a) is anywhere near (b), a flow-corrected IBR member is worth building. If (a) is far below,
the video scenes die the same death as the towers and we stop thinking about it.
"""
import argparse, csv, os, re, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat

Image.MAX_IMAGE_PIXELS = None
DATA = "/mnt/d/avv/data/phase1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="chair")
    ap.add_argument("--render_dir", default="/mnt/d/avv/r2r9/models/chair_ut42/train_png")
    ap.add_argument("--hold", type=int, default=4, help="hold out every Nth train frame")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    root = os.path.join(DATA, "private_set2", args.scene)
    imgd = os.path.join(root, "train", "images")
    have = {os.path.splitext(f)[0]: f for f in os.listdir(imgd)}
    ex = read_extrinsics_binary(os.path.join(root, "train", "sparse", "0", "images.bin"))
    pose = {}
    for im in ex.values():
        s = os.path.splitext(im.name)[0]
        if s in have:
            R = qvec2rotmat(np.array(im.qvec))
            pose[s] = (-R.T @ np.array(im.tvec), R[2])
    stems = sorted(pose)
    held = [s for i, s in enumerate(stems) if i % args.hold == 0]
    pool = [s for s in stems if s not in set(held)]
    rendered = [s for s in held if os.path.exists(os.path.join(args.render_dir, s + ".png"))]
    rng = np.random.RandomState(0)
    ev = [rendered[i] for i in rng.permutation(len(rendered))[:args.n]]
    print(f"{args.scene}: {len(stems)} train poses, {len(held)} held out, "
          f"{len(pool)} in pool, {len(ev)} scored (need a production render)\n", flush=True)

    PC = np.stack([pose[s][0] for s in pool])
    PZ = np.stack([pose[s][1] for s in pool])

    def ld(p):
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

    def sc(img, g):
        r = torch.from_numpy(np.ascontiguousarray(np.clip(img, 0, 1))).permute(2, 0, 1
                                                                              ).unsqueeze(0).to(dev)
        with torch.no_grad():
            P = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
            S = float(repo_ssim(r, g)); L = float(vgg(r * 2 - 1, g * 2 - 1).item())
        return P, S, L

    acc = {"paste_nearest": [0.0, 0.0, 0.0], "our_render": [0.0, 0.0, 0.0]}
    dists, angs = [], []
    for c, s in enumerate(ev):
        cc, zz = pose[s]
        d = np.linalg.norm(PC - cc, axis=1)
        j = int(np.argmin(d))
        dists.append(float(d[j]))
        angs.append(float(np.degrees(np.arccos(np.clip(PZ[j] @ zz, -1, 1)))))
        gt = ld(os.path.join(imgd, have[s]))
        g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(dev)
        for k, im in (("paste_nearest", ld(os.path.join(imgd, have[pool[j]]))),
                      ("our_render", ld(os.path.join(args.render_dir, s + ".png")))):
            if im.shape != gt.shape:
                continue
            P, S, L = sc(im, g)
            acc[k][0] += P; acc[k][1] += S; acc[k][2] += L
        if c % 10 == 0:
            print(f"  {c}/{len(ev)}", flush=True)

    n = len(ev)
    print(f"\n{args.scene}: nearest pool frame is {np.mean(dists):.4f} away "
          f"({np.mean(angs):.2f} deg) -- real test frames see {np.mean(dists):.4f}-ish too")
    print(f"{'arm':>16} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8}")
    res = {}
    for k in acc:
        P, S, L = (x / n for x in acc[k])
        res[k] = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        print(f"{k:>16} {res[k]:9.4f} {P:8.4f} {S:7.4f} {L:8.4f}")
    print(f"\nVERDICT: paste-nearest is {res['paste_nearest']-res['our_render']:+.3f} vs our render."
          f"\n  D3's tower number for the same test was 9.4 dB vs 24.5 dB = hopeless.")


if __name__ == "__main__":
    main()
