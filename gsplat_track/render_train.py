#!/usr/bin/env python
"""Render a trained model at its own TRAIN poses.

render_gsplat.py only renders from a test-pose CSV. To calibrate the systematic
render->photo displacement field we must measure it on TRAIN views, because the
private set gives us no test GT -- train photos are the only ground truth we are
ever allowed to fit against there.

Uses the trainer's own load_scene() and the same 3DGUT render path, so a train
render is pixel-pipeline-identical to a test render.
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gsplat import load_scene
from render_test_poses import DistortionWarp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--source", required=True, help="scene train/ dir (has sparse/0)")
    p.add_argument("--images", default="images")
    p.add_argument("--out", required=True, help="dir for rendered PNGs")
    p.add_argument("--limit", type=int, default=0, help="0 = all train views")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--ut_render", choices=("native", "warp"), default="native",
                   help="MUST match the path the scene's TEST renders ship through. "
                        "HNI0131/HNI0265 (k1=-0.115) ship via warp: fitting the "
                        "displacement field on one pixel pipeline and applying it to "
                        "another mis-targets it at exactly the sub-pixel scale it "
                        "operates on.")
    args = p.parse_args()

    from gsplat import rasterization

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    splats = {n: t.to(dev) for n, t in ckpt["splats"].items()}
    sh_degree = ckpt["sh_degree"]
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

    ut = bool(ckpt.get("ut", False))
    mode = "classic" if ut else "antialiased"
    views, K_np, (W, H), _, _, k1 = load_scene(args.source, args.images)
    k1 = float(ckpt.get("k1", k1))
    ut_native = ut and args.ut_render == "native"

    warp, pad, radial_coeffs = None, 0, None
    if ut_native:
        radial_coeffs = torch.tensor([[k1, 0., 0., 0., 0., 0.]],
                                     dtype=torch.float32, device=dev)
    elif ut:
        # mirror render_gsplat.py's warp branch exactly: pinhole render on a padded
        # canvas, then DistortionWarp. A UT-trained ckpt keeps with_ut/with_eval3d
        # even here (pinhole UT is valid and fold-free) -- dropping eval3d would
        # change the gaussian response model between train and render.
        warp = DistortionWarp(W, H, float(K_np[0, 0]), float(K_np[1, 1]), k1)
        pad = warp.pad

    K_np = K_np.copy()
    K_np[0, 2] += pad
    K_np[1, 2] += pad
    K = torch.tensor(K_np, dtype=torch.float32, device=dev)
    print(f"{len(splats['means'])} gaussians, ut={ut}, k1={k1:+.6f}, {W}x{H}, "
          f"path={'native' if ut_native else ('warp pad=%d' % pad)}")

    views = views[:: args.stride]
    if args.limit:
        views = views[: args.limit]

    for i, v in enumerate(views):
        w2c = torch.tensor(v["w2c"], dtype=torch.float32, device=dev)[None]
        with torch.no_grad():
            render, _, _ = rasterization(
                means=splats["means"], quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors, viewmats=w2c, Ks=K[None],
                width=W + 2 * pad, height=H + 2 * pad,
                sh_degree=sh_degree, rasterize_mode=mode, near_plane=0.01,
                packed=False, radial_coeffs=radial_coeffs,
                with_ut=ut, with_eval3d=ut)
        arr = render[0].clamp(0, 1).cpu().numpy()
        if warp is not None:
            arr = warp.apply(arr)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
        stem = os.path.splitext(v["name"])[0]
        Image.fromarray(arr).save(os.path.join(args.out, stem + ".png"))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(views)}", flush=True)
    print(f"rendered {len(views)} train views -> {args.out}")


if __name__ == "__main__":
    main()
