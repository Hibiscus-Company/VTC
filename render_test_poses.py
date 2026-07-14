#
# Render images for competition test poses given in a test_poses.csv file.
#
# The CSV uses the COLMAP world-to-camera convention (qw,qx,qy,qz,tx,ty,tz),
# verified to be byte-identical to the entries in the scene's images.bin.
#
# Usage:
#   python render_test_poses.py -m output/<scene> --csv <scene>/test/test_poses.csv \
#       --out renders/<scene> --mult 0.7 [--iteration 30000] [--force_png]
#

import os
import csv
import math
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

from scene.gaussian_model import GaussianModel
from scene.colmap_loader import qvec2rotmat
from scene.cameras import MiniCam
from gaussian_renderer import render_fastgs
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from utils.system_utils import searchForMaxIteration


class PipeStub:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False
    separate_sh = True


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def ss_dims(row, ss):
    # supersampled render dims + exactly matching focal scale (identical FoV)
    w0, h0 = int(row["width"]), int(row["height"])
    hw, hh = int(round(w0 * ss)), int(round(h0 * ss))
    return hw, hh, float(row["fx"]) * hw / w0, float(row["fy"]) * hh / h0


def make_camera(row, pad=0, ss=1.0, znear=0.01, zfar=100.0):
    # ss = supersample factor (may be fractional): render at ss× the target
    # resolution (focal and base resolution scaled together → identical FoV,
    # more pixels). pad is in rendered (hr) pixels. The hr image is
    # downsampled back to target size afterwards for anti-aliasing.
    qvec = np.array([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])])
    tvec = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])
    hw, hh, fx, fy = ss_dims(row, ss)
    width = hw + 2 * pad
    height = hh + 2 * pad

    R = np.transpose(qvec2rotmat(qvec))
    T = tvec
    FoVx = focal2fov(fx, width)
    FoVy = focal2fov(fy, height)

    world_view = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return MiniCam(width, height, FoVy, FoVx, znear, zfar, world_view, full_proj)


class DistortionWarp:
    """Warp an (optionally padded) ideal pinhole render into the SIMPLE_RADIAL
    camera geometry of the ground-truth images (distorted = x_u * (1 + k r^2)).

    For each distorted output pixel we find its undistorted source position
    (cv2.undistortPoints inverts the radial polynomial) and bilinearly/bicubic
    sample the pinhole render there. pad enlarges the render canvas so
    out-of-frame source positions (k < 0) stay inside it.
    """

    def __init__(self, width, height, fx, fy, k):
        cx, cy = width / 2.0, height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k, 0, 0, 0, 0], dtype=np.float64)

        xs, ys = np.meshgrid(np.arange(width, dtype=np.float64),
                             np.arange(height, dtype=np.float64))
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, K, dist, P=None).reshape(-1, 2)
        map_x = (und[:, 0] * fx + cx).reshape(height, width)
        map_y = (und[:, 1] * fy + cy).reshape(height, width)

        self.pad = int(np.ceil(max(
            0.0,
            -map_x.min(), map_x.max() - (width - 1),
            -map_y.min(), map_y.max() - (height - 1),
        ))) + 4
        self.map_x = (map_x + self.pad).astype(np.float32)
        self.map_y = (map_y + self.pad).astype(np.float32)

    def apply(self, img):
        out = cv2.remap(img, self.map_x, self.map_y,
                        interpolation=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)
        return np.clip(out, 0.0, 1.0)


def read_k_from_sparse(sparse_dir):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene/colmap_loader.py"))
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    cams = cl.read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    cam = list(cams.values())[0]
    assert cam.model in ("SIMPLE_RADIAL", "RADIAL"), cam.model
    return float(cam.params[3])


def save_image(arr, path, quantize=False, jpeg_quality=100, jpeg_subsampling=0, png_dir=None):
    img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))
    if png_dir:
        # lossless archive copy: scoring/ensembling/re-encoding source, so the
        # final JPEG below stays single-generation at any future quality choice
        os.makedirs(png_dir, exist_ok=True)
        img.save(os.path.join(png_dir, os.path.splitext(os.path.basename(path))[0] + ".png"))
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        img.save(path, quality=jpeg_quality, subsampling=jpeg_subsampling, optimize=True)
    else:
        img.save(path)
        if quantize:
            # shrink truecolor PNG (~2.2MB) to ~700KB, matching the source-JPG
            # size, via pngquant (256-col palette + Floyd-Steinberg dither) then
            # oxipng lossless. ~37-38dB vs the lossless render → sub-0.002 score
            # impact, visually indistinguishable on this imagery.
            import subprocess
            subprocess.run(["pngquant", "--quality=70-95", "--speed", "1",
                            "--ext", ".png", "--force", path],
                           check=False, stderr=subprocess.DEVNULL)
            subprocess.run(["oxipng", "-o", "4", "--strip", "safe", "-q", path],
                           check=False, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default=None, help="output dir (default <model_path>/test_poses_renders)")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--mult", type=float, default=0.7, help="compact-box mult, must match training")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--force_png", action="store_true", help="save as <stem>.png regardless of csv extension")
    parser.add_argument("--png_quantize", action="store_true",
                        help="compress PNG output to ~700KB (pngquant+oxipng, near-lossless)")
    parser.add_argument("--distort", default="none",
                        help="'none': plain pinhole render; 'auto': read radial k from --sparse "
                             "and warp output into the distorted camera geometry; or a numeric k")
    parser.add_argument("--sparse", default=None, help="scene sparse/0 dir (for --distort auto)")
    parser.add_argument("--supersample", type=float, default=1.0,
                        help="render at Nx resolution then downsample (test-time anti-aliasing)")
    parser.add_argument("--ss_filter", choices=("box", "lanczos"), default="lanczos",
                        help="downsample filter for --supersample (box=cv2 INTER_AREA)")
    parser.add_argument("--jpeg_quality", type=int, default=100)
    parser.add_argument("--jpeg_subsampling", type=int, default=0, help="0=4:4:4, 2=4:2:0")
    parser.add_argument("--png_dir", default=None,
                        help="also write a lossless PNG copy of every render here")
    args = parser.parse_args()
    ss = args.supersample

    k = None
    if args.distort == "auto":
        assert args.sparse, "--distort auto requires --sparse"
        k = read_k_from_sparse(args.sparse)
        print(f"Distortion warp enabled, k={k:+.6f}")
    elif args.distort != "none":
        k = float(args.distort)
        print(f"Distortion warp enabled, k={k:+.6f}")

    iteration = args.iteration
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    out_dir = args.out or os.path.join(args.model_path, "test_poses_renders")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {ply_path}")
    gaussians = GaussianModel(args.sh_degree, optimizer_type="default")
    gaussians.load_ply(ply_path)
    print(f"{gaussians._xyz.shape[0]} gaussians")

    bg = torch.tensor([1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0], device="cuda")
    pipe = PipeStub()

    rows = load_csv(args.csv)

    warp = None
    if k is not None:
        r0 = rows[0]
        for r in rows:
            assert (r["fx"], r["fy"], r["width"], r["height"]) == \
                   (r0["fx"], r0["fy"], r0["width"], r0["height"]), "csv must share one camera"
        # warp operates at the (supersampled) render resolution
        hw, hh, hfx, hfy = ss_dims(r0, ss)
        warp = DistortionWarp(hw, hh, hfx, hfy, k)
        print(f"Render canvas padding: {warp.pad}px")
    if ss != 1.0:
        print(f"Supersampling {ss}x then {args.ss_filter}-downsampling")

    def downsample(arr, w0, h0):
        if arr.shape[0] == h0 and arr.shape[1] == w0:
            return arr
        interp = cv2.INTER_AREA if args.ss_filter == "box" else cv2.INTER_LANCZOS4
        return np.clip(cv2.resize(arr, (w0, h0), interpolation=interp), 0.0, 1.0)

    with torch.no_grad():
        for row in rows:
            cam = make_camera(row, pad=warp.pad if warp else 0, ss=ss)
            image = render_fastgs(cam, gaussians, pipe, bg, args.mult)["render"]
            arr = image.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
            if warp is not None:
                arr = warp.apply(arr)        # -> (ss*H, ss*W, 3)
            arr = downsample(arr, int(row["width"]), int(row["height"]))  # -> (H, W, 3)
            name = row["image_name"]
            if args.force_png:
                name = os.path.splitext(name)[0] + ".png"
            save_image(arr, os.path.join(out_dir, name), quantize=args.png_quantize,
                       jpeg_quality=args.jpeg_quality, jpeg_subsampling=args.jpeg_subsampling,
                       png_dir=args.png_dir)
    print(f"Wrote {len(rows)} images to {out_dir}")


if __name__ == "__main__":
    main()
