#
# Camera/IO helpers shared by the gsplat pipeline (render_gsplat, render_train,
# fit_field tooling). Extracted from the legacy render_test_poses.py so the live
# path no longer imports the FastGS stack (GaussianModel / compiled rasterizer).
#
# test_poses.csv uses the COLMAP world-to-camera convention (qw,qx,qy,qz,tx,ty,tz),
# verified byte-identical to the entries in the scene's images.bin.
#

import csv
import os

import cv2
import numpy as np
from PIL import Image

from colmap_loader import read_intrinsics_binary


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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
    cams = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
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
