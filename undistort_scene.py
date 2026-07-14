#
# Undistort a scene's training images (COLMAP SIMPLE_RADIAL -> pinhole).
#
# Writes undistorted images to <scene>/train/images_undist next to the
# originals, keeping the same K (f, cx, cy). The sparse model is unchanged:
# poses and 3D points are distortion-free already; only the pixel observations
# change, so training can simply use -i images_undist and treat the camera as
# pinhole. SIMPLE_RADIAL matches OpenCV's k1-only Brown model exactly.
#
# Usage: python undistort_scene.py <scene_dir> [<scene_dir> ...]
#

import os
import sys
import glob
import importlib.util

import numpy as np
import cv2
from tqdm import tqdm

spec = importlib.util.spec_from_file_location(
    "cl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene/colmap_loader.py"))
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)


def undistort_dir(scene_dir):
    cams = cl.read_intrinsics_binary(os.path.join(scene_dir, "train/sparse/0/cameras.bin"))
    cam = list(cams.values())[0]
    assert cam.model == "SIMPLE_RADIAL", cam.model
    f, cx, cy, k = cam.params
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k, 0, 0, 0, 0], dtype=np.float64)
    W, H = cam.width, cam.height

    # map from undistorted pixel -> distorted source coords, same K
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (W, H), cv2.CV_32FC1)

    src_dir = os.path.join(scene_dir, "train/images")
    dst_dir = os.path.join(scene_dir, "train/images_undist")
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(os.listdir(src_dir))
    for fname in tqdm(files, desc=os.path.basename(scene_dir)):
        img = cv2.imread(os.path.join(src_dir, fname), cv2.IMREAD_COLOR)
        assert img.shape[:2] == (H, W), (fname, img.shape)
        out = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(os.path.join(dst_dir, fname), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"{scene_dir}: {len(files)} images -> {dst_dir} (k={k:+.5f})")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        undistort_dir(os.path.expanduser(d))
