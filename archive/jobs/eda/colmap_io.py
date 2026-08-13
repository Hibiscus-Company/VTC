"""Minimal COLMAP binary model reader (no pycolmap needed)."""
import struct, collections, numpy as np, os

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {c.model_id: c for c in CAMERA_MODELS}


def _read(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_cameras_binary(path):
    cams = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            props = _read(f, 24, "iiQQ")
            cam_id, model_id, width, height = props
            m = CAMERA_MODEL_IDS[model_id]
            params = _read(f, 8 * m.num_params, "d" * m.num_params)
            cams[cam_id] = dict(id=cam_id, model=m.model_name, width=width,
                                height=height, params=np.array(params))
    return cams


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            props = _read(f, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            cam_id = props[8]
            name = ""
            c = _read(f, 1, "c")[0]
            while c != b"\x00":
                name += c.decode("utf-8", "ignore")
                c = _read(f, 1, "c")[0]
            npts = _read(f, 8, "Q")[0]
            arr = _read(f, 24 * npts, "ddq" * npts)
            xys = np.array(arr).reshape(-1, 3)[:, :2] if npts else np.zeros((0, 2))
            p3d = np.array(arr).reshape(-1, 3)[:, 2].astype(np.int64) if npts else np.zeros(0, np.int64)
            images[image_id] = dict(id=image_id, qvec=qvec, tvec=tvec, camera_id=cam_id,
                                    name=name, xys=xys, point3D_ids=p3d)
    return images


def read_points3D_binary(path):
    ids, xyzs, rgbs, errs, tracklens = [], [], [], [], []
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            props = _read(f, 43, "QdddBBBd")
            ids.append(props[0])
            xyzs.append(props[1:4])
            rgbs.append(props[4:7])
            errs.append(props[7])
            tl = _read(f, 8, "Q")[0]
            f.read(8 * tl)
            tracklens.append(tl)
    return (np.array(ids, np.int64), np.array(xyzs), np.array(rgbs, np.uint8),
            np.array(errs), np.array(tracklens, np.int64))


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def cam_center(q, t):
    R = qvec2rotmat(q)
    return -R.T @ np.asarray(t)


def optical_axis(q):
    """world-space direction the camera looks (COLMAP: +z in cam frame)."""
    R = qvec2rotmat(q)
    return R.T @ np.array([0.0, 0.0, 1.0])
