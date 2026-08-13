"""Pose-geometry analysis: test vs train camera manifold."""
import os, sys, csv, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import (read_cameras_binary, read_images_binary, read_points3D_binary,
                       qvec2rotmat, cam_center, optical_axis)

SET2 = "/mnt/d/avv/data/phase1/private_set2"
SET1 = "/mnt/d/avv/data/phase1/private_set1"
PUB = "/mnt/d/avv/data/phase1/public_set"
EVAL = "/mnt/d/avv/evalsplit"


def load_poses_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                name=r["image_name"],
                q=np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]),
                t=np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])]),
                fx=float(r["fx"]), fy=float(r["fy"]), cx=float(r["cx"]), cy=float(r["cy"]),
                w=int(r["width"]), h=int(r["height"])))
    return rows


def find_sparse(root):
    for c in ["train/sparse/0", "train/sparse", "sparse/0", "sparse"]:
        p = os.path.join(root, c)
        if os.path.exists(os.path.join(p, "images.bin")):
            return p
    return None


def train_poses(root):
    sp = find_sparse(root)
    imgs = read_images_binary(os.path.join(sp, "images.bin"))
    out = []
    for k, v in imgs.items():
        out.append(dict(name=v["name"], q=v["qvec"], t=v["tvec"], id=v["id"],
                        camera_id=v["camera_id"],
                        n3d=int((v["point3D_ids"] >= 0).sum()), nkp=len(v["point3D_ids"])))
    out.sort(key=lambda d: d["name"])
    return out, sp


def centers(poses):
    return np.array([cam_center(p["q"], p["t"]) for p in poses])


def axes(poses):
    return np.array([optical_axis(p["q"]) for p in poses])


def nn_stats(Ctr, Cte, k=4):
    """distance from each test center to nearest / k-nearest train centers."""
    D = np.linalg.norm(Cte[:, None, :] - Ctr[None, :, :], axis=2)
    Ds = np.sort(D, axis=1)
    d1 = Ds[:, 0]
    dk = Ds[:, :k].mean(axis=1)
    return d1, dk, D


def train_spacing(Ctr):
    D = np.linalg.norm(Ctr[:, None, :] - Ctr[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, 0]  # nearest-neighbour spacing per train cam


def hull_test(Ctr, Cte):
    """Fraction of test cams inside convex hull of train cams (3D). Uses scipy Delaunay."""
    from scipy.spatial import Delaunay, ConvexHull
    try:
        tri = Delaunay(Ctr)
        inside = tri.find_simplex(Cte) >= 0
    except Exception:
        inside = np.zeros(len(Cte), bool)
    # also: signed distance outside hull (approx via max plane violation)
    try:
        hull = ConvexHull(Ctr)
        eqs = hull.equations  # [n,4] a.x + b <= 0 inside
        viol = (Cte @ eqs[:, :3].T + eqs[:, 3])  # >0 = outside
        out_dist = viol.max(axis=1)
    except Exception:
        out_dist = np.full(len(Cte), np.nan)
    return inside, out_dist


def angle_to_nn(Ctr, Atr, Cte, Ate, k=4):
    D = np.linalg.norm(Cte[:, None, :] - Ctr[None, :, :], axis=2)
    idx = np.argsort(D, axis=1)
    a1, ak = [], []
    for i in range(len(Cte)):
        d = Ate[i]
        cs = np.clip(Atr[idx[i, :k]] @ d, -1, 1)
        ang = np.degrees(np.arccos(cs))
        a1.append(ang[0]); ak.append(ang.max())
    return np.array(a1), np.array(ak)


def pct(a, q):
    return float(np.percentile(a, q))


def summ(a):
    a = np.asarray(a, float)
    return dict(median=float(np.median(a)), p90=pct(a, 90), max=float(a.max()),
                mean=float(a.mean()), p10=pct(a, 10), min=float(a.min()))
