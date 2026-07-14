#
# Sanity-check visualization: trained gaussian point cloud vs COLMAP sparse
# input, in three orthographic projections, with train/test camera positions.
#
# Usage: python viz_pointcloud.py <scene_data_dir> <model_dir> <out.png>
#

import os
import sys
import csv
import importlib.util

import numpy as np
from plyfile import PlyData
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location(
    "cl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene/colmap_loader.py"))
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)

SH_C0 = 0.28209479177387814


def load_gaussians(ply_path, max_pts=250_000):
    v = PlyData.read(ply_path).elements[0]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    rgb = np.clip(0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1), 0, 1)
    opa = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))
    if len(xyz) > max_pts:
        sel = np.random.default_rng(0).choice(len(xyz), max_pts, replace=False)
        xyz, rgb, opa = xyz[sel], rgb[sel], opa[sel]
    return xyz, rgb, opa


def load_sparse(ply_path, max_pts=250_000):
    v = PlyData.read(ply_path).elements[0]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1) / 255.0
    if len(xyz) > max_pts:
        sel = np.random.default_rng(0).choice(len(xyz), max_pts, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return xyz, rgb


def cam_center(qvec, tvec):
    R = cl.qvec2rotmat(qvec)
    return -R.T @ tvec


def load_cameras(scene_dir):
    imgs = cl.read_extrinsics_binary(os.path.join(scene_dir, "train/sparse/0/images.bin"))
    train_files = set(os.listdir(os.path.join(scene_dir, "train/images")))
    train_c = np.array([cam_center(e.qvec, np.array(e.tvec))
                        for e in imgs.values() if e.name in train_files])
    with open(os.path.join(scene_dir, "test/test_poses.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    test_c = np.array([cam_center(
        np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]),
        np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])) for r in rows])
    return train_c, test_c


def main():
    scene_dir, model_dir, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
    name = os.path.basename(os.path.normpath(scene_dir))

    g_xyz, g_rgb, g_opa = load_gaussians(
        os.path.join(model_dir, "point_cloud/iteration_30000/point_cloud.ply"))
    s_xyz, s_rgb = load_sparse(os.path.join(scene_dir, "train/sparse/0/points3D.ply"))
    train_c, test_c = load_cameras(scene_dir)

    # shared axis limits from the sparse cloud + cameras (robust to gaussian floaters)
    ref = np.concatenate([s_xyz, train_c, test_c], axis=0)
    lo, hi = np.percentile(ref, 0.5, axis=0), np.percentile(ref, 99.5, axis=0)
    pad = 0.15 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    pairs = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    n_gauss_total = PlyData.read(os.path.join(
        model_dir, "point_cloud/iteration_30000/point_cloud.ply")).elements[0].count

    for col, (a, b, la, lb) in enumerate(pairs):
        ax = axes[0][col]
        ax.scatter(g_xyz[:, a], g_xyz[:, b], s=0.05, c=g_rgb, alpha=0.5, linewidths=0)
        ax.scatter(train_c[:, a], train_c[:, b], s=7, c="blue", marker="o", label="train cam")
        ax.scatter(test_c[:, a], test_c[:, b], s=16, c="red", marker="^", label="test cam")
        ax.set_xlim(lo[a], hi[a]); ax.set_ylim(lo[b], hi[b])
        ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_aspect("equal")
        ax.set_title(f"gaussians {la}{lb}")
        if col == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[1][col]
        ax.scatter(s_xyz[:, a], s_xyz[:, b], s=0.05, c=s_rgb, alpha=0.6, linewidths=0)
        ax.scatter(train_c[:, a], train_c[:, b], s=7, c="blue", marker="o")
        ax.scatter(test_c[:, a], test_c[:, b], s=16, c="red", marker="^")
        ax.set_xlim(lo[a], hi[a]); ax.set_ylim(lo[b], hi[b])
        ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_aspect("equal")
        ax.set_title(f"COLMAP sparse {la}{lb}")

    inside = np.all((g_xyz >= lo) & (g_xyz <= hi), axis=1).mean() * 100
    fig.suptitle(
        f"{name} — gaussians: {n_gauss_total:,} (top, sampled), "
        f"sparse: {len(s_xyz):,} pts (bottom) | "
        f"{inside:.1f}% gaussians inside plot box | "
        f"cams: {len(train_c)} train / {len(test_c)} test",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=110)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
