import os, sys, numpy as np
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                 read_points3D_binary, qvec2rotmat)
from PIL import Image

SRC = "/mnt/d/avv/data/phase1/private_set2/bonsai/train"
sp = os.path.join(SRC, "sparse", "0")
imgs = read_extrinsics_binary(os.path.join(sp, "images.bin"))
cams = read_intrinsics_binary(os.path.join(sp, "cameras.bin"))
xyz, rgb, err = read_points3D_binary(os.path.join(sp, "points3D.bin"))
print("N points3D:", xyz.shape[0])
cam = list(cams.values())[0]
print("camera:", cam.model, cam.width, cam.height, cam.params)

names = sorted(os.listdir(os.path.join(SRC, "images")))
print("train imgs:", len(names), names[:2])

RD = "/mnt/d/avv/bonsai_eval/K2_clip_full/eval_render"
ev = sorted(os.listdir(RD))
evstem = set(os.path.splitext(f)[0] for f in ev)
print("eval frames:", len(evstem))

byname = {im.name: im for im in imgs.values()}
print("images.bin entries:", len(byname))

W, H = cam.width, cam.height
if cam.model in ("PINHOLE",):
    fx, fy, cx, cy = cam.params
elif cam.model in ("SIMPLE_PINHOLE",):
    f, cx, cy = cam.params; fx = fy = f
else:
    fx = cam.params[0]; fy = cam.params[0]; cx = cam.params[1]; cy = cam.params[2]

GX, GY = 12, 8   # grid cells
dens = np.zeros((GY, GX)); errm = np.zeros((GY, GX)); cnt = 0
depth_cells = np.zeros((GY, GX)); depth_n = np.zeros((GY, GX))
per_img = []
for stem in sorted(evstem):
    cand = [stem + e for e in (".jpg", ".JPG", ".png", ".PNG")]
    im = None
    for c in cand:
        if c in byname: im = byname[c]; break
    if im is None:
        print("MISS", stem); continue
    R = qvec2rotmat(im.qvec); t = im.tvec
    P = (xyz @ R.T) + t
    z = P[:, 2]
    m = z > 1e-6
    u = fx * P[m, 0] / z[m] + cx
    v = fy * P[m, 1] / z[m] + cy
    zz = z[m]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[inb]; v = v[inb]; zz = zz[inb]
    gi = (v / H * GY).astype(int).clip(0, GY - 1)
    gj = (u / W * GX).astype(int).clip(0, GX - 1)
    d = np.zeros((GY, GX))
    np.add.at(d, (gi, gj), 1.0)
    dens += d
    np.add.at(depth_cells, (gi, gj), zz)
    np.add.at(depth_n, (gi, gj), 1.0)
    # error
    gtp = os.path.join(SRC, "images", im.name)
    rp = os.path.join(RD, stem + ".jpg")
    if not os.path.exists(rp): rp = os.path.join(RD, stem + ".png")
    gt = np.asarray(Image.open(gtp).convert("RGB"), dtype=np.float32)
    rr = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32)
    if gt.shape != rr.shape:
        print("shape mismatch", gt.shape, rr.shape); continue
    e = np.abs(gt - rr).mean(2)
    eh = e.reshape(GY, H // GY, GX, W // GX).mean(axis=(1, 3)) if (H % GY == 0 and W % GX == 0) else None
    if eh is None:
        ys = np.linspace(0, H, GY + 1).astype(int); xs = np.linspace(0, W, GX + 1).astype(int)
        eh = np.array([[e[ys[a]:ys[a+1], xs[b]:xs[b+1]].mean() for b in range(GX)] for a in range(GY)])
    errm += eh
    per_img.append((stem, d.sum(), e.mean()))
    cnt += 1

dens /= cnt; errm /= cnt
np.set_printoptions(precision=1, suppress=True, linewidth=200)
print("\n=== SfM points projected per cell (mean over %d eval views) ===" % cnt)
print(dens)
print("\n=== mean |err| (0-255) per cell ===")
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print(errm)
print("\n=== mean point depth per cell ===")
print(np.where(depth_n > 0, depth_cells / np.maximum(depth_n, 1), np.nan))
d = dens.ravel(); e = errm.ravel()
print("\ncorr(density, err) =", np.corrcoef(d, e)[0, 1])
print("corr(log1p density, err) =", np.corrcoef(np.log1p(d), e)[0, 1])
print("\nper-image: mean pts %.0f  mean err %.2f" % (np.mean([p[1] for p in per_img]), np.mean([p[2] for p in per_img])))
pi = np.array([[p[1], p[2]] for p in per_img])
print("corr across images(pts, err) =", np.corrcoef(pi[:, 0], pi[:, 1])[0, 1])
