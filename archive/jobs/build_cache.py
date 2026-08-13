import os, sys, numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
GT_DIR = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
ROOT = "/mnt/d/avv/output"

variants = sorted([d for d in os.listdir(ROOT)
                   if d.startswith("HCM0181_")
                   and os.path.isdir(os.path.join(ROOT, d, "test_poses_renders_png"))
                   and len(os.listdir(os.path.join(ROOT, d, "test_poses_renders_png"))) == 60])
# also the prebuilt k4 ensemble
extra = [("k4", "/mnt/d/avv/prodharness/k4/png")]

gt_files = sorted(os.listdir(GT_DIR))
stems = [os.path.splitext(f)[0] for f in gt_files]
print(len(variants), "variants", len(stems), "images")

def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

H, W = load(os.path.join(GT_DIR, gt_files[0])).shape[:2]
print("HW", H, W)

names = [v.replace("HCM0181_", "") for v in variants] + [e[0] for e in extra]
dirs = [os.path.join(ROOT, v, "test_poses_renders_png") for v in variants] + [e[1] for e in extra]

R = np.lib.format.open_memmap(os.path.join(OUT, "renders_u8.npy"), mode="w+",
                              dtype=np.uint8, shape=(len(dirs), 60, H, W, 3))
G = np.lib.format.open_memmap(os.path.join(OUT, "gt_u8.npy"), mode="w+",
                              dtype=np.uint8, shape=(60, H, W, 3))

jobs = []
for vi, d in enumerate(dirs):
    fs = sorted(os.listdir(d))
    fstem = {os.path.splitext(f)[0]: f for f in fs}
    for ii, s in enumerate(stems):
        assert s in fstem, (d, s)
        jobs.append((vi, ii, os.path.join(d, fstem[s])))

with ProcessPoolExecutor(max_workers=12) as ex:
    for (vi, ii, p), arr in zip(jobs, ex.map(load, [j[2] for j in jobs], chunksize=8)):
        assert arr.shape == (H, W, 3), (p, arr.shape)
        R[vi, ii] = arr
for ii, f in enumerate(gt_files):
    G[ii] = load(os.path.join(GT_DIR, f))
R.flush(); G.flush()
with open(os.path.join(OUT, "names.txt"), "w") as f:
    f.write("\n".join(names))
print("done", names)
