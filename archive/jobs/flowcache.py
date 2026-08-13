"""Cache the DIS train-render->train-photo flow stacks at several downsample factors.

One pass over each public scene's TRAIN pairs (60 renders vs the scene's own train photos --
never test GT). Everything downstream (ds sweep, parametric fits, pooling) reuses these.
"""
import os, sys, time
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(8)

DSS = [2, 4, 8, 16, 32]
OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/flow"
os.makedirs(OUT, exist_ok=True)


def scene_pairs(render_dir, gt_dir):
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)}
    files = sorted(f for f in os.listdir(render_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    return [(os.path.join(render_dir, f), os.path.join(gt_dir, gt_by[os.path.splitext(f)[0]]),
             os.path.splitext(f)[0])
            for f in files if os.path.splitext(f)[0] in gt_by]


def run(name, render_dir, gt_dir, clip=6.0):
    dst = os.path.join(OUT, name + ".npz")
    if os.path.exists(dst):
        print(f"{name}: cached"); return
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    pairs = scene_pairs(render_dir, gt_dir)
    stacks = {d: [] for d in DSS}
    stems = []
    t0 = time.time()
    for rp, gp, stem in pairs:
        r = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0
        if r.shape != g.shape:
            continue
        H, W, _ = r.shape
        rg = (cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        gg = (cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        fl = np.clip(dis.calc(gg, rg, None), -clip, clip)
        for d in DSS:
            stacks[d].append(cv2.resize(fl, (W // d, H // d), interpolation=cv2.INTER_AREA))
        stems.append(stem)
    np.savez_compressed(dst, stems=np.array(stems),
                        **{f"ds{d}": np.stack(stacks[d]).astype(np.float32) for d in DSS})
    print(f"{name}: {len(stems)} pairs in {time.time()-t0:.1f}s -> {dst}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "public"
    if which == "public":
        for s in ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]:
            run(s, f"/mnt/d/avv/output/{s}_gsplatB9ut/train_renders",
                f"/mnt/d/avv/data/phase1/public_set/{s}/train/images")
    else:
        for s in ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674"]:
            run("P_" + s, f"/mnt/d/avv/r2r9/models/{s}_ut42/train_png",
                f"/mnt/d/avv/data/phase1/private_set2/{s}/train/images")
