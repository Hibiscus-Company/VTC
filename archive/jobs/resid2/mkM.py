#!/usr/bin/env python
"""Cache the TEST-POSE view-consistent displacement oracle M (diagnostic only, never shipped).

M = median over test views of DIS flow(test photo -> production ensemble render), pooled at 1/8.
Also caches the per-view stack so split-half / hold-out statistics can be computed later without
recomputing any flow.  Rule 10: nothing fitted from this file reaches a submission; it exists to
answer "is there view-consistent structure left that a train-only field could chase".
"""
import os, sys, time
import numpy as np
import cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
OUT = os.path.join(HERE, "resid2")
Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(int(os.environ.get("NT", "8")))
DS = 8
POOL = {"HCM0181": ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]}


def ld(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def gray8(x):
    return (cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    tag = os.environ.get("TAG", "HCM0181")
    mems = POOL.get(tag, ["gsplatB9ut"])
    dirs = [f"/mnt/d/avv/output/{tag}_{m}/test_poses_renders_png" for m in mems]
    gtd = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    st, t0 = [], time.time()
    for i, s in enumerate(stems):
        r = np.mean([ld(os.path.join(d, s + ".png")) for d in dirs], 0)
        g = ld(os.path.join(gtd, gt_by[s]))
        H, W, _ = r.shape
        fl = np.clip(dis.calc(gray8(g), gray8(r), None), -6.0, 6.0)
        st.append(cv2.resize(fl, (W // DS, H // DS), interpolation=cv2.INTER_AREA))
        if i % 15 == 0:
            print(f"  {tag} {i}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    T = np.stack(st).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, f"M_{tag}.npz"), T=T.astype(np.float16),
                        stems=np.array(stems), HW=np.array([H, W]))
    print(f"{tag}: T {T.shape}  |M| {np.linalg.norm(np.median(T,0),axis=2).mean():.4f} px  "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
