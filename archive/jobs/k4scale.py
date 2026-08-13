"""Scale sweep on the PRODUCTION object: the 4-member pixel-mean ensemble of HCM0181,
scored as PNG and after the shipped q100/ss2 JPEG encode.

The field is the one production would use: fit on the B9ut member's TRAIN renders vs the
scene's TRAIN photos.  Only the scale multiplier changes.
"""
import os, sys, json, time
import numpy as np
import cv2
import torch
import fieldlib as F

cv2.setNumThreads(8)
torch.set_num_threads(8)
RES = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/res"
SCALES = [0.0, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8]
REN = "/mnt/d/avv/prodharness/k4/png"
GT = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"

sc = F.Scorer("cuda")
stack, _ = F.load_stack("HCM0181", 8)
med = F.pool(stack, "median")
gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
files = sorted(f for f in os.listdir(REN) if f.lower().endswith(".png"))
files = [f for f in files if os.path.splitext(f)[0] in gt_by]
acc = {(a, j): np.zeros(3) for a in SCALES for j in (0, 1)}
maps = {}
t0 = time.time()
for i, f in enumerate(files):
    r = F.load_u8(os.path.join(REN, f))
    g = F.load_u8(os.path.join(GT, gt_by[os.path.splitext(f)[0]]))
    H, W, _ = r.shape
    gt_t = F.to_t(g, sc.dev)
    for a in SCALES:
        if a == 0.0:
            y = r
        else:
            if a not in maps:
                maps[a] = F.make_maps(med * a, H, W)
            y = F.warp_u8(r, *maps[a], cv2.INTER_LANCZOS4)
        acc[(a, 0)] += np.array(sc(y, gt_t))
        acc[(a, 1)] += np.array(sc(F.jpeg_rt(y, 100, 2), gt_t))
    if i % 20 == 0:
        print(f"  {i}/{len(files)} {time.time()-t0:.0f}s", flush=True)
n = len(files)
out = {}
print(f"\n== HCM0181 k4 ENSEMBLE, real test GT (n={n}) ==")
for j, tag in ((0, "PNG "), (1, "JPEG")):
    base = F.score(*(acc[(0.0, j)] / n))
    for a in SCALES:
        P, S, L = acc[(a, j)] / n
        s = F.score(P, S, L)
        out[f"{tag.strip()}_{a}"] = dict(psnr=P, ssim=S, lpips=L, score=s, n=n)
        print(f"  {tag} scale {a:4.2f}  P {P:7.4f}  S {S:.5f}  L {L:.5f}  score {s:8.4f}"
              f"  d {s-base:+.4f}")
json.dump(out, open(f"{RES}/k4scale.json", "w"), indent=1)
