#!/usr/bin/env python
"""VIEW-DEPTH of the fit pool. gsplatB11ut60k has 120 train renders on HCM0181 (the shipped
private fields are fit on 120 of 240 train photos). Compute the flow stack over ALL 120 so the
scorer can build fields from n=15/30/60/120 views and read the saturation curve. If 60->120 is
flat, rendering the remaining train views to deepen the fit is worthless.
"""
import os, sys, time
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
TAG = "HCM0181"
GT = f"/mnt/d/avv/data/phase1/public_set/{TAG}/train/images"
R11 = f"/mnt/d/avv/output/{TAG}_gsplatB11ut60k/train_renders"
OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/fieldrefit"
DSL = (8,)


def ld(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def gray8(x):
    return (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)


def main():
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
    stems = sorted(s for s in (os.path.splitext(f)[0] for f in os.listdir(R11)
                               if f.endswith(".png")) if s in gt_by)
    print(f"{len(stems)} matched train pairs", flush=True)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = {f"s{d}": [] for d in DSL}
    keep = []
    t0 = time.time()
    for n, s in enumerate(stems):
        g = ld(os.path.join(GT, gt_by[s]))
        b = ld(os.path.join(R11, s + ".png"))
        if b.shape != g.shape:
            continue
        H, W, _ = g.shape
        fl = np.clip(dis.calc(gray8(g), gray8(b), None), -6.0, 6.0)
        for d in DSL:
            out[f"s{d}"].append(cv2.resize(fl, (W // d, H // d),
                                           interpolation=cv2.INTER_AREA).astype(np.float16))
        keep.append(s)
        if n % 20 == 0:
            print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(OUT, f"flow_{TAG}_B11all.npz"),
                        stems=np.array(keep), HW=np.array([H, W]),
                        **{k: np.stack(v) for k, v in out.items()})
    print(f"wrote flow_{TAG}_B11all.npz  n={len(keep)}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
