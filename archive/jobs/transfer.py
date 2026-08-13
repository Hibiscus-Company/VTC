#!/usr/bin/env python
"""Does per-scene TRAIN fit predict per-scene TEST fit? (the transfer question)

The capacity pivot assumed higher train fit -> higher test. But per-scene train PSNR varies
26-30 EVEN AMONG the 240-image scenes (same capacity), so scene DIFFICULTY, not view count,
drives train fit. The question that actually matters: when a scene fits train higher, does it
also test higher? If yes, pushing train fit is worth it. If the train-test GAP is constant,
higher train fit just means the scene is easier and there's no lever.

5 public scenes, real test GT. All from the same gsplatB9ut recipe, so recipe is held fixed
and the only variable is the scene. Correlate train PSNR vs test PSNR across scenes.
"""
import os, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

PUB = os.path.expanduser("~/data/phase1/public_set")
SCENES = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]


def psnr_dir(rd, gt):
    st = {os.path.splitext(f)[0]: f for f in os.listdir(gt)}
    v = []
    for f in sorted(os.listdir(rd)):
        s = os.path.splitext(f)[0]
        if s not in st:
            continue
        a = np.asarray(Image.open(os.path.join(rd, f)).convert("RGB"), dtype=np.float32) / 255
        b = np.asarray(Image.open(os.path.join(gt, st[s])).convert("RGB"), dtype=np.float32) / 255
        if a.shape != b.shape:
            continue
        v.append(10 * np.log10(1 / max(float(((a - b) ** 2).mean()), 1e-12)))
    return float(np.mean(v)) if v else float("nan")


print(f"{'scene':9s} {'TRAIN':>7s} {'TEST':>7s} {'gap':>6s}")
tr, te = [], []
for s in SCENES:
    d = f"/mnt/d/avv/output/{s}_gsplatB9ut"
    t = psnr_dir(f"{d}/train_renders", f"{PUB}/{s}/train/images")
    e = psnr_dir(f"{d}/test_poses_renders_png", f"{PUB}/{s}/test/images")
    tr.append(t); te.append(e)
    print(f"{s:9s} {t:7.3f} {e:7.3f} {t-e:6.3f}")

tr, te = np.array(tr), np.array(te)
r = np.corrcoef(tr, te)[0, 1]
slope = np.polyfit(tr, te, 1)[0]
print(f"\ntrain-test correlation across scenes: r = {r:+.3f}")
print(f"slope d(test)/d(train) = {slope:+.3f}   "
      f"(1.0 = full transfer, 0 = no transfer)")
print(f"gap mean {np.mean(tr-te):.3f}  std {np.std(tr-te):.3f}  "
      f"(constant gap => train fit just tracks scene difficulty, not a lever)")
print("\n  slope near 1, r high  -> raising train fit RAISES test: capacity/fidelity is a lever")
print("  slope near 0 / const gap -> train fit tracks difficulty; no test lever there")
