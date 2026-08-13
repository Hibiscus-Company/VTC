"""DIAGNOSTIC ONLY -- never produces a shippable field.

Measures the ratio between the misregistration at TEST poses (which the model never saw) and
at TRAIN poses (which it was fitted to).  If the model absorbs part of the misregistration at
its own training views, the train-fitted field understates the test-pose field by this ratio,
and that -- not any estimator bias -- is what the scale multiplier corrects.

Writes only a ratio to stdout.  No .npy is saved anywhere the submission pipeline can see.
"""
import os
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(6)
PUB = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]


def fit(render_dir, gt_dir, ds=8, clip=6.0):
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)}
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    st = []
    for f in sorted(os.listdir(render_dir)):
        stem = os.path.splitext(f)[0]
        if not f.lower().endswith(".png") or stem not in gt_by:
            continue
        r = np.asarray(Image.open(os.path.join(render_dir, f)).convert("RGB"), np.float32) / 255
        g = np.asarray(Image.open(os.path.join(gt_dir, gt_by[stem])).convert("RGB"), np.float32) / 255
        if r.shape != g.shape:
            continue
        H, W, _ = r.shape
        rg = (cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        gg = (cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        st.append(cv2.resize(np.clip(dis.calc(gg, rg, None), -clip, clip),
                             (W // ds, H // ds), interpolation=cv2.INTER_AREA))
    return np.median(np.stack(st), 0)


print(f"{'scene':10s} {'train |d|':>10s} {'test |d|':>10s} {'LS k':>7s} {'|test|/|train|':>15s} {'corr':>7s}")
ks = []
for s in PUB:
    tr = fit(f"/mnt/d/avv/output/{s}_gsplatB9ut/train_renders",
             f"/mnt/d/avv/data/phase1/public_set/{s}/train/images")
    te = fit(f"/mnt/d/avv/output/{s}_gsplatB9ut/test_poses_renders_png",
             f"/mnt/d/avv/data/phase1/public_set/{s}/test/images")
    k = float((tr * te).sum() / (tr * tr).sum())
    r = float(np.corrcoef(tr.ravel(), te.ravel())[0, 1])
    a = float(np.linalg.norm(tr, axis=2).mean()); b = float(np.linalg.norm(te, axis=2).mean())
    ks.append(k)
    print(f"{s:10s} {a:10.4f} {b:10.4f} {k:7.3f} {b/a:15.3f} {r:7.3f}", flush=True)
print(f"mean LS k (test = k * train) = {np.mean(ks):.3f}")
