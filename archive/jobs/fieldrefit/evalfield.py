#!/usr/bin/env python
"""CAN THE SHRINKAGE BE SEEN WITHOUT TEST GT?  No test photo is touched anywhere in this script.

The train-fitted field undershoots because the model was optimised against the very photos the
field is fit on. /mnt/d/avv/evalsplit holds models trained on a SUBSET of the train photos, and
/mnt/d/avv/evalgen holds their renders at the HELD-OUT train poses, whose photos (eval_gt) are
TRAIN photos the model never saw. So:

    f_held = DIS(eval_gt photo, eval-split render at that pose)   <- unfitted registration
    f_fit  = the production field, fit on a model's OWN train renders vs its OWN train photos
    s*     = <f_fit, f_held> / <f_fit, f_fit>

s* is the rescale the fitted field needs to reach the registration of a pose the model did not
train on -- which is exactly the test-pose regime. Rule 10 is untouched: eval_gt lives under
.../evalsplit/<scene>/eval_gt, never under a test/ directory.

HCM0421 is a PRIVATE tower, so this is the only estimate of the constant that can be made on a
scene we actually submit.
"""
import os, sys, time
import numpy as np
import cv2
from PIL import Image

cv2.setNumThreads(4)
Image.MAX_IMAGE_PIXELS = None
HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"

JOBS = {
    # scene: (held-out render dir, held-out photo dir, the production train-fitted field)
    "HCM0421 (PRIVATE)": ("/mnt/d/avv/evalgen/HCM0421/eval_png",
                          "/mnt/d/avv/evalsplit/HCM0421/eval_gt",
                          "/mnt/d/avv/fields_median/HCM0421.npy"),
}


def gray8(x):
    return (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)


def ld(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def held_field(rd, gd, ds=8):
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gd)}
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    st = []
    for f in sorted(os.listdir(rd)):
        s = os.path.splitext(f)[0]
        if not f.lower().endswith(".png") or s not in gt_by:
            continue
        r, g = ld(os.path.join(rd, f)), ld(os.path.join(gd, gt_by[s]))
        if r.shape != g.shape:
            continue
        H, W, _ = g.shape
        fl = np.clip(dis.calc(gray8(g), gray8(r), None), -6.0, 6.0)
        st.append(cv2.resize(fl, (W // ds, H // ds), interpolation=cv2.INTER_AREA))
    return np.median(np.stack(st), 0).astype(np.float32), len(st)


def main():
    t0 = time.time()
    assert os.sep + "test" + os.sep not in str(JOBS), "Rule 10: no test dir may appear here"
    print(f"{'scene':>20} {'n':>4} {'|f_fit|':>8} {'|f_held|':>9} {'ratio':>7} {'s*':>7} {'corr':>6}")
    for name, (rd, gd, fp) in JOBS.items():
        fit = np.load(fp)
        held, n = held_field(rd, gd)
        if held.shape != fit.shape:
            held = cv2.resize(held, (fit.shape[1], fit.shape[0]), interpolation=cv2.INTER_CUBIC)
        mf, mh = np.linalg.norm(fit, axis=2).mean(), np.linalg.norm(held, axis=2).mean()
        num, den = float((fit * held).sum()), float((fit * fit).sum())
        c = num / np.sqrt(den * float((held * held).sum()))
        print(f"{name:>20} {n:4d} {mf:8.4f} {mh:9.4f} {mh/mf:7.3f} {num/den:7.3f} {c:6.3f}",
              flush=True)
    print(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
