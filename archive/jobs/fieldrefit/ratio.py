#!/usr/bin/env python
"""WHY THE TRAIN-FITTED FIELD UNDERSHOOTS -- measured, not argued. DIAGNOSTIC ONLY.

The field is fit on TRAIN renders vs TRAIN photos and applied at TEST poses. The model was
optimised against those train photos, so its train renders are ALREADY better registered than
its test renders: the train-fitted field is a SHRUNK copy of the field the test poses need.

For each public tower, fit the ORACLE field at TEST poses against REAL TEST GT (never shipped --
same class as the D5-D8 oracle fields already on disk) and compare it to the shipped train-fitted
field. Report
    ratio  = mean|f_test| / mean|f_train|
    s*     = <f_train, f_test> / <f_train, f_train>   (the LS-optimal scalar to multiply by)
    corr   = cos angle between them   (if this is high, only the SIZE is wrong, not the SHAPE)

If s* is stable across scenes, a single constant transfers to the private towers.
"""
import os, sys, time, json
import numpy as np
import cv2
from PIL import Image

cv2.setNumThreads(2)
Image.MAX_IMAGE_PIXELS = None
HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, os.path.join(HERE, "lens"))

TEST_REN = {
    "HCM0181": "/mnt/d/avv/prodharness/k4/png",                                  # k4 ensemble
    "HCM0193": "/mnt/d/avv/output/HCM0193_gsplatB9ut/test_poses_renders_png",
    "HCM0204": "/mnt/d/avv/output/HCM0204_gsplatB9ut/test_poses_renders_png",
    "hcm0031": "/mnt/d/avv/output/hcm0031_gsplatB9ut/test_poses_renders_png",
    "hcm0034": "/mnt/d/avv/output/hcm0034_gsplatB9ut/test_poses_renders_png",
}
EXTRA = {"HCM0181_B9single": "/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png"}


def gray8(x):
    return (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)


def ld(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def oracle(tag, rd, ds=8):
    gtd = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    st = []
    for f in sorted(os.listdir(rd)):
        s = os.path.splitext(f)[0]
        if not f.endswith(".png") or s not in gt_by:
            continue
        r, g = ld(os.path.join(rd, f)), ld(os.path.join(gtd, gt_by[s]))
        if r.shape != g.shape:
            continue
        H, W, _ = g.shape
        fl = np.clip(dis.calc(gray8(g), gray8(r), None), -6.0, 6.0)
        st.append(cv2.resize(fl, (W // ds, H // ds), interpolation=cv2.INTER_AREA))
    return np.median(np.stack(st), 0), len(st)


def main():
    out = {}
    t0 = time.time()
    print(f"{'scene':>18} {'n':>4} {'|f_train|':>10} {'|f_test|':>9} {'ratio':>7} "
          f"{'s*':>7} {'corr':>6}")
    jobs = [(t, t, d) for t, d in TEST_REN.items()] + \
           [(k, k.split("_")[0], d) for k, d in EXTRA.items()]
    for name, tag, rd in jobs:
        tr = np.median(np.load(f"{HERE}/lens/cache/pub_{tag}.npz")["s8"].astype(np.float32), 0)
        te, n = oracle(tag, rd)
        if te.shape != tr.shape:
            print(f"{name:>18} shape mismatch {te.shape} vs {tr.shape}")
            continue
        mt, me = np.linalg.norm(tr, axis=2).mean(), np.linalg.norm(te, axis=2).mean()
        num, den = float((tr * te).sum()), float((tr * tr).sum())
        c = num / np.sqrt(den * float((te * te).sum()))
        out[name] = dict(n=n, mag_train=mt, mag_test=me, ratio=me / mt, s_star=num / den, corr=c)
        print(f"{name:>18} {n:4d} {mt:10.4f} {me:9.4f} {me/mt:7.3f} {num/den:7.3f} {c:6.3f}",
              flush=True)
    ss = [v["s_star"] for k, v in out.items() if k in TEST_REN]
    print(f"\ns* over the 5 public towers: mean {np.mean(ss):.3f}  sd {np.std(ss):.3f}  "
          f"min {min(ss):.3f}  max {max(ss):.3f}   ({time.time()-t0:.0f}s)")
    json.dump(out, open(os.path.join(HERE, "fieldrefit", "res_ratio.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
