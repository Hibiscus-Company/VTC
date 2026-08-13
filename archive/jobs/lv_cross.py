#!/usr/bin/env python
"""S6: THE CROSS-SCENE DESIGN THAT ACTUALLY DISENTANGLES SCENE FROM REGIME.

The production harness has multi-member renders for exactly ONE scene (HCM0181), so the fitted
lambda could be an HCM0181 artefact.  The eval-split store, however, has MANY members for FOUR
scenes -- HCM0181, HCM0421, chair, bonsai -- all from the same 'tw_test' config-jitter family.
That is the STARVED regime (models see 75-83% of photos), so magnitudes are not production
magnitudes.  But by ALSO measuring HCM0181 in the starved regime at the same k with the same
family, the regime shift is measured directly and can be divided out:

    HCM0181 production k=4  (known)  ---- regime factor ----  HCM0181 eval-split k=4
                                                                     |
                                                              scene variation
                                                                     |
                                                         chair / bonsai / HCM0421 eval-split

Every scene gets the SAME lambda sweep so we can see whether the optimum sits at the same place.
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
from lv_main import H, agg, dscore, tbl, HDR, DEV
Image.MAX_IMAGE_PIXELS = None

TW = "/mnt/d/avv/tw_test"
GTD = "/mnt/d/avv/evalsplit/{s}/eval_gt"

# 4 members per scene, all from the SAME config-jitter family (tw_test), chosen only by
# availability -- no score-based selection, which would bias the comparison.
SETS = {
    "HCM0181 (tower, tuned scene)": (
        "HCM0181", [f"{TW}/HCM0181_{t}/eval_png" for t in
                    ("baseline_screen", "ema099", "aniso01", "minop02")]),
    "chair (indoor video)": (
        "chair", [f"{TW}/chair_{t}/eval_png" for t in
                  ("baseline_screen", "ema099", "aniso01", "minop02")]),
    "bonsai (indoor video)": (
        "bonsai", [f"{TW}/bonsai_{t}/eval_png" for t in
                   ("baseline_screen", "ema099", "tw1.0", "uncweight")]),
}
LAMS = (0.5, 0.75, 1.0, 1.25, 1.5)

if __name__ == "__main__":
    torch.manual_seed(0)
    out = {}
    for label, (scene, dirs) in SETS.items():
        print("\n" + "=" * 96)
        print(f"{label}   [eval-split / starved regime, k=4, tw_test family]")
        print("=" * 96)
        h = H(dirs, gtdir=GTD.format(s=scene), names=[d.split("/")[-2] for d in dirs])
        b = h.run(0, tag=f"xs_{scene}_base")
        print(HDR)
        tbl("pixel-mean k=4", b, b)
        row = {}
        for lam in LAMS:
            d, per = tbl(f"energy lam{lam} win3", h.run(lam, tag=f"xs_{scene}_l{lam}"), b)
            row[lam] = (d, float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))))
        out[scene] = row
        del h
        torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print("SUMMARY: dScore vs lambda, per scene (eval-split regime, k=4).  t = paired per-view t")
    print("=" * 96)
    print(f"{'scene':<12}" + "".join(f"{'lam' + str(l):>16}" for l in LAMS))
    for s, row in out.items():
        print(f"{s:<12}" + "".join(f"{row[l][0]:+9.4f}(t{row[l][1]:5.1f})" for l in LAMS))
    json.dump({s: {str(k): v for k, v in r.items()} for s, r in out.items()},
              open(os.path.join(HERE, "lv_cross.json"), "w"), indent=1)
