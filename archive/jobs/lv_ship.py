#!/usr/bin/env python
"""S7: THE EXACT SHIPPING CONFIGURATION, end to end.

    members -> weighted pixel mean -> LEVEL-0 ENERGY RESTORATION -> median lens field
            -> JPEG q100 ss2 -> score against REAL test GT

Two map estimators are measured, because they have very different production costs:
  FULL  : needs every member render at the test poses (towers would have to be re-rendered from
          the surviving r2r9 seed checkpoints -- pure inference, but a pipeline step).
  DEV-1 : needs the EXISTING ensemble PNG plus ONE surviving member render, using
          mean_i E_i = E_mean + mean_i E(L_i - L_mean).  r22_seed101 and r25_mip3d already
          exist on disk for all five towers, so this route costs ZERO new rendering.
"""
import os, sys, io
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track")
sys.path.insert(0, HERE)
from lv_main import H, agg, dscore, tbl, HDR, DEV, UT, D, JPEG_KW
from lv_s1b import fuse2
from fit_field import apply_field

FIELD = np.load(os.path.join(HERE, "HCM0181_median.npy"))


def chain(h, lam, devsub=None, field=True, jpeg=True):
    rows = []
    for i in range(len(h.stems)):
        st = h.stack(i)
        img = st.mean(0, keepdim=True) if lam == 0 else fuse2(st, lam, devsub=devsub)[0]
        a = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        if field:
            a = (np.clip(apply_field(a.astype(np.float32) / 255.0, FIELD), 0, 1) * 255
                 ).round().astype(np.uint8)
        if jpeg:
            buf = io.BytesIO(); Image.fromarray(a).save(buf, "JPEG", **JPEG_KW); buf.seek(0)
            a = np.asarray(Image.open(buf).convert("RGB"))
        rows.append(h.eval_img(torch.from_numpy(np.ascontiguousarray(a)), i))
    return np.array(rows)


if __name__ == "__main__":
    torch.manual_seed(0)
    h = H([D(t) for t in UT], names=UT)
    print("\n" + "=" * 96)
    print("S7. SHIPPED CHAIN  (ensemble -> energy restore -> median field lanczos4 -> JPEG q100 ss2)")
    print("=" * 96)
    print(HDR)
    b = chain(h, 0)
    tbl("pixel-mean  [SHIPPED TODAY]", b, b)
    res = {}
    for lam in (0.75, 1.0, 1.25):
        res[("full", lam)] = tbl(f"FULL map   lam{lam}", chain(h, lam), b)[0]
    print()
    for lam in (0.75, 1.0, 1.25):
        res[("dev2", lam)] = tbl(f"DEV-2 map  lam{lam}", chain(h, lam, devsub=[0, 1]), b)[0]
    print()
    for lam in (0.75, 1.0, 1.25):
        res[("dev1", lam)] = tbl(f"DEV-1 map  lam{lam}", chain(h, lam, devsub=[0]), b)[0]
    print()
    for lam in (0.75, 1.0, 1.25):
        res[("dev1b", lam)] = tbl(f"DEV-1 map (member 3) lam{lam}", chain(h, lam, devsub=[2]), b)[0]
    print("\n  estimator x lambda, dScore on the shipped chain")
    print(f"  {'est':<8}" + "".join(f"{l:>10}" for l in (0.75, 1.0, 1.25)))
    for e in ("full", "dev2", "dev1", "dev1b"):
        print(f"  {e:<8}" + "".join(f"{res[(e, l)]:+10.4f}" for l in (0.75, 1.0, 1.25)))
