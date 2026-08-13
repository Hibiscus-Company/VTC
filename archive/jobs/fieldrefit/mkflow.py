#!/usr/bin/env python
"""Build DIS flow stacks (GT train photo -> render) for several FIT TARGETS on HCM0181.

Identical flow computation to lens/flowcache.py and gsplat_track/fit_field.py:
DIS MEDIUM preset, clip 6 px, INTER_AREA downsample of the full-res flow.

Fit targets:
  B9    single member gsplatB9ut          (what the shipped field is fit on: one model)
  B11   single member gsplatB11ut60k      (a DIFFERENT single member -- the control)
  M2    pixel mean of B9 and B11          (an ENSEMBLE fit target)
  M2ER  M2 after energy restore lam=1.0   (matches the shipped chain's actual input to the warp)
"""
import os, sys, time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lens")
Image.MAX_IMAGE_PIXELS = None

TAG = "HCM0181"
GT = f"/mnt/d/avv/data/phase1/public_set/{TAG}/train/images"
R9 = f"/mnt/d/avv/output/{TAG}_gsplatB9ut/train_renders"
R11 = f"/mnt/d/avv/output/{TAG}_gsplatB11ut60k/train_renders"
OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/fieldrefit"
DSL = (4, 8, 16)


def ld(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def gray8(x):
    return (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)


def main():
    import torch
    from energy_restore import restore

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
    stems = sorted(s for s in
                   (os.path.splitext(f)[0] for f in os.listdir(R9) if f.endswith(".png"))
                   if s in gt_by and os.path.exists(os.path.join(R11, s + ".png")))
    print(f"{len(stems)} matched train pairs", flush=True)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    targets = ["B9", "B11", "M2", "M2ER"]
    out = {t: {f"s{d}": [] for d in DSL} for t in targets}
    t0 = time.time()
    for n, s in enumerate(stems):
        g = ld(os.path.join(GT, gt_by[s]))
        a = ld(os.path.join(R9, s + ".png"))
        b = ld(os.path.join(R11, s + ".png"))
        if a.shape != g.shape or b.shape != g.shape:
            continue
        H, W, _ = g.shape
        m2 = 0.5 * (a + b)
        # energy restore with the TRUE production k=8 correction, deviation subset = the two
        # members we have (the operator's own docstring licenses a strict subset)
        ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
        tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0)
        tm = torch.from_numpy(m2).permute(2, 0, 1).unsqueeze(0)
        er = restore(tm, [ta, tb], 1.0, 8).clamp(0, 1)[0].permute(1, 2, 0).numpy()
        gg = gray8(g)
        for t, img in (("B9", a), ("B11", b), ("M2", m2), ("M2ER", er)):
            fl = np.clip(dis.calc(gg, gray8(np.ascontiguousarray(img)), None), -6.0, 6.0)
            for d in DSL:
                out[t][f"s{d}"].append(
                    cv2.resize(fl, (W // d, H // d),
                               interpolation=cv2.INTER_AREA).astype(np.float16))
        if n % 10 == 0:
            print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    for t in targets:
        np.savez_compressed(os.path.join(OUT, f"flow_{TAG}_{t}.npz"),
                            stems=np.array(stems), HW=np.array([H, W]),
                            **{k: np.stack(v) for k, v in out[t].items()})
        print(f"wrote flow_{TAG}_{t}.npz", flush=True)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
