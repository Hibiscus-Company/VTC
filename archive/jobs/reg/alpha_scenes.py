#!/usr/bin/env python
"""Independent replication of the claimant's GEOMETRIC route on scenes other than HCM0181.

alpha* = <M,f>/<f,f> where
  f = train-fitted median field (the SHIPPED field)
  M = median over TEST views of DIS flow (test photo -> render at test pose)
M uses public test GT: DIAGNOSTIC ONLY, nothing fitted here ships.

If alpha* ~ 1.3 on one scene only, the scalar is a scene quirk.  If it clusters across scenes,
the amplitude deficit is a property of the pipeline.
CPU only, no GPU.
"""
import os, sys, json
import numpy as np
import cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(2)
DS = 8


def gray8(a):
    return (cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)


def run(scene):
    rd = f"/mnt/d/avv/output/{scene}_gsplatB9ut/test_poses_renders_png"
    gtd = f"/mnt/d/avv/data/phase1/public_set/{scene}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by if os.path.exists(os.path.join(rd, s + ".png")))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    stack = []
    for s in stems:
        r = np.asarray(Image.open(os.path.join(rd, s + ".png")).convert("RGB"),
                       dtype=np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                       dtype=np.float32) / 255.0
        H, W, _ = r.shape
        fl = np.clip(dis.calc(gray8(g), gray8(r), None), -6.0, 6.0)
        stack.append(cv2.resize(fl, (W // DS, H // DS), interpolation=cv2.INTER_AREA))
    T = np.stack(stack).astype(np.float32)
    M = np.median(T, axis=0)
    f = np.median(np.load(f"{HERE}/flow/{scene}.npz")["ds8"].astype(np.float32), axis=0)
    a = float((f * M).sum() / (f * f).sum())
    r2 = lambda g: float(1.0 - ((M - g * f) ** 2).sum() / (M ** 2).sum())
    idx = np.random.RandomState(0).permutation(len(T))
    M1 = np.median(T[idx[:len(T) // 2]], 0); M2 = np.median(T[idx[len(T) // 2:]], 0)
    out = dict(n=len(T), alpha=a, R2_1=r2(1.0), R2_a=r2(a),
               mean_f=float(np.linalg.norm(f, axis=2).mean()),
               mean_M=float(np.linalg.norm(M, axis=2).mean()),
               splithalf_corr=float(np.corrcoef(M1.ravel(), M2.ravel())[0, 1]),
               alpha_half1=float((f * M1).sum() / (f * f).sum()),
               alpha_half2=float((f * M2).sum() / (f * f).sum()))
    print(f"{scene:9s} n={out['n']:3d} alpha*={a:.4f} (halves {out['alpha_half1']:.3f}/"
          f"{out['alpha_half2']:.3f}) R2 {out['R2_1']:.4f}->{out['R2_a']:.4f}  "
          f"|f|={out['mean_f']:.4f} |M|={out['mean_M']:.4f} shc={out['splithalf_corr']:.3f}",
          flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for sc in sys.argv[1:]:
        res[sc] = run(sc)
    json.dump(res, open(f"{HERE}/reg/alpha_scenes.json", "w"), indent=1)
    al = [v["alpha"] for v in res.values()]
    print(f"\nalpha* across {len(al)} scenes: mean {np.mean(al):.4f} "
          f"min {min(al):.4f} max {max(al):.4f}")
