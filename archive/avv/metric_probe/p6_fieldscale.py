"""P6: is the shipped lens field UNDER-scaled?

The field is the mean of DIS optical flows.  DIS is a regularised estimator: it shrinks small
displacements toward zero.  If the true displacement is d and DIS reports alpha*d with alpha<1,
every field we ship under-corrects by (1-alpha) and there is free score on the table.

Test: take the PRODUCTION field (fit on all 240 TRAIN views of the production model), apply it at
several SCALES to eval-split renders from a DIFFERENT model, score against held-out train photos.
A peak above 1.0 means systematic under-correction.
"""
import os, sys, json
import numpy as np
import cv2
import torch
import mlib

torch.set_num_threads(int(os.environ.get("NT", "6")))
cv2.setNumThreads(int(os.environ.get("NT", "6")))


def apply(img8, field, scale):
    H, W, _ = img8.shape
    fu = cv2.resize(field * scale, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = cv2.remap(img8.astype(np.float32) / 255.0,
                    (xx + fu[..., 0]).astype(np.float32),
                    (yy + fu[..., 1]).astype(np.float32),
                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


CASES = {
    "HCM0421": ("/mnt/d/avv/evalgen/HCM0421/eval_png", "/mnt/d/avv/evalsplit/HCM0421/eval_gt",
                "/mnt/d/avv/r2r9/fields/HCM0421.npy", ["DJI_20241230093301_0003_V"]),
    "chair": ("/mnt/d/avv/chair_eval/base60k/eval_png", "/mnt/d/avv/evalsplit/chair/eval_gt",
              "/mnt/d/avv/r2r9/fields/chair.npy", []),
}
key = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rd, gd, fp, skip = CASES[key]
fld = np.load(fp)
print(f"field {fp}: rms {np.sqrt((fld**2).sum(-1)).mean():.3f} px  max {np.sqrt((fld**2).sum(-1)).max():.3f}")
gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gd)}
files = [f for f in sorted(os.listdir(rd)) if os.path.splitext(f)[0] not in skip]
files = files[:: max(1, len(files) // N)][:N]
lp = mlib.LP("cpu")
SC = [0.0, 0.75, 1.0, 1.25, 1.5, 1.75]
acc = {s: [0.0, 0.0, 0.0, 0] for s in SC}
for i, f in enumerate(files):
    s_ = os.path.splitext(f)[0]
    r8 = mlib.load_u8(os.path.join(rd, f))
    g = mlib.to_t(mlib.load_u8(os.path.join(gd, gt_by[s_])))
    for s in SC:
        y = mlib.to_t(r8 if s == 0 else apply(r8, fld, s))
        acc[s][0] += mlib.psnr(y, g); acc[s][1] += float(mlib.ssim(y, g))
        acc[s][2] += lp(y, g); acc[s][3] += 1
    print(f"  [{i+1}/{len(files)}] {s_}", flush=True)
print(f"\n=== P6 {key} n={len(files)} production-field scale sweep ===")
rows = [(s, acc[s][0] / acc[s][3], acc[s][1] / acc[s][3], acc[s][2] / acc[s][3]) for s in SC]
rows = [(s, P, S, L, mlib.score(P, S, L)) for s, P, S, L in rows]
ref = rows[0]
for s, P, S, L, sc in rows:
    print(f"scale {s:4.2f}  PSNR {P:7.4f}({P-ref[1]:+.4f})  SSIM {S:.5f}({S-ref[2]:+.5f})  "
          f"LPIPS {L:.5f}({L-ref[3]:+.5f})  SCORE {sc:8.4f}  d {sc-ref[4]:+.4f}")
json.dump(rows, open(f"/mnt/d/avv/metric_probe/p6_{key}.json", "w"), indent=1)
