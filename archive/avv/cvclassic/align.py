#!/usr/bin/env python
"""Is the ensemble mean blurring itself through inter-member misregistration?

M1: dense flow between members (each -> their mean). If |d| >> 0 the pixel mean is
    averaging displaced copies of the same structure = self-inflicted blur.
M2: dense flow between the mean and GT, per patch, AFTER the global field would be
    applied. Split residual energy into "removable by local warp" vs "not".
M3: is the member<->mean displacement field CONSISTENT across frames (=> fittable
    without GT) or frame-specific?
"""
import os, sys, numpy as np, cv2
from PIL import Image

GT = "/mnt/d/avv/evalsplit/HCM0181/eval_gt"
MEM = [
    "/mnt/d/avv/seedbank/HCM0181_ut_s101/eval_png",
    "/mnt/d/avv/seedbank/HCM0181_ut_s202/eval_png",
    "/mnt/d/avv/mip3d/HCM0181_mf0.2/eval_png",
    "/mnt/d/avv/lpsweep/HCM0181_lp0.3/eval_png",
    "/mnt/d/avv/tw_test/HCM0181_absgrad0/eval_png",
]
N_IMG = int(os.environ.get("NIMG", "8"))
gtmap = {os.path.splitext(f)[0]: os.path.join(GT, f) for f in os.listdir(GT)}
stems = sorted(gtmap)
sel = stems[:: max(1, len(stems) // N_IMG)][:N_IMG]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def gray8(a):
    return (np.clip(0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2], 0, 1) * 255).astype(np.uint8)


dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
dis.setFinestScale(0)

mag_mm = []      # member -> mean
mag_mg = []      # mean   -> GT
fields = []      # per-frame member0->mean field, downsampled, for consistency test
for s in sel:
    g = load(gtmap[s])
    ms = [load(os.path.join(d, s + ".png")) for d in MEM]
    if any(m.shape != g.shape for m in ms):
        continue
    mean = np.stack(ms).mean(0)
    mg8 = gray8(mean)
    for i, m in enumerate(ms):
        fl = dis.calc(mg8, gray8(m), None)
        mag_mm.append(np.linalg.norm(fl, axis=2).mean())
        if i == 0:
            H, W = mg8.shape
            fields.append(cv2.resize(fl, (W // 16, H // 16), interpolation=cv2.INTER_AREA))
    flg = dis.calc(gray8(g), mg8, None)
    mag_mg.append(np.linalg.norm(flg, axis=2).mean())

print(f"frames used: {len(mag_mg)}   members: {len(MEM)}")
print(f"M1 member -> ensemble-mean displacement : mean |d| = {np.mean(mag_mm):.4f} px "
      f"(per-member spread {np.std(mag_mm):.4f})")
print(f"M2 ensemble-mean -> GT displacement     : mean |d| = {np.mean(mag_mg):.4f} px")

# M3 consistency of the member0->mean field across frames
Fs = np.stack(fields)                      # n,h,w,2
mu = Fs.mean(0)
num = ((Fs - mu) ** 2).sum()
den = (Fs ** 2).sum()
print(f"M3 member0->mean field: energy in the FRAME-INVARIANT part "
      f"= {100*(1-num/den):.2f}%   (frame-specific {100*num/den:.2f}%)")
print(f"   invariant field mean |d| = {np.linalg.norm(mu,axis=2).mean():.4f} px")
