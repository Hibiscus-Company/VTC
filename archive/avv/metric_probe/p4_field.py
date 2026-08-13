"""P4: 2-fold cross-validated lens-field test on the scenes that ship UNWARPED.

bonsai ships with no field at all, yet its mean render->photo displacement measures 1.33 px rms --
4.6x the tower field that is worth +0.73 on the leaderboard.  Fit the field on fold A, apply to
fold B (and vice versa), score all three metrics.  Nothing self-fitted.
"""
import os, sys, json
import numpy as np
import cv2
import torch
import mlib

torch.set_num_threads(int(os.environ.get("NT", "6")))
cv2.setNumThreads(int(os.environ.get("NT", "6")))

CASES = {
    "bonsai": ("/mnt/d/avv/bonsai_eval/K4_pC_seed7/eval_png", "/mnt/d/avv/evalsplit/bonsai2/eval_gt", []),
    "chair": ("/mnt/d/avv/chair_eval/base60k/eval_png", "/mnt/d/avv/evalsplit/chair/eval_gt", []),
    "HCM0421": ("/mnt/d/avv/evalgen/HCM0421/eval_png", "/mnt/d/avv/evalsplit/HCM0421/eval_gt",
                ["DJI_20241230093301_0003_V"]),
}


def fit(pairs, ds=8, clip=6.0):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    acc = None
    for r8, g8 in pairs:
        rg = cv2.cvtColor(r8, cv2.COLOR_RGB2GRAY)
        gg = cv2.cvtColor(g8, cv2.COLOR_RGB2GRAY)
        fl = np.clip(dis.calc(gg, rg, None), -clip, clip)
        H, W = rg.shape
        s = cv2.resize(fl, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
        acc = s if acc is None else acc + s
    return acc / len(pairs)


def apply(img8, field, scale=1.0):
    H, W, _ = img8.shape
    fu = cv2.resize(field * scale, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = cv2.remap(img8.astype(np.float32) / 255.0,
                    (xx + fu[..., 0]).astype(np.float32),
                    (yy + fu[..., 1]).astype(np.float32),
                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


key = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
rd, gd, skip = CASES[key]
gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gd)}
files = [f for f in sorted(os.listdir(rd)) if os.path.splitext(f)[0] not in skip]
files = files[:: max(1, len(files) // N)][:N]
R8 = [mlib.load_u8(os.path.join(rd, f)) for f in files]
G8 = [mlib.load_u8(os.path.join(gd, gt_by[os.path.splitext(f)[0]])) for f in files]

lp = mlib.LP("cpu")
A = list(range(0, len(files), 2)); B = list(range(1, len(files), 2))
SCALES = [0.0, 0.5, 0.75, 1.0]
acc = {s: [0.0, 0.0, 0.0, 0] for s in SCALES}
for fitset, testset in ((A, B), (B, A)):
    fld = fit([(R8[i], G8[i]) for i in fitset])
    m = np.sqrt((fld ** 2).sum(-1))
    print(f"  fold field: rms {m.mean():.3f} px  max {m.max():.3f}", flush=True)
    for i in testset:
        g = mlib.to_t(G8[i])
        for s in SCALES:
            y = mlib.to_t(R8[i] if s == 0 else apply(R8[i], fld, s))
            acc[s][0] += mlib.psnr(y, g); acc[s][1] += float(mlib.ssim(y, g))
            acc[s][2] += lp(y, g); acc[s][3] += 1
        print(f"   img {i} done", flush=True)

print(f"\n=== P4 {key} n={len(files)} 2-fold CV field ===")
rows = []
for s in SCALES:
    P, S, L, k = acc[s]
    rows.append((s, P / k, S / k, L / k, mlib.score(P / k, S / k, L / k)))
ref = rows[0]
for s, P, S, L, sc in rows:
    print(f"scale {s:4.2f}  PSNR {P:7.4f}({P-ref[1]:+.4f})  SSIM {S:.5f}({S-ref[2]:+.5f})  "
          f"LPIPS {L:.5f}({L-ref[3]:+.5f})  SCORE {sc:8.4f}  d {sc-ref[4]:+.4f}")
json.dump(rows, open(f"/mnt/d/avv/metric_probe/p4_{key}.json", "w"), indent=1)
