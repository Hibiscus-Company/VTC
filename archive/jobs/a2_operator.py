#!/usr/bin/env python
"""Does an operator KEYED ON THE LEGAL PREDICTION beat the best global operator,
and what fraction of the oracle-keyed version does it recover?

Operator: unsharp mask  out = clip(I + a*(I - G_1.0(I)), 0, 1)  with a per-frame amount
    a_i = clip(a0 + k*(key_i - mean(key)), 0, 2)
key = (target log_lapvar) - (render log_lapvar),  i.e. how much sharper the photo should be.
  arm 'global' : k = 0
  arm 'legal'  : key from the +-neighbour predictor (no GT of the hole)
  arm 'oracle' : key from the hole's ACTUAL GT log_lapvar
CPU only.
"""
import os, sys, csv, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp")
import torch
torch.set_num_threads(6)
cv2.setNumThreads(1)
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
from blur_predictor import BlurPredictor, STRIDE

OUT = "/mnt/d/avv/r42_bonsai78/a2_blurpred"
EVAL = sorted(int(x) for x in json.load(open("/mnt/d/avv/evalsplit/bonsai/split.json"))["eval_frames"])
RD = "/mnt/d/avv/r36_shape/sr01/eval_png"
GD = "/mnt/d/avv/evalsplit/bonsai/eval_gt"

def load_csv(p):
    return {int(r["frame"]): {k: float(v) for k, v in r.items() if k != "name"}
            for r in csv.DictReader(open(p))}
T = load_csv(f"{OUT}/train248_sharp.csv"); R = load_csv(f"{OUT}/render_sr01_sharp.csv")
SUB = [f for f in sorted(T) if f not in set(EVAL)]
bp = BlurPredictor.fit({f: T[f]["log_lapvar"] for f in SUB}, mode="nw")
pred = bp.predict_many(EVAL)
pred1 = bp.predict_many(EVAL, {f: (f + STRIDE,) for f in EVAL})
gtv = np.array([T[f]["log_lapvar"] for f in EVAL])
rnv = np.array([R[f]["log_lapvar"] for f in EVAL])

KEYS = dict(legal=pred - rnv, legal1nb=pred1 - rnv, oracle=gtv - rnv)
for k in KEYS: KEYS[k] = KEYS[k] - KEYS[k].mean()

print("loading images...", flush=True)
imgs = {}
for f in EVAL:
    r = np.asarray(Image.open(f"{RD}/frame_{f:06d}.png").convert("RGB"), np.float32) / 255.0
    g = np.asarray(Image.open(f"{GD}/frame_{f:06d}.jpg").convert("RGB"), np.float32) / 255.0
    imgs[f] = (r, cv2.GaussianBlur(r, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE),
               torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0))
vgg = lpips_pkg.LPIPS(net="vgg").eval()

def score_arm(amounts):
    P = S = L = 0.0
    with torch.no_grad():
        for f, a in zip(EVAL, amounts):
            r, rb, gt = imgs[f]
            o = r if abs(a) < 1e-6 else np.clip(r + a * (r - rb), 0, 1)
            # go through uint8 like a real submission does
            o = torch.from_numpy(np.round(o * 255).astype(np.float32) / 255.0
                                 ).permute(2, 0, 1).unsqueeze(0)
            mse = ((o - gt) ** 2).mean().item()
            P += 10 * np.log10(1 / max(mse, 1e-12))
            S += float(repo_ssim(o, gt))
            L += float(vgg(o * 2 - 1, gt * 2 - 1).item())
    n = len(EVAL); P, S, L = P / n, S / n, L / n
    return P, S, L, 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1))

rows = []
t0 = time.time()
plan = [("global", 0.0, 0.0), ("global", 0.25, 0.0), ("global", 0.50, 0.0)]
best_a0 = None
for arm, a0, k in plan:
    a = np.full(len(EVAL), a0)
    P, S, L, sc = score_arm(a)
    rows.append((arm, a0, k, P, S, L, sc))
    print(f"{arm:9s} a0={a0:.2f} k={k:.2f}  PSNR {P:.4f} SSIM {S:.4f} LPIPS {L:.4f} "
          f"SCORE {sc:.4f}   [{time.time()-t0:.0f}s]", flush=True)
best_a0 = max(rows, key=lambda r: r[6])[1]
print(f"--> best global a0 = {best_a0}", flush=True)
for key in ["legal", "oracle", "legal1nb"]:
    for k in [0.40]:
        a = np.clip(best_a0 + k * KEYS[key], 0.0, 2.0)
        P, S, L, sc = score_arm(a)
        rows.append((key, best_a0, k, P, S, L, sc))
        print(f"{key:9s} a0={best_a0:.2f} k={k:.2f}  PSNR {P:.4f} SSIM {S:.4f} LPIPS {L:.4f} "
              f"SCORE {sc:.4f}   [{time.time()-t0:.0f}s]", flush=True)
with open(f"{OUT}/operator_sweep.csv", "w") as fh:
    fh.write("arm,a0,k,psnr,ssim,lpips,score\n")
    for r in rows:
        fh.write(",".join(str(x) for x in r) + "\n")
print("wrote", f"{OUT}/operator_sweep.csv")
