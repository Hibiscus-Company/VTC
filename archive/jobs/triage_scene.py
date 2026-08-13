#!/usr/bin/env python
"""Per-scene triage scorer. Same metric as scripts/eval_score.py (repo SSIM + lpips-vgg)."""
import os, sys, json, time
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
torch.set_num_threads(24)
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg

DEV = "cuda" if (len(sys.argv) > 1 and sys.argv[1] == "cuda") else "cpu"
vgg = lpips_pkg.LPIPS(net="vgg").to(DEV).eval()

D = "/mnt/d/avv"
PUB = f"{D}/data/phase1/public_set"

PAIRS = [
    ("0181_SHIPCHAIN", f"{D}/prodharness/k4f", f"{PUB}/HCM0181/test/images", "prod-real-GT full chain"),
    ("0181_solo",  f"{D}/output/HCM0181_gsplatB9ut/test_poses_renders_png", f"{PUB}/HCM0181/test/images", "prod-real-GT solo"),
    ("0193_solo",  f"{D}/output/HCM0193_gsplatB9ut/test_poses_renders_png", f"{PUB}/HCM0193/test/images", "prod-real-GT solo"),
    ("0204_solo",  f"{D}/output/HCM0204_gsplatB9ut/test_poses_renders_png", f"{PUB}/HCM0204/test/images", "prod-real-GT solo"),
    ("0031_solo",  f"{D}/output/hcm0031_gsplatB9ut/test_poses_renders_png", f"{PUB}/hcm0031/test/images", "prod-real-GT solo"),
    ("0034_solo",  f"{D}/output/hcm0034_gsplatB9ut/test_poses_renders_png", f"{PUB}/hcm0034/test/images", "prod-real-GT solo"),
    ("ES_0181",    f"{D}/lpsweep/HCM0181_lp0.3/eval_png", f"{D}/evalsplit/HCM0181/eval_gt", "evalsplit proxy"),
    ("ES_0421",    f"{D}/evalgen/HCM0421/eval_png",      f"{D}/evalsplit/HCM0421/eval_gt", "evalsplit proxy"),
    ("ES_chair",   f"{D}/chair_eval/lpearly60k/eval_png", f"{D}/evalsplit/chair/eval_gt",  "evalsplit proxy"),
    ("ES_bonsai",  f"{D}/bonsai_perc/pC_lpearly/eval_png", f"{D}/evalsplit/bonsai/eval_gt", "evalsplit proxy"),
    ("ES_0181_dp", f"{D}/depth/HCM0181_dp05/eval_png",    f"{D}/evalsplit/HCM0181/eval_gt", "evalsplit proxy"),
    ("ES_chair_b", f"{D}/chair_eval/base60k/eval_png",    f"{D}/evalsplit/chair/eval_gt",   "evalsplit proxy"),
]

def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0)

out = []
for tag, rd, gd, regime in PAIRS:
    if not (os.path.isdir(rd) and os.path.isdir(gd)):
        print(f"MISSING {tag}: {rd} | {gd}", flush=True); continue
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gd)}
    rn_by = {os.path.splitext(f)[0]: f for f in os.listdir(rd)
             if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")}
    stems = sorted(set(gt_by) & set(rn_by))
    drop = (len(gt_by) - len(stems), len(rn_by) - len(stems))
    P = S = L = 0.0
    t0 = time.time()
    ok = True
    with torch.no_grad():
        for s in stems:
            r = load(os.path.join(rd, rn_by[s])).to(DEV)
            g = load(os.path.join(gd, gt_by[s])).to(DEV)
            if r.shape != g.shape:
                print(f"SHAPE MISMATCH {tag} {s}: {tuple(r.shape)} vs {tuple(g.shape)}", flush=True)
                ok = False; break
            mse = ((r - g) ** 2).mean().item()
            P += 10 * np.log10(1.0 / max(mse, 1e-12))
            S += float(repo_ssim(r, g))
            L += float(vgg(r * 2 - 1, g * 2 - 1).item())
    if not ok:
        continue
    n = len(stems)
    P, S, L = P / n, S / n, L / n
    sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    rec = dict(tag=tag, regime=regime, n=n, dropped_gt=drop[0], dropped_render=drop[1],
               psnr=P, ssim=S, lpips=L, score=sc, secs=round(time.time() - t0, 1))
    out.append(rec)
    print(f"RES {tag:15s} n={n:3d} drop={drop} PSNR {P:8.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {sc:.4f} ({rec['secs']}s)", flush=True)
    json.dump(out, open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/triage.json", "w"), indent=1)
print("DONE", flush=True)
