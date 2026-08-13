#!/usr/bin/env python
"""r37 bonsai COMPOSITION: how should long45k members mix with the scale_reg-0.1-30k members?

r36's bonsai is 13 members, all on recipes that measure ~0.76 BELOW long45k on the eval split.
The standing rules pull in opposite directions here:
  - ADD-not-REPLACE is LB-proven (r20 +0.0948) and pure-new lost the r36-era mixsweep
  - but the member-quality rule says only ensemble members within ~0.15 of the best single model,
    and the old members are 0.76 away now -- that rule has never been tested at this gap
So it has to be measured, not argued. Everything here is ENCODED through the ship JPEG and scored
with the real metric; mixsweeps inflate ~2.2x vs the LB so these are for RANKING only.
"""
import os, io, re, sys, itertools, numpy as np, torch
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from PIL import Image
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(6)
GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
OLD = {  # the recipe class r36's bonsai is actually built from (scale_reg 0.1, 30k, churn 15k/8k)
    "o1": "/mnt/d/avv/r36_shape/sr01/eval_png",
    "o2": "/mnt/d/avv/r36_shape/sr01b/eval_png",
    "o3": "/mnt/d/avv/r38/sr01_s101/eval_png",
}
NEW = {  # long45k, gate-passed at +0.9043 paired
    "n1": "/mnt/d/avv/r43_bonsai/long45k/eval_png",
    "n2": "/mnt/d/avv/r43_bonsai/long45k_s101/eval_png",
}
MID = {  # churn_r25n25, the intermediate recipe (also gate-passed, +0.4630)
    "m1": "/mnt/d/avv/r43_bonsai/churn_r25n25/eval_png",
    "m2": "/mnt/d/avv/r43_bonsai/churn_r25n25_s101/eval_png",
}
ALL = {**OLD, **NEW, **MID}
ENC = dict(quality=100, subsampling=2, optimize=True, progressive=True)
FIRST8 = {10, 120, 190, 260, 440, 510, 630, 710}
gt = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
stems = sorted(gt)
vgg = lpips_pkg.LPIPS(net="vgg").eval()
cache = {k: {} for k in ALL}
for k, d in ALL.items():
    for s in stems:
        cache[k][s] = np.asarray(Image.open(f"{d}/{s}.png").convert("RGB"), dtype=np.float32)
G = {s: torch.from_numpy(np.asarray(Image.open(f"{GT}/{gt[s]}").convert("RGB"), dtype=np.float32) / 255.
                         ).permute(2, 0, 1).unsqueeze(0) for s in stems}
def sc(P, S, L): return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1))
def score(keys):
    per = {}
    with torch.no_grad():
        for s in stems:
            acc = sum(cache[k][s] for k in keys) / len(keys)
            b = io.BytesIO()
            Image.fromarray(np.clip(acc + 0.5, 0, 255).astype(np.uint8)).save(b, "JPEG", **ENC)
            r = torch.from_numpy(np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                                            dtype=np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)
            g = G[s]
            per[int(re.findall(r"\d+", s)[0])] = (
                10 * np.log10(1 / max(((r - g) ** 2).mean().item(), 1e-12)),
                float(repo_ssim(r, g)), float(vgg(r * 2 - 1, g * 2 - 1).item()))
    ks = sorted(per); f8 = [k for k in ks if k in FIRST8]; l20 = [k for k in ks if k not in FIRST8]
    a = lambda kk: sc(*np.array([per[k] for k in kk]).mean(0))
    return a(ks), a(f8), a(l20)
MIXES = [
    ("3 old only (the r36 recipe class)", ["o1", "o2", "o3"]),
    ("2 new only (long45k)",              ["n1", "n2"]),
    ("2 mid only (churn_r25n25)",         ["m1", "m2"]),
    ("3 old + 2 new",                     ["o1", "o2", "o3", "n1", "n2"]),
    ("2 old + 2 new",                     ["o1", "o2", "n1", "n2"]),
    ("1 old + 2 new",                     ["o1", "n1", "n2"]),
    ("2 mid + 2 new",                     ["m1", "m2", "n1", "n2"]),
    ("3 old + 2 mid + 2 new (everything)",["o1", "o2", "o3", "m1", "m2", "n1", "n2"]),
]
print(f"{'composition':<38}{'k':>3}{'SCORE':>10}{'first8':>10}{'last20':>10}", flush=True)
res = []
for name, keys in MIXES:
    S, S8, S20 = score(keys)
    res.append((S, name, len(keys), S8, S20))
    print(f"{name:<38}{len(keys):>3}{S:>10.4f}{S8:>10.4f}{S20:>10.4f}", flush=True)
print("\nranked:", flush=True)
for S, name, k, S8, S20 in sorted(res, reverse=True):
    print(f"  {S:.4f}  k={k:<2} {name}", flush=True)
