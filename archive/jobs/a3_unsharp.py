"""A3 direction test: per-image unsharp mask on the fixed renders. Converts the
'render is missing 55-67% of HF energy' diagnosis into a signed score delta."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_common import *  # noqa
import numpy as np
import torch
import torch.nn.functional as F
import lpips as lpips_pkg
import cv2

OUT = "/mnt/d/avv/r42_bonsai78/a3_diag"
pairs = pair_list("/mnt/d/avv/r36_shape/sr01/eval_png", "/mnt/d/avv/evalsplit/bonsai/eval_gt")
pairs.sort(key=lambda t: t[0])
model = lpips_pkg.LPIPS(net="vgg").to(DEV).eval()


def lp_scalar(a, b):
    x, y = model.scaling_layer(a), model.scaling_layer(b)
    o0, o1 = model.net.forward(x), model.net.forward(y)
    v = 0.0
    for kk in range(model.L):
        f0 = lpips_pkg.normalize_tensor(o0[kk]); f1 = lpips_pkg.normalize_tensor(o1[kk])
        v += float(model.lins[kk]((f0 - f1) ** 2).mean())
    return v


def unsharp(R, sigma, amt):
    a = R[0].permute(1, 2, 0).numpy()
    blur = cv2.GaussianBlur(a, (0, 0), sigma)
    out = np.clip(a + amt * (a - blur), 0, 1)
    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)


CFG = [(0.0, 0.0), (1.0, 0.3), (1.0, 0.6), (1.0, 1.0), (2.0, 0.5)]
res = {c: dict(P=[], S=[], L=[]) for c in CFG}
with torch.no_grad():
    for i, (stem, rp, gp) in enumerate(pairs):
        R0 = load(rp); G = load(gp)
        for c in CFG:
            R = R0 if c[1] == 0 else unsharp(R0, c[0], c[1])
            mse = float(((R - G) ** 2).mean())
            res[c]["P"].append(10 * np.log10(1 / mse))
            res[c]["S"].append(float(repo_ssim(R, G)))
            res[c]["L"].append(lp_scalar(R * 2 - 1, G * 2 - 1))
        print(f"[{i+1}/{len(pairs)}] {stem}", flush=True)

base = None
out = []
print()
for c in CFG:
    P = np.mean(res[c]["P"]); S = np.mean(res[c]["S"]); L = np.mean(res[c]["L"])
    sc = score_from(P, S, L)
    fs = np.array([score_from(p, s, l) for p, s, l in zip(res[c]["P"], res[c]["S"], res[c]["L"])])
    if base is None:
        base = sc
    print(f"unsharp sigma {c[0]:.1f} amount {c[1]:.2f}: PSNR {P:7.4f} SSIM {S:.4f} "
          f"LPIPS {L:.4f} SCORE {sc:.4f} (d {sc-base:+.4f})  "
          f"first8 {fs[:8].mean():.3f} last20 {fs[8:].mean():.3f}")
    out.append(dict(sigma=c[0], amount=c[1], psnr=P, ssim=S, lpips=L, score=sc,
                    dscore=sc - base, first8=float(fs[:8].mean()), last20=float(fs[8:].mean()),
                    per_frame=fs.tolist()))
json.dump(out, open(f"{OUT}/bonsai_unsharp_sweep.json", "w"), indent=1)
print(f"\nwrote {OUT}/bonsai_unsharp_sweep.json")
