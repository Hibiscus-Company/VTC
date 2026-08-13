#!/usr/bin/env python
"""How much weight should the mip3d member get in the tower ensemble?

Uniform weighting was measured DEAD for same-family members (LS-optimal weights came out
0.274/0.258/0.240/0.228, 2-fold CV -0.001). But the historical +0.23 win came from FAMILY
weighting -- and mip3d is a genuinely different family (band-limited gaussians, a different
model class), AND it is measurably BETTER solo than a normal member (+0.457 on this very scene).
A better, decorrelated member should carry MORE than its 1/N share.

Measured on the HCM0421 proxy, where both members have real GT:
   baseline member (seed42+ema999)      75.7616
   mip3d member  (seed42+ema999+mip3d)  76.2188
Sweep w_mip for the 2-member blend. w=0.5 is uniform. If the optimum sits above 0.5, the mip3d
member deserves overweighting in production, and we scale that up to the 6-normal + 1-mip3d case.
"""
import os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None

BASE = "/mnt/d/avv/evalgen/HCM0421/eval_png"
MIP = "/mnt/d/avv/mip3d/HCM0421_mip0.2/eval_png"
GT = "/mnt/d/avv/evalsplit/HCM0421/eval_gt"


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    gt_by = {os.path.splitext(f)[0]: os.path.join(GT, f) for f in os.listdir(GT)}
    stems = sorted(s for s in gt_by if os.path.exists(os.path.join(BASE, s + ".png"))
                   and os.path.exists(os.path.join(MIP, s + ".png")))
    print(f"HCM0421 proxy: {len(stems)} poses with both members + GT")

    def ld(d, s):
        return torch.from_numpy(np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                                           dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    print(f"{'w_mip':>6} {'SCORE':>9} {'LPIPS':>8} {'SSIM':>7} {'PSNR':>8}   {'vs uniform':>10}")
    out = {}
    for w in [0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        P = S = L = 0.0
        for s in stems:
            a, b = ld(BASE, s).to(dev), ld(MIP, s).to(dev)
            img = ((1 - w) * a + w * b).clamp(0, 1)
            g = torch.from_numpy(np.asarray(Image.open(gt_by[s]).convert("RGB"),
                                            dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P += 10 * np.log10(1.0 / max(((img - g) ** 2).mean().item(), 1e-12))
                S += float(repo_ssim(img, g))
                L += float(vgg(img * 2 - 1, g * 2 - 1).item())
        n = len(stems)
        P, S, L = P / n, S / n, L / n
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        out[w] = sc
        print(f"{w:>6.2f} {sc:9.4f} {L:8.4f} {S:7.4f} {P:8.4f}   {sc - out.get(0.5, sc):+10.4f}")
    best = max(out, key=out.get)
    print(f"\nbest w_mip = {best:.2f} (uniform=0.50), gain over uniform {out[best]-out[0.5]:+.4f}")
    print(f"solo baseline w=0 -> {out[0.0]:.4f} | solo mip3d w=1 -> {out[1.0]:.4f}")
    if best > 0.5:
        eff = best / (1 - best)
        print(f"=> mip3d deserves ~{eff:.2f}x a normal member's weight. In a 6-normal + 1-mip3d")
        print(f"   ensemble that is w_mip = {eff/(6+eff):.3f} instead of the uniform {1/7:.3f}.")
    else:
        print("=> no overweighting justified; ship the mip3d member at uniform 1/N.")


if __name__ == "__main__":
    main()
