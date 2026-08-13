#!/usr/bin/env python
"""PRODUCTION-REGIME HARNESS -- the thing we should have been validating on all along.

THE PROBLEM IT FIXES: our eval-split proxy trains on 180/240 photos and scores at synthetic holes.
It is DATA-STARVED, so proxy renders are noisier than production renders, and every cleanup-class
intervention measures too well there. That bias cost us r23 (JPEG encode: +0.158 predicted,
-0.026 delivered), inverted EMA by 170x, and is why the restoration head is benched.

THE FIX, which was on disk the whole time: /mnt/d/avv/data/phase1/public_set/*/test/images holds
290 REAL test-GT images over 5 drone-tower scenes whose models were trained on 100% of their train
photos, and /mnt/d/avv/output/HCM0181_*/test_poses_renders_png holds 21 render variants at those
REAL test poses. Full training density + real test poses + real test GT = production regime by
construction. Different scenes from the graded private_set2, so zero leakage. EXPERIMENTS.md:7
already designates this as the sanctioned local-scoring GT.

STEP 1 (this script): CALIBRATE. Score single members and k=1..4 pixel-mean ensembles. The ensemble
k-curve is the calibration signal: our LB-observed tower marginal for 3->4 seeds was +0.142. If this
harness reproduces that within +/-0.10 it is calibrated to production and every post-processing
operator should be read here instead of on the eval split.
"""
import os, sys, itertools
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None

GT = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
MEMBERS = [
    "/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
    "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
    "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
    "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png",
]


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    gt_by = {os.path.splitext(f)[0]: os.path.join(GT, f) for f in os.listdir(GT)}
    avail = [d for d in MEMBERS if os.path.isdir(d)]
    print(f"members available: {len(avail)}/{len(MEMBERS)}")
    if not avail:
        sys.exit("no member render dirs found")
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in avail))
    print(f"REAL test poses with GT and all members: {len(stems)}\n")

    def ld(d, s):
        return np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                          dtype=np.float32) / 255.0

    def sc(dirs):
        P = S = L = 0.0
        for s in stems:
            img = np.mean([ld(d, s) for d in dirs], axis=0)
            r = torch.from_numpy(np.clip(img, 0, 1)).permute(2, 0, 1).unsqueeze(0).to(dev)
            g = torch.from_numpy(np.asarray(Image.open(gt_by[s]).convert("RGB"),
                                            dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                S += float(repo_ssim(r, g))
                L += float(vgg(r * 2 - 1, g * 2 - 1).item())
        n = len(stems)
        P, S, L = P / n, S / n, L / n
        return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)), P, S, L

    print(f"{'k':>2} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8}  {'marginal':>9}")
    prev = None
    for k in range(1, len(avail) + 1):
        s, P, S, L = sc(avail[:k])
        m = "" if prev is None else f"{s - prev:+9.4f}"
        print(f"{k:>2} {s:9.4f} {P:8.4f} {S:7.4f} {L:8.4f}  {m:>9}")
        if k == 3:
            k3 = s
        if k == 4:
            k4 = s
        prev = s
    print()
    if len(avail) >= 4:
        marg = k4 - k3
        print(f"CALIBRATION: 3->4 member marginal here = {marg:+.4f}")
        print(f"             LB-observed tower 3->4 seeds  = +0.1420")
        d = abs(marg - 0.142)
        print(f"             |deviation| = {d:.4f}  -> "
              f"{'CALIBRATED: read every post-processing operator here' if d <= 0.10 else f'OFFSET {marg-0.142:+.3f}: usable with a fixed haircut'}")


if __name__ == "__main__":
    main()
