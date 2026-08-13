#!/usr/bin/env python
"""End-to-end replication through the REAL production code path.

Uses the shipped gsplat_track/fit_field.py (median estimator, ds=8) to fit the field, then
applies it with the shipped apply_field vs. the proposed one-line patch (INTER_CUBIC ->
INTER_LANCZOS4 in cv2.remap), on the real production artefacts: the HCM0181 4-member
pixel-mean ensemble at real test poses, scored against real test GT, with and without the
shipped JPEG encode.  This is the check that the thing I measured is the thing the patch does.
"""
import io, os, sys
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track")
from fit_field import fit_field, apply_field as apply_shipped

Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)


def apply_patched(img, field):
    """apply_field with the single proposed change: INTER_CUBIC -> INTER_LANCZOS4 in remap."""
    H, W, _ = img.shape
    fu = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, (xx + fu[..., 0]).astype(np.float32),
                     (yy + fu[..., 1]).astype(np.float32),
                     cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    field = fit_field("/mnt/d/avv/output/HCM0181_gsplatB9ut/train_renders",
                      "/mnt/d/avv/data/phase1/public_set/HCM0181/train/images",
                      ds=8, estimator="median")
    ENS = "/mnt/d/avv/prodharness/k4/png"
    GT = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
    stems = sorted(s for s in gt_by if os.path.exists(os.path.join(ENS, s + ".png")))
    print(f"{len(stems)} real test poses")

    def jenc(x):
        b = io.BytesIO()
        Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG",
                                                                             **SHIPPED_JPEG)
        return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                          dtype=np.float32) / 255.0

    keys = ["shipped_png", "patched_png", "shipped_jpg", "patched_jpg"]
    acc = {k: [0.0, 0.0, 0.0] for k in keys}
    for s in stems:
        img = np.asarray(Image.open(os.path.join(ENS, s + ".png")).convert("RGB"),
                         dtype=np.float32) / 255.0
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(GT, gt_by[s])).convert("RGB"),
                                        dtype=np.float32) / 255.0).permute(2, 0, 1
                                                                           ).unsqueeze(0).to(dev)
        a = np.clip(apply_shipped(img, field), 0, 1)
        b = np.clip(apply_patched(img, field), 0, 1)
        for k, x in (("shipped_png", a), ("patched_png", b),
                     ("shipped_jpg", jenc(a)), ("patched_jpg", jenc(b))):
            r = torch.from_numpy(np.ascontiguousarray(np.clip(x, 0, 1))).permute(
                2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                acc[k][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[k][1] += float(repo_ssim(r, g))
                acc[k][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
    n = len(stems)
    out = {}
    print(f"{'':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    for k in keys:
        P, S, L = (x / n for x in acc[k])
        out[k] = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        print(f"{k:>12} {out[k]:9.4f} {P:8.4f} {S:8.5f} {L:8.5f}")
    print(f"\nPNG  patched - shipped = {out['patched_png']-out['shipped_png']:+.4f}")
    print(f"JPEG patched - shipped = {out['patched_jpg']-out['shipped_jpg']:+.4f}")


if __name__ == "__main__":
    main()
