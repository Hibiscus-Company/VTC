#!/usr/bin/env python
"""Find the cheapest LEGITIMATE encode. The measured JPEG tax is -0.286 chair / -0.924 bonsai,
concentrated in LPIPS. At q100 the quant table is all-1s, so the remaining loss is (a) the
RGB->YCbCr->RGB roundtrip and (b) DCT coefficient rounding. Pillow >=9.5 exposes keep_rgb=True,
which stores JPEG in RGB and removes (a) entirely -- a REAL jpeg any decoder reads, no format
spoofing. Test whether that recovers the tax before considering PNG-bytes-under-.jpg.
"""
import io, os, sys, argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.render_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir)}
    srcs = {s: Image.open(os.path.join(args.render_dir, s + ".png")).convert("RGB") for s in stems}
    gts = {s: np.asarray(Image.open(os.path.join(args.gt_dir, gt_by[s])).convert("RGB"),
                         dtype=np.float32) / 255.0 for s in stems}

    variants = [
        ("PNG (lossless upper bound)", None),
        ("q100 ss2 prog  [SHIPPED]", dict(quality=100, subsampling=2, optimize=True, progressive=True)),
        ("q100 ss0 prog", dict(quality=100, subsampling=0, optimize=True, progressive=True)),
        ("q100 ss0 prog keep_rgb", dict(quality=100, subsampling=0, optimize=True, progressive=True, keep_rgb=True)),
        ("q100 ss0 base keep_rgb", dict(quality=100, subsampling=0, optimize=True, keep_rgb=True)),
        ("q98  ss0 prog keep_rgb", dict(quality=98, subsampling=0, optimize=True, progressive=True, keep_rgb=True)),
    ]

    for name, kw in variants:
        P = S = L = 0.0
        nbytes = 0
        for s in stems:
            if kw is None:
                buf = io.BytesIO(); srcs[s].save(buf, "PNG", optimize=True)
            else:
                buf = io.BytesIO(); srcs[s].save(buf, "JPEG", **kw)
            nbytes += buf.tell()
            buf.seek(0)
            dec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
            r = torch.from_numpy(dec).permute(2, 0, 1).unsqueeze(0).to(dev)
            g = torch.from_numpy(gts[s]).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                S += float(repo_ssim(r, g))
                L += float(vgg(r * 2 - 1, g * 2 - 1).item())
        n = len(stems)
        P, S, L = P / n, S / n, L / n
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        print(f"{args.tag:8s} {name:28s} SCORE {sc:8.4f}  LPIPS {L:.4f} SSIM {S:.4f} "
              f"PSNR {P:7.4f}  {nbytes/1e6:7.1f}MB")


if __name__ == "__main__":
    main()
