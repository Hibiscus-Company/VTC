#!/usr/bin/env python
"""Ensemble-depth curve measured through the FULL SHIPPED CHAIN.

The float depth curve ignores two steps production actually performs: uint8 rounding and the
shipped JPEG encode (q100, subsampling=2, optimize, progressive). Both inject structured
high-frequency content, and the r23 post-mortem established that structured HF partially
compensates an over-smoothed pixel-mean. So the depth penalty must be re-measured through the
real chain before it can be acted on.
"""
import argparse, os, sys, io, json, time
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg

SHIP = dict(quality=100, subsampling=2, optimize=True, progressive=True)


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    mem = args.members.split(",")
    K = len(mem)
    lp = lpips_pkg.LPIPS(net="vgg").eval()
    gt_by = {os.path.splitext(f)[0]: os.path.join(args.gt, f) for f in os.listdir(args.gt)}
    stems = sorted(gt_by)
    acc = {k: [0.0, 0.0, 0.0] for k in range(1, K + 1)}
    mb = {k: 0 for k in range(1, K + 1)}
    t0 = time.time()
    for n, st in enumerate(stems):
        g = torch.from_numpy(load(gt_by[st])).permute(2, 0, 1).unsqueeze(0)
        run = None
        for i, m in enumerate(mem):
            f = [x for x in os.listdir(m) if os.path.splitext(x)[0] == st][0]
            a = load(os.path.join(m, f))
            run = a if run is None else run + a
            u8 = np.clip(run / (i + 1) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(u8).save(buf, format="JPEG", **SHIP)
            mb[i + 1] += buf.getbuffer().nbytes
            buf.seek(0)
            r = torch.from_numpy(np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
                                 ).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                acc[i + 1][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[i + 1][1] += float(repo_ssim(r, g))
                acc[i + 1][2] += float(lp(r * 2 - 1, g * 2 - 1).item())
        print(f"[{n+1}/{len(stems)}] {st} {time.time()-t0:.0f}s", flush=True)
    N = len(stems)
    res = {}
    for k in range(1, K + 1):
        P, S, L = [v / N for v in acc[k]]
        res[k] = dict(psnr=P, ssim=S, lpips=L, mb=mb[k] / 1e6,
                      score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1)))
    json.dump({"n": N, "members": mem, "res": res}, open(args.out, "w"), indent=1)
    print("\n=== DEPTH CURVE THROUGH SHIPPED CHAIN (uint8 + q100 ss2 prog JPEG) ===")
    prev = None
    for k in range(1, K + 1):
        v = res[k]
        d = "" if prev is None else f"  delta {v['score']-prev:+.4f}"
        print(f"  k={k}  PSNR {v['psnr']:.4f} SSIM {v['ssim']:.4f} LPIPS {v['lpips']:.4f} "
              f"SCORE {v['score']:.4f}{d}   {v['mb']:.1f}MB")
        prev = v["score"]


if __name__ == "__main__":
    main()
