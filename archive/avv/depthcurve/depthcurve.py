#!/usr/bin/env python
"""Ensemble-DEPTH curve on an existing eval split, using renders already on disk.

For each held-out eval pose we form the pixel-mean of the first k members (fixed order),
and score that k-member mean with the PROJECT scorer (repo SSIM + lpips-vgg + PSNR@50).
Also reports each member's SOLO score so the ordering can be sanity-checked.

No training, no GPU: pure CPU so it cannot disturb the two production training jobs.
"""
import argparse, os, sys, json, time
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="comma-separated render dirs, best-first")
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
    # every member must cover exactly the same stems (audit rule: no subset poisoning)
    for m in mem:
        s = {os.path.splitext(f)[0] for f in os.listdir(m)
             if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")}
        assert s == set(stems), f"{m} stem set differs from GT ({len(s)} vs {len(stems)})"

    acc = {("cum", k): [0.0, 0.0, 0.0] for k in range(1, K + 1)}
    acc.update({("solo", i): [0.0, 0.0, 0.0] for i in range(K)})

    def sc(pred, g, slot):
        r = torch.from_numpy(np.clip(pred, 0, 1)).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            acc[slot][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
            acc[slot][1] += float(repo_ssim(r, g))
            acc[slot][2] += float(lp(r * 2 - 1, g * 2 - 1).item())

    t0 = time.time()
    for n, st in enumerate(stems):
        gt = load(gt_by[st])
        g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0)
        imgs = []
        for m in mem:
            f = [x for x in os.listdir(m) if os.path.splitext(x)[0] == st][0]
            imgs.append(load(os.path.join(m, f)))
        run = np.zeros_like(imgs[0])
        for i in range(K):
            sc(imgs[i], g, ("solo", i))
            run += imgs[i]
            sc(run / (i + 1), g, ("cum", i + 1))
        print(f"[{n+1}/{len(stems)}] {st}  {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    res = {"n": N, "members": mem, "solo": {}, "cum": {}}
    for i in range(K):
        P, S, L = [v / N for v in acc[("solo", i)]]
        res["solo"][mem[i]] = dict(psnr=P, ssim=S, lpips=L,
                                   score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1)))
    for k in range(1, K + 1):
        P, S, L = [v / N for v in acc[("cum", k)]]
        res["cum"][k] = dict(psnr=P, ssim=S, lpips=L,
                             score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50, 1)))
    json.dump(res, open(args.out, "w"), indent=1)
    print("\n=== SOLO ===")
    for k, v in res["solo"].items():
        print(f"  {os.path.basename(k):24s} PSNR {v['psnr']:.4f} SSIM {v['ssim']:.4f} "
              f"LPIPS {v['lpips']:.4f} SCORE {v['score']:.4f}")
    print("=== CUMULATIVE PIXEL-MEAN DEPTH CURVE ===")
    prev = None
    for k in range(1, K + 1):
        v = res["cum"][k]
        d = "" if prev is None else f"  delta {v['score']-prev:+.4f}"
        print(f"  k={k}  PSNR {v['psnr']:.4f} SSIM {v['ssim']:.4f} LPIPS {v['lpips']:.4f} "
              f"SCORE {v['score']:.4f}{d}")
        prev = v["score"]


if __name__ == "__main__":
    main()
