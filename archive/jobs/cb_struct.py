#!/usr/bin/env python
"""COMBINER LENS -- step 0: member family structure at HCM0181 (production harness).
CPU only, no GPU. Computes per-variant solo PSNR vs REAL test GT and the pairwise
RMS distance matrix, then agglomerates into families. Cheap: 6 views, half-res.
"""
import os, sys, json, time
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

OUT = "/mnt/d/avv/output"
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
VARS = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
        "gsplatB1", "gsplatB2", "gsplatB3", "gsplatB4warm", "gsplatB5affine",
        "gsplatB7ppisp2", "gsplatB8pure",
        "e15ceil95", "e16app", "e17visnorm", "m31b_nolpips", "m31b_taillpips",
        "sh0", "sh1", "sh2", "sh3"]


def mdir(v):
    return os.path.join(OUT, "HCM0181_" + v, "test_poses_renders_png")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    gt_by = {os.path.splitext(f)[0]: os.path.join(GTD, f) for f in os.listdir(GTD)}
    have = [v for v in VARS if os.path.isdir(mdir(v))]
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(mdir(v), s + ".png")) for v in have))
    step = max(1, len(stems) // n)
    stems = stems[::step][:n]
    print(f"variants={len(have)} views={len(stems)}", flush=True)
    k = len(have)
    D = np.zeros((k, k)); mse = np.zeros(k); t0 = time.time()
    for s in stems:
        g = np.asarray(Image.open(gt_by[s]).convert("RGB"), np.float32) / 255.
        X = np.stack([np.asarray(Image.open(os.path.join(mdir(v), s + ".png")).convert("RGB"),
                                 np.float32) / 255. for v in have], 0)
        for i in range(k):
            mse[i] += ((X[i] - g) ** 2).mean()
            for j in range(i + 1, k):
                d = ((X[i] - X[j]) ** 2).mean()
                D[i, j] += d; D[j, i] += d
        del X
        print(f"  {s} {time.time()-t0:.0f}s", flush=True)
    D /= len(stems); mse /= len(stems)
    psnr = 10 * np.log10(1.0 / mse)
    rms = np.sqrt(D) * 255.0
    order = np.argsort(-psnr)
    print("\nsolo PSNR (best first):")
    for i in order:
        print(f"  {have[i]:<18} {psnr[i]:7.3f}")
    print("\npairwise RMS distance (uint8 levels):")
    print("            " + "".join(f"{have[j][:7]:>8}" for j in order))
    for i in order:
        print(f"{have[i][:11]:<12}" + "".join(f"{rms[i,j]:8.2f}" for j in order))
    # nearest neighbour of each variant
    print("\nnearest neighbours:")
    for i in order:
        nb = np.argsort(rms[i] + np.eye(k)[i] * 1e9)[:3]
        print(f"  {have[i]:<18} -> " + ", ".join(f"{have[j]}({rms[i,j]:.2f})" for j in nb))
    json.dump(dict(variants=have, psnr=psnr.tolist(), rms=rms.tolist(),
                   stems=stems), open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/cb_struct.json", "w"))


if __name__ == "__main__":
    main()
