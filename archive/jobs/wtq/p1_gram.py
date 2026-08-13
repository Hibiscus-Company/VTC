#!/usr/bin/env python
"""PASS 1 (CPU only, no GPU): exact least-squares machinery for ANY member weighting.

For each test pose on the production harness (HCM0181, real test GT), load every member render,
apply the shipped lens-field warp (lanczos4), and accumulate -- in the numerically stable
DIFFERENCE basis D_i = x_i - x_0 -- the per-image quantities

    Gd[i,j] = D_i . D_j        bd[i] = D_i . (g - x_0)        s = |g - x_0|^2

From these the MSE of ANY simplex weighting w (sum w = 1) is EXACT and free:

    P * MSE(w) = w' Gd w - 2 w' bd + s

so the LS-optimal weights, the 2-fold CV weights, every 2-family blend optimum and every member's
solo PSNR all come out of a single cheap pass with no re-rendering and no GPU.  Pass 2 then takes
only the handful of weightings that matter through the full shipped chain (energy restore -> field
-> JPEG) where SSIM and LPIPS also get a vote.
"""
import os, sys, time
import numpy as np
import cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, os.path.join(HERE, "lens"))
from fieldlib import LooPool, upsample

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(3)

TAG = "HCM0181"
GTD = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
OUT = "/mnt/d/avv/output/HCM0181_%s/test_poses_renders_png"

MEMBERS = [
    ("gsplatB9ut",       "UT"),
    ("gsplatB10ut8M",    "UT"),
    ("gsplatB11ut60k",   "UT"),
    ("gsplatB12ut8Ms7",  "UT"),
    ("gsplatB8pure",     "PURE"),
    ("gsplatB6bilagrid", "APP"),
    ("gsplatB7ppisp2",   "APP"),
    ("gsplatB5affine",   "APP"),
    ("e15ceil95",        "E"),
    ("e16app",           "E"),
    ("e17visnorm",       "E"),
    ("m31b_taillpips",   "M31"),
]


def main():
    n_lim = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(OUT % m, s + ".png")) for m, _ in MEMBERS))
    stems = stems[:n_lim]
    M = len(MEMBERS)
    z = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in z["HW"]]
    lens = upsample(LooPool(z["s8"]).pooled("median"), H, W, "cubic")
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    MX = (xx + lens[..., 0]).astype(np.float32)
    MY = (yy + lens[..., 1]).astype(np.float32)
    P = H * W * 3
    print(f"{TAG}: {len(stems)} poses, {M} members, {H}x{W}", flush=True)

    Gd = np.zeros((len(stems), M, M))
    bd = np.zeros((len(stems), M))
    ss = np.zeros(len(stems))
    D = np.empty((M, P), dtype=np.float32)
    t0 = time.time()
    for n, s in enumerate(stems):
        for i, (m, _) in enumerate(MEMBERS):
            a = np.asarray(Image.open(os.path.join(OUT % m, s + ".png")).convert("RGB"),
                           dtype=np.float32) / 255.0
            a = cv2.remap(a, MX, MY, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            D[i] = np.clip(a, 0, 1).reshape(-1)
        g = (np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"),
                        dtype=np.float32) / 255.0).reshape(-1)
        x0 = D[0].copy()
        D -= x0                      # difference basis: well conditioned, tiny magnitudes
        gd = g - x0
        Gd[n] = (D @ D.T).astype(np.float64)
        bd[n] = (D @ gd).astype(np.float64)
        ss[n] = float(gd @ gd)
        if n % 10 == 0:
            print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gram.npz"),
             Gd=Gd, bd=bd, ss=ss, P=P, stems=np.array(stems),
             names=np.array([m for m, _ in MEMBERS]), fam=np.array([f for _, f in MEMBERS]))
    print(f"done {time.time()-t0:.0f}s -> gram.npz")


if __name__ == "__main__":
    main()
