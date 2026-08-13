#!/usr/bin/env python
"""STEP 0: prove the harness+pyramid plumbing. (a) the Laplacian pyramid analysis/synthesis is
exact, (b) mean-at-every-level reproduces the pixel mean to float precision, (c) the pixel-mean
baseline reproduces the already-published prodharness numbers."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lap_run import Harness, agg
from lapfuse import lap_pyr, lap_recon, fuse_image, _K

H = Harness(["m1", "m2", "m3", "m4"], nlev=5)
k = _K.to("cuda")
st = H.stack(0)
laps, res, sizes = lap_pyr(st, 5, k)
rec = lap_recon(laps, res, sizes, k)
print("(a) pyramid roundtrip maxabs err :", (rec - st).abs().max().item())
print("    level shapes:", [tuple(l.shape[-2:]) for l in laps], "res", tuple(res.shape[-2:]))
print("    per-level rms energy (member0):", [f"{l[0].pow(2).mean().sqrt().item():.5f}" for l in laps])

cfg_mean = dict(nalt=5, rule="mean")
f = fuse_image(st, cfg_mean, 5, k)
print("(b) mean-rule vs pixel-mean maxabs:", (f - st.mean(0, keepdim=True)).abs().max().item())

rows = H.score_cfg(None, jpeg=False)
s, P, S, L = agg(rows)
print(f"(c) PIXEL-MEAN k4 uint8 PNG : SCORE {s:.4f}  PSNR {P:.4f} SSIM {S:.4f} LPIPS {L:.4f}")
print("    published prodharness PNG :        76.6745        24.9034      0.8650       0.1055")
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/lap_base_png.npy", rows)
rowsj = H.score_cfg(None, jpeg=True)
sj, Pj, Sj, Lj = agg(rowsj)
print(f"(c) PIXEL-MEAN k4 JPEGq100ss2: SCORE {sj:.4f}  PSNR {Pj:.4f} SSIM {Sj:.4f} LPIPS {Lj:.4f}")
print("    published prodharness JPEG:        76.6937        24.8933      0.8637       0.1038")
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/lap_base_jpg.npy", rowsj)
