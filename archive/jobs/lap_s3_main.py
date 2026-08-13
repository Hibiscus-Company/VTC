#!/usr/bin/env python
"""STEP 3: the decisive measurements.
  A  k-curve (k=2,3,4)  -- the SIGNATURE test. Cleanup-class tricks DECAY with ensemble depth and
     invert in production; a genuine texture-restoration should be flat or GROW.
  B  post-JPEG (q100 ss2, the shipped encode) for every candidate.
  C  lam / win / rmax fine sweep with 2-fold VIEW cross-validation.
  D  controls: does the gain need the real ensemble-disagreement map, or is any adaptive
     sharpening map as good?
  E  cross-family transfer: same fixed (lam,win) on two DIFFERENT member families.
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lap_run import Harness, agg
from lap_s2_sweep import name, OUT

torch.manual_seed(0)
os.makedirs(OUT, exist_ok=True)
LOG = {}


def run(H, cfg, tag, jpeg=False, quiet=False):
    key = f"{tag}{'_jpg' if jpeg else '_png'}__{name(cfg)}"
    f = f"{OUT}/{key}.npy"
    if os.path.exists(f):
        rows = np.load(f)
    else:
        t0 = time.time()
        rows = H.score_cfg(cfg, jpeg=jpeg)
        np.save(f, rows)
    LOG[key] = rows
    return rows


def line(nm, rows, base):
    s, P, S, L = agg(rows)
    b = agg(base)[0]
    print(f"{nm:<38} {s:9.4f} {s-b:+8.4f} {P:8.4f} {S:7.4f} {L:8.4f}", flush=True)
    return s - b


HDR = f"{'config':<38} {'SCORE':>9} {'dScore':>8} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8}"
ENERGY = dict(nalt=1, rule="energy", kw=dict(lam=1.0, win=5))
PNORM = dict(nalt=1, rule="pnorm", kw=dict(p=2.0, win=3))

# ============================================================ A. k-curve  (SIGNATURE TEST)
print("\n" + "=" * 96)
print("A. k-CURVE -- does the gain decay with ensemble depth (cleanup-class => DANGER) or hold/grow?")
print("=" * 96)
print(f"{'k':>2}  {'pixel-mean':>10}  {'energy n1':>10}  {'d(energy)':>10}  {'pnorm n1':>10}  {'d(pnorm)':>10}")
allm = ["m1", "m2", "m3", "m4"]
for k in (2, 3, 4):
    H = Harness(allm[:k])
    b = run(H, None, f"k{k}")
    e = run(H, ENERGY, f"k{k}")
    p = run(H, PNORM, f"k{k}")
    sb, se, sp = agg(b)[0], agg(e)[0], agg(p)[0]
    print(f"{k:>2}  {sb:10.4f}  {se:10.4f}  {se-sb:+10.4f}  {sp:10.4f}  {sp-sb:+10.4f}", flush=True)
    del H
    torch.cuda.empty_cache()

# ============================================================ B/C/D/E on k=4
H = Harness(allm)
base_png = run(H, None, "k4")
base_jpg = run(H, None, "k4", jpeg=True)

print("\n" + "=" * 96)
print("B. POST-JPEG (q100 ss2 prog -- the SHIPPED encode). production writes JPEG, so this is the")
print("   number that actually ships.")
print("=" * 96)
print(HDR)
line("pixelmean  [PNG]", base_png, base_png)
line("pixelmean  [JPEG shipped]", base_jpg, base_png)
for cfg in [ENERGY, PNORM,
            dict(nalt=1, rule="energy", kw=dict(lam=1.5, win=5)),
            dict(nalt=1, rule="gain", kw=dict(g=1.02))]:
    line(name(cfg) + "  [PNG]", run(H, cfg, "k4"), base_png)
    line(name(cfg) + "  [JPEG]", run(H, cfg, "k4", jpeg=True), base_jpg)

# ============================================================ C. lam/win/rmax sweep
print("\n" + "=" * 96)
print("C. PARAMETER SWEEP (PNG). fitted params = (lam, win, rmax); cross-validated below.")
print("=" * 96)
print(HDR)
grid = []
for lam in (0.75, 1.0, 1.25, 1.5):
    for win in (3, 5, 9, 17):
        grid.append(dict(nalt=1, rule="energy", kw=dict(lam=lam, win=win)))
for rmax in (1.5, 2.0, 8.0):
    grid.append(dict(nalt=1, rule="energy", kw=dict(lam=1.0, win=5, rmax=rmax)))
for cfg in grid:
    line(name(cfg), run(H, cfg, "k4"), base_png)

# ============================================================ D. controls
print("\n" + "=" * 96)
print("D. CONTROLS -- is the real ensemble-disagreement map load-bearing?")
print("   'const'   = same per-image AVERAGE boost, no spatial map")
print("   'shuffle' = same boost HISTOGRAM, spatially scrambled")
print("   'blurNNN' = boost map smoothed over NNN px")
print("=" * 96)
print(HDR)
for m in ("raw", "const", "shuffle", "blur33", "blur129"):
    cfg = dict(nalt=1, rule="energy", kw=dict(lam=1.0, win=5, map=m))
    line(name(cfg), run(H, cfg, "k4"), base_png)

# ============================================================ E. cross-family transfer
print("\n" + "=" * 96)
print("E. CROSS-FAMILY TRANSFER -- fixed (lam=1.0,win=5,n=1) on member families it was NOT tuned on")
print("=" * 96)
print(f"{'family':<24} {'pixel-mean':>10}  {'energy':>10}  {'delta':>9}")
del H
torch.cuda.empty_cache()
FAM = {"UT prod (tuned on)": allm,
       "sh0-sh3": ["s0", "s1", "s2", "s3"],
       "gsplatB1/2/3/8": ["b1", "b2", "b3", "b8"]}
for fname, mem in FAM.items():
    Hf = Harness(mem)
    b = run(Hf, None, "f_" + fname.split()[0])
    e = run(Hf, ENERGY, "f_" + fname.split()[0])
    sb, se = agg(b)[0], agg(e)[0]
    print(f"{fname:<24} {sb:10.4f}  {se:10.4f}  {se-sb:+9.4f}", flush=True)
    del Hf
    torch.cuda.empty_cache()

np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/lap_log_keys.npy", np.array(sorted(LOG)))
print("\nrows cached to", OUT)
