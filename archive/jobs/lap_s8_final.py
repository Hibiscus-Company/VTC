#!/usr/bin/env python
"""STEP 8: remaining measurements -- 3rd member family, win=1, the CONSERVATIVE operating point
(lam=0.5, the cross-scene optimum) at k=4 PNG+JPEG, and the k-curve at that conservative setting."""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lap_run import Harness, agg
from lap_s2_sweep import name, OUT

def run(H, cfg, tag, jpeg=False):
    key = f"{tag}{'_jpg' if jpeg else '_png'}__{name(cfg)}"
    f = f"{OUT}/{key}.npy"
    if os.path.exists(f):
        return np.load(f)
    rows = H.score_cfg(cfg, jpeg=jpeg)
    np.save(f, rows)
    return rows

E = lambda lam, win: dict(nalt=1, rule="energy", kw=dict(lam=lam, win=win))
allm = ["m1", "m2", "m3", "m4"]

print("=" * 96)
print("E(cont). CROSS-FAMILY TRANSFER, third family")
print("=" * 96)
Hf = Harness(["b1", "b2", "b3", "b8"])
b = agg(run(Hf, None, "fB"))[0]; e = agg(run(Hf, E(1.0, 5), "fB"))[0]
print(f"{'gsplatB1/2/3/8':<24} {b:10.4f}  {e:10.4f}  {e-b:+9.4f}", flush=True)
del Hf; torch.cuda.empty_cache()

print("\n" + "=" * 96)
print("F. CONSERVATIVE OPERATING POINT. lam=1.25/win=3 is the HCM0181 optimum; lam=0.5 is the")
print("   HCM0421 cross-scene optimum. Measure both ends at k=4, PNG and shipped JPEG.")
print("=" * 96)
H = Harness(allm)
bp = agg(run(H, None, "k4"))[0]; bj = agg(run(H, None, "k4", jpeg=True))[0]
print(f"{'config':<28} {'PNG':>9} {'dPNG':>8} {'JPEG':>9} {'dJPEG':>8}")
print(f"{'pixelmean (baseline)':<28} {bp:9.4f} {0:+8.4f} {bj:9.4f} {0:+8.4f}")
for lam, win in [(0.5, 3), (0.5, 5), (0.75, 3), (1.0, 1), (1.0, 3), (1.25, 3), (1.5, 3)]:
    cfg = E(lam, win)
    p = agg(run(H, cfg, "k4"))[0]; j = agg(run(H, cfg, "k4", jpeg=True))[0]
    print(f"{'lam%.2f win%d' % (lam, win):<28} {p:9.4f} {p-bp:+8.4f} {j:9.4f} {j-bj:+8.4f}", flush=True)
del H; torch.cuda.empty_cache()

print("\n" + "=" * 96)
print("G. k-CURVE at the conservative setting (lam=0.5,win=3) and the aggressive one (1.25,win3)")
print("=" * 96)
print(f"{'k':>2} {'pixelmean':>10} {'d(0.5,w3)':>10} {'d(1.25,w3)':>11}")
for k in (2, 3, 4):
    Hk = Harness(allm[:k])
    b = agg(run(Hk, None, f"k{k}"))[0]
    c = agg(run(Hk, E(0.5, 3), f"k{k}"))[0]
    a = agg(run(Hk, E(1.25, 3), f"k{k}"))[0]
    print(f"{k:>2} {b:10.4f} {c-b:+10.4f} {a-b:+11.4f}", flush=True)
    del Hk; torch.cuda.empty_cache()
