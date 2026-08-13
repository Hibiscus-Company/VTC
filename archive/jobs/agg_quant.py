"""RECONCILIATION + quantisation check.

The brief's sanctioned k=4 PNG baseline is 76.6745, but agg_run.py's float32 mean reads
76.6487 (the JPEG number, 76.6937, matches the brief EXACTLY). Hypothesis: the shipped
pipeline writes uint8, and 8-bit quantisation of the ensemble mean is itself worth
~+0.026 -- the same "structured noise is load-bearing" mechanism as the JPEG bonus.

This scores, on the same 60 real test poses:
  1. the pre-built production ensemble at /mnt/d/avv/prodharness/k4/png   (ground truth ref)
  2. our float32 mean of the same 4 members                              (agg_run regime)
  3. that mean rounded to uint8                                          (shipped regime)
  4. the k=7 winner winsor1 in float and uint8   -- does the delta survive quantisation?
"""
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_lib as A
import agg_core as C
from agg_run import POOLS

PRE = "/mnt/d/avv/prodharness/k4/png"


def q8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8).astype(np.float32) / 255.0


def main():
    dev = A.init()
    p4, p7 = POOLS["p4"], POOLS["p7"]
    stems, gt_by = A.stems_for(p4)
    sc = A.Scorer(dev)
    for i, s in enumerate(stems):
        g = A.gt_tensor(gt_by[s], dev)
        pre = os.path.join(PRE, s + ".png")
        if os.path.exists(pre):
            sc.add("prebuilt k4/png", A.load(PRE, s), g)
        X4 = np.stack([A.load(A.mdir(v), s) for v in p4], 0)
        m4 = C.agg_mean(X4)
        sc.add("p4 mean float32", m4, g)
        sc.add("p4 mean uint8", q8(m4), g)
        del X4
        X7 = np.stack([A.load(A.mdir(v), s) for v in p7], 0)
        m7, w7 = C.agg_mean(X7), C.agg_winsor(X7, 1)
        sc.add("p7 mean float32", m7, g)
        sc.add("p7 mean uint8", q8(m7), g)
        sc.add("p7 winsor1 float32", w7, g)
        sc.add("p7 winsor1 uint8", q8(w7), g)
        del X7, g
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(stems)}", flush=True)
    print(f"\n{'variant':<22}{'SCORE':>9}{'PSNR':>9}{'SSIM':>8}{'LPIPS':>9}{'n':>5}")
    tab = {r[0]: r for r in sc.table()}
    for k in ["prebuilt k4/png", "p4 mean float32", "p4 mean uint8",
              "p7 mean float32", "p7 mean uint8", "p7 winsor1 float32", "p7 winsor1 uint8"]:
        if k not in tab:
            continue
        _, s, P, S, L, n = tab[k]
        print(f"{k:<22}{s:9.4f}{P:9.4f}{S:8.4f}{L:9.4f}{n:5d}")
    if "p7 mean uint8" in tab and "p7 winsor1 uint8" in tab:
        print(f"\nwinsor1 - mean  in float32 : {tab['p7 winsor1 float32'][1]-tab['p7 mean float32'][1]:+.4f}")
        print(f"winsor1 - mean  in uint8   : {tab['p7 winsor1 uint8'][1]-tab['p7 mean uint8'][1]:+.4f}")


if __name__ == "__main__":
    main()
