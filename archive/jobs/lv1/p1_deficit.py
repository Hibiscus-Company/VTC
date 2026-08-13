#!/usr/bin/env python
"""PHASE A (diagnostic, CPU only): per-Laplacian-level energy deficit of the k-member pixel mean
against the REAL test GT, as a function of k, on the production harness scene HCM0181.

For each level l and each subset S of size k:
    deficit_l(k)  = sqrt( E(L_l(gt)) / E(L_l(mean_S)) )      "ideal global gain"
    op_l(k)       = energy-weighted mean of the shipped operator's r map at that level
                    r = sqrt(1 + k/(k-1) * mean_i E(L_l,i - L_l,mean)/E(L_l,mean))
Shared pyramids are computed ONCE per image and reused across all k -- no redundant recompute.
"""
import os, sys, json, itertools
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.dirname(HERE)
sys.path.insert(0, JOB)
from PIL import Image
from lapfuse import lap_pyr, boxf, _K
Image.MAX_IMAGE_PIXELS = None

torch.set_num_threads(4)

ROOT = "/mnt/d/avv/output"
POOL = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
        "m31b_nolpips", "e17visnorm", "gsplatB8pure", "gsplatB4warm"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
NLEV = 5
WIN = 3


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    dirs = [os.path.join(ROOT, "HCM0181_" + v, "test_poses_renders_png") for v in POOL]
    stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
    nimg = int(os.environ.get("NIMG", len(stems)))
    stems = stems[:nimg]
    print(f"n={len(stems)} images, pool={len(POOL)}", flush=True)

    # subsets: prefix of size k, plus 3 random subsets of size k (fixed seed)
    rng = np.random.RandomState(0)
    subsets = {}
    for k in range(1, len(POOL) + 1):
        ss = [tuple(range(k))]
        for _ in range(3):
            ss.append(tuple(sorted(rng.choice(len(POOL), k, replace=False))))
        subsets[k] = sorted(set(ss))

    # accumulators: sums of raw energies so ratios are computed on TOTAL energy over the set
    Egt = np.zeros(NLEV)
    Emean = {}   # (k, subset) -> per-level sum of E(L_l(mean_S))
    Odev = {}    # (k, subset) -> per-level sum of mean_i E(L_l,i - L_l,mean)
    for k, ss in subsets.items():
        for s in ss:
            Emean[(k, s)] = np.zeros(NLEV)
            Odev[(k, s)] = np.zeros(NLEV)

    K = _K
    for c, st in enumerate(stems):
        g = load(os.path.join(GTD, gt_by[st]))
        gl = lap_pyr(g, NLEV, K)[0]
        for l in range(NLEV):
            Egt[l] += float((gl[l] ** 2).sum())
        del g, gl
        mlaps = []
        for d in dirs:
            m = load(os.path.join(d, st + ".png"))
            mlaps.append(lap_pyr(m, NLEV, K)[0])
            del m
        for k, ss in subsets.items():
            for s in ss:
                for l in range(NLEV):
                    stk = torch.stack([mlaps[i][l] for i in s])       # [k,1,3,h,w]
                    mu = stk.mean(0)
                    Emean[(k, s)][l] += float((mu ** 2).sum())
                    Odev[(k, s)][l] += float(((stk - mu) ** 2).sum()) / len(s)
                    del stk, mu
        del mlaps
        if c % 10 == 0:
            print(f"  {c}/{len(stems)}", flush=True)

    out = {"n": len(stems), "pool": POOL, "Egt": Egt.tolist(), "rows": []}
    print("\nPER-LEVEL ENERGY, HCM0181 real test GT, raw renders (no field, no jpeg)")
    print(f"{'k':>2} {'subset':>22} | " + " | ".join(
        f"L{l}: {'deficit':>7} {'op_r':>6}" for l in range(NLEV)))
    for k in sorted(subsets):
        for s in subsets[k]:
            em, od = Emean[(k, s)], Odev[(k, s)]
            defi = np.sqrt(Egt / np.maximum(em, 1e-12))
            if k > 1:
                opr = np.sqrt(1.0 + (k / (k - 1.0)) * od / np.maximum(em, 1e-12))
            else:
                opr = np.full(NLEV, np.nan)
            tag = "prefix" if s == tuple(range(k)) else ",".join(map(str, s))
            out["rows"].append(dict(k=k, subset=[int(x) for x in s], deficit=defi.tolist(),
                                    op_r=opr.tolist(), Emean=em.tolist()))
            print(f"{k:>2} {tag:>22} | " + " | ".join(
                f"L{l}: {defi[l]:7.4f} {opr[l]:6.4f}" for l in range(NLEV)))
    json.dump(out, open(os.path.join(HERE, "p1_deficit.json"), "w"))

    # compact summary: mean over subsets of each size
    print("\nSUMMARY (mean over subsets of each size)")
    print(f"{'k':>2} | " + " | ".join(f"L{l} def / op" for l in range(NLEV)))
    for k in sorted(subsets):
        rs = [r for r in out["rows"] if r["k"] == k]
        d = np.mean([r["deficit"] for r in rs], 0)
        o = np.mean([r["op_r"] for r in rs], 0)
        print(f"{k:>2} | " + " | ".join(f"{d[l]:6.4f} / {o[l]:6.4f}" for l in range(NLEV)))


if __name__ == "__main__":
    main()
