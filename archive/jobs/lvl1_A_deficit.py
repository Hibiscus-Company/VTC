#!/usr/bin/env python
"""PHASE A -- per-level Laplacian ENERGY DEFICIT of the pixel mean, as a function of k.

The shipped energy operator only touches level 0 because levels >=1 once measured "already at GT
energy".  That was at k=4; production ships k=7-8, and deeper averaging destroys more everywhere.

For each Laplacian level l and each pool size k we report, summed over all pixels and images:
    D_l(k)  = E(L_mean) / E(L_gt)                      <- the true deficit (want < 1)
    R_l(k)  = (E(L_mean) + V_l) / E(L_gt)              <- what the operator's own correction predicts
              V_l = k/(k-1) * mean_i E(L_i - L_mean)   (the disagreement term, GLOBAL version)
If D_1(k) falls with k there is a level-1 deficit to chase; if R_1 lands near 1 the operator's
own statistic is the right size for it; if R_1 >> 1 a lam1 of 1.0 would overshoot.

HOISTED: lap_pyr is LINEAR, so pyr(mean of S) = mean over S of pyr(member).  Each member's pyramid
is built ONCE per image and every k / every subset is then free.  CPU only.
"""
import itertools, json, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
from lapfuse import lap_pyr, _K

Image.MAX_IMAGE_PIXELS = None
TAG = "HCM0181"
D = lambda m: f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png"
UT4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
DIV4 = ["gsplatB8pure", "e17visnorm", "m31b_taillpips", "gsplatB6bilagrid"]
POOL8 = UT4 + DIV4
NLEV = 4


def ld(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def subsets(n, k, rng, cap=6):
    all_s = list(itertools.combinations(range(n), k))
    if len(all_s) <= cap:
        return all_s
    idx = rng.choice(len(all_s), cap, replace=False)
    return [all_s[i] for i in idx]


def main():
    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(D(m), s + ".png")) for m in POOL8))
    rng = np.random.default_rng(0)
    SUBS = {k: subsets(8, k, rng) for k in range(1, 9)}
    UTSUBS = {k: subsets(4, k, np.random.default_rng(1)) for k in range(1, 5)}

    # accumulators: [scheme][k][level] -> (E_mean, V, E_gt)
    acc = {sch: {k: np.zeros((NLEV, 3)) for k in ks}
           for sch, ks in (("all8", SUBS), ("ut4", UTSUBS))}
    t0 = time.time()
    for n, s in enumerate(stems):
        gt = ld(os.path.join(gtd, gt_by[s]))
        gl = lap_pyr(gt, NLEV, _K)[0]
        Egt = [float((L ** 2).sum()) for L in gl]
        mem = [ld(os.path.join(D(m), s + ".png")) for m in POOL8]
        pyr = [lap_pyr(m, NLEV, _K)[0] for m in mem]         # HOISTED: 8 pyramids, reused by all k
        for sch, subs, pool_idx in (("all8", SUBS, list(range(8))),
                                    ("ut4", UTSUBS, list(range(4)))):
            for k, ss in subs.items():
                for S in ss:
                    idx = [pool_idx[i] for i in S]
                    for l in range(NLEV):
                        Lm = torch.stack([pyr[i][l] for i in idx]).mean(0)
                        Em = float((Lm ** 2).sum())
                        if k > 1:
                            V = float(np.mean([float(((pyr[i][l] - Lm) ** 2).sum())
                                               for i in idx])) * (k / (k - 1.0))
                        else:
                            V = 0.0
                        acc[sch][k][l] += np.array([Em, V, Egt[l]]) / len(ss)
        if n % 10 == 0:
            print(f"  {n}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    out = {}
    for sch in ("all8", "ut4"):
        print(f"\n=== {sch} pool, {TAG}, n={len(stems)} images, "
              f"energy summed over all pixels ===")
        hdr = "  k " + "".join(f"   D_L{l}    R_L{l} " for l in range(NLEV))
        print(hdr)
        for k in sorted(acc[sch]):
            row = acc[sch][k]
            cells = ""
            rec = []
            for l in range(NLEV):
                Em, V, Eg = row[l]
                Dl, Rl = Em / Eg, (Em + V) / Eg
                cells += f" {Dl:7.4f} {Rl:8.4f}"
                rec.append((Dl, Rl))
            print(f" {k:2d}{cells}")
            out[f"{sch}_k{k}"] = rec
    json.dump(out, open(f"{HERE}/lvl1_A_deficit.json", "w"), indent=1)
    print(f"\nD_l = E(mean)/E(gt);  R_l = (E(mean)+disagreement)/E(gt).  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
