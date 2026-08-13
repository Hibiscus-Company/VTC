#!/usr/bin/env python
"""B2 design probe, part 2 -- CPU ONLY.

probe 1 found the pair-stratified (block=2, keep 1) gate arm separates the sharp and
blurry subsets by only +0.345 nats (0.41 sd), because sharpness is autocorrelated at
lag 10 so adjacent frames are nearly as blurry as each other.  A weak dose makes a null
result uninformative.  Search the (block size B, keep m) grid for the design that
maximises the sharpness separation at a fixed view count and a fixed, MATCHED coverage
pattern (every block contributes exactly m frames to every arm).
"""
import os, sys, json, csv
import numpy as np

HERE = "/mnt/d/avv/r42_bonsai78/b2_train"
A2 = "/mnt/d/avv/r42_bonsai78/a2_blurpred"
ES = "/mnt/d/avv/evalsplit/bonsai"

rows = list(csv.DictReader(open(f"{A2}/train248_sharp.csv")))
S = {int(r["frame"]): {k: float(v) for k, v in r.items() if k != "name"} for r in rows}
EVAL = set(json.load(open(f"{ES}/split.json"))["eval_frames"])
SUB = sorted(f for f in S if f not in EVAL)
llv = np.array([S[f]["log_lapvar"] for f in SUB])
SD = llv.std()


def blocks(B):
    return [SUB[i:i + B] for i in range(0, len(SUB), B) if len(SUB[i:i + B]) == B]


def pick(B, m, how):
    out = []
    for blk in blocks(B):
        v = np.array([S[f]["log_lapvar"] for f in blk])
        o = np.argsort(-v)
        idx = o[:m] if how == "sharp" else o[-m:]
        out += [blk[i] for i in sorted(idx)]
    return sorted(out)


def gap(fs):
    g = np.diff(np.sort(fs))
    return int(g.max()), float(g.mean())


print(f"{'B':>2} {'m':>2} {'n':>4} {'maxgap':>7} {'meangap':>8} {'d_llv(nats)':>12} "
      f"{'sd units':>9} {'sharp mean':>11} {'blurry mean':>12}")
best = None
for B in (2, 3, 4, 5, 6, 8):
    for m in range(1, B):
        sh, bl = pick(B, m, "sharp"), pick(B, m, "blurry")
        if len(sh) < 50:
            continue
        a = np.mean([S[f]["log_lapvar"] for f in sh])
        b = np.mean([S[f]["log_lapvar"] for f in bl])
        mg, ag = gap(sh)
        mg2, _ = gap(bl)
        print(f"{B:2d} {m:2d} {len(sh):4d} {max(mg,mg2):7d} {ag:8.1f} {a-b:12.4f} "
              f"{(a-b)/SD:9.2f} {a:11.4f} {b:12.4f}")
        cand = (a - b, len(sh), B, m)
        if len(sh) >= 100 and (best is None or cand[0] > best[0]):
            best = cand
print(f"\nbest at n>=100: d_llv {best[0]:+.4f} nats ({best[0]/SD:+.2f} sd) at B={best[2]} m={best[3]}, n={best[1]}")

# emit the chosen design (B=4, m=2 -> n=110, same count as the pair design)
rng = np.random.RandomState(42)
for B, m, tag in ((4, 2, "b4m2"), (6, 3, "b6m3")):
    for how in ("sharp", "blurry"):
        sel = pick(B, m, how)
        with open(f"{HERE}/subsets/{tag}_{how}.txt", "w") as fh:
            for f in sel:
                fh.write(f"frame_{f:06d}.jpg\n")
        v = np.array([S[f]["log_lapvar"] for f in sel])
        mg, ag = gap(sel)
        print(f"wrote {tag}_{how}.txt  n={len(sel)} maxgap {mg} meangap {ag:.1f} "
              f"llv mean {v.mean():+.4f} sd {v.std():.4f}")
    sel = []
    for blk in blocks(B):
        sel += sorted(rng.choice(blk, m, replace=False).tolist())
    sel = sorted(int(x) for x in sel)
    with open(f"{HERE}/subsets/{tag}_rand.txt", "w") as fh:
        for f in sel:
            fh.write(f"frame_{f:06d}.jpg\n")
    v = np.array([S[f]["log_lapvar"] for f in sel])
    mg, ag = gap(sel)
    print(f"wrote {tag}_rand.txt    n={len(sel)} maxgap {mg} meangap {ag:.1f} "
          f"llv mean {v.mean():+.4f} sd {v.std():.4f}")

# --------------------------------------------------------------- weight forms, z-score
print("\nexponential-tilt weights  w = exp(beta*z),  z = (llv - mean)/sd,  renorm mean 1")
z = (llv - llv.mean()) / SD
ess = lambda w: float(w.sum() ** 2 / (w ** 2).sum())
for beta in (0.25, 0.5, 1.0, 1.5, 2.0):
    w = np.exp(beta * z)
    w = w / w.mean()
    print(f"  beta={beta:<4} min {w.min():.3f} p10 {np.percentile(w,10):.3f} "
          f"p90 {np.percentile(w,90):.3f} max {w.max():.3f} CV {w.std():.3f} "
          f"ESS {ess(w):.0f}/220 ({100*ess(w)/220:.0f}%)  mean-weighted llv shift "
          f"{float((w*llv).mean() - llv.mean()):+.4f} nats")

# --------------------------------------------------------------- sigma init sensitivity
print("\nsigma_init = clip(gain*(quantile_q(llv) - llv_i), 0, cap)")
for q in (0.5, 0.75, 0.9, 0.95):
    ref = np.quantile(llv, q)
    for gain in (0.5, 1.0):
        sg = np.clip(gain * (ref - llv), 0.0, 2.5)
        print(f"  q={q:<5} gain={gain:<4} zero-frac {100*(sg<=1e-6).mean():4.1f}%  "
              f"med {np.median(sg):.3f} p90 {np.percentile(sg,90):.3f} max {sg.max():.3f} "
              f"mean {sg.mean():.3f}")
