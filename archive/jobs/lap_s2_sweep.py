#!/usr/bin/env python
"""STEP 2: the sweep. Every config is scored with the PROJECT scorer on the production harness,
uint8-quantised (production always writes uint8).  Per-image rows are cached so step 3 can do
paired cross-validation on identical numbers.
"""
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lap_run import Harness, agg

OUT = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lap_rows"
os.makedirs(OUT, exist_ok=True)


def name(cfg):
    if cfg is None:
        return "pixelmean"
    kw = cfg.get("kw", {})
    s = f"{cfg['rule']}_n{cfg['nalt']}"
    for a in sorted(kw):
        s += f"_{a}{kw[a]}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", default=["m1", "m2", "m3", "m4"])
    ap.add_argument("--set", default="coarse")
    ap.add_argument("--jpeg", action="store_true")
    ap.add_argument("--tag", default="k4")
    args = ap.parse_args()

    cfgs = [None]
    if args.set == "coarse":
        for n in (1, 2, 3):
            cfgs += [
                dict(nalt=n, rule="maxmag", kw=dict(win=3)),
                dict(nalt=n, rule="median"),
                dict(nalt=n, rule="pnorm", kw=dict(p=2.0, win=3)),
                dict(nalt=n, rule="energy", kw=dict(lam=1.0, win=5)),
                dict(nalt=n, rule="inject", kw=dict(j=0, lam=0.5)),
            ]
        cfgs += [dict(nalt=1, rule="maxmag", kw=dict(win=1)),
                 dict(nalt=1, rule="maxmag", kw=dict(win=9)),
                 dict(nalt=1, rule="pnorm", kw=dict(p=1.0, win=3)),
                 dict(nalt=1, rule="pnorm", kw=dict(p=4.0, win=3)),
                 dict(nalt=1, rule="energy", kw=dict(lam=0.5, win=5)),
                 dict(nalt=1, rule="energy", kw=dict(lam=2.0, win=5)),
                 dict(nalt=1, rule="inject", kw=dict(j=0, lam=1.0)),
                 dict(nalt=1, rule="gain", kw=dict(g=1.02)),
                 dict(nalt=1, rule="gain", kw=dict(g=1.05)),
                 dict(nalt=1, rule="gain", kw=dict(g=1.10)),
                 dict(nalt=1, rule="gain", kw=dict(g=0.98)),
                 dict(nalt=5, rule="median"),
                 dict(nalt=5, rule="maxmag", kw=dict(win=3)),
                 ]
    elif args.set == "fine":
        for g in (1.01, 1.03, 1.04, 1.06, 1.08):
            cfgs.append(dict(nalt=1, rule="gain", kw=dict(g=g)))
        for g in (1.02, 1.05, 1.10):
            cfgs.append(dict(nalt=2, rule="gain", kw=dict(g=g)))
    elif args.set == "energy":
        for lam in (0.25, 0.75, 1.5, 3.0):
            cfgs.append(dict(nalt=1, rule="energy", kw=dict(lam=lam, win=5)))
        for w in (3, 9, 17):
            cfgs.append(dict(nalt=1, rule="energy", kw=dict(lam=1.0, win=w)))
    elif args.set == "best":
        pass

    H = Harness(args.members)
    print(f"\n{'config':<34} {'SCORE':>9} {'dScore':>8} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8}",
          flush=True)
    base = None
    for cfg in cfgs:
        t0 = time.time()
        rows = H.score_cfg(cfg, jpeg=args.jpeg)
        s, P, S, L = agg(rows)
        if base is None:
            base = s
        nm = name(cfg)
        np.save(f"{OUT}/{args.tag}{'_jpg' if args.jpeg else '_png'}__{nm}.npy", rows)
        print(f"{nm:<34} {s:9.4f} {s-base:+8.4f} {P:8.4f} {S:7.4f} {L:8.4f}"
              f"   [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
