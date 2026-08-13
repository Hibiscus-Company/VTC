#!/usr/bin/env python
"""Opacity/scale census of bonsai ckpts -- CPU only, read-only."""
import sys, torch, numpy as np
torch.set_num_threads(2)
for p in sys.argv[1:]:
    c = torch.load(p, map_location="cpu", weights_only=False)
    sp = c["splats"]
    o = torch.sigmoid(sp["opacities"].float()).flatten().numpy()
    S = torch.exp(sp["scales"].float()).numpy()
    N = len(o)
    qs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    frac = {q: float((o < q).mean()) for q in qs}
    # opacity mass and "effective" count
    print(f"\n### {p}")
    print(f"  N = {N:,}   median opacity {np.median(o):.4f}  mean {o.mean():.4f}  sum {o.sum():,.0f}")
    print("  frac below: " + "  ".join(f"{q}:{frac[q]*100:5.2f}%" for q in qs))
    live = o > 0.05
    print(f"  live(op>0.05) N = {live.sum():,} ({100*live.mean():.2f}%)   "
          f"op>0.005 N = {(o>0.005).sum():,} ({100*(o>0.005).mean():.2f}%)")
    smax = S.max(1); smin = S.min(1)
    for tag, m in (("ALL", np.ones(N, bool)), ("live", live), ("dead(op<0.005)", o < 0.005)):
        if m.sum() == 0: continue
        print(f"  {tag:16s} n={m.sum():>9,}  geo-mean maxscale {np.exp(np.log(np.maximum(smax[m],1e-30)).mean()):.6g}  "
              f"median maxscale {np.median(smax[m]):.6g}  median minscale {np.median(smin[m]):.6g}")
    # how much of the rendered alpha budget the dead reservoir could ever carry
    print(f"  opacity mass in live splats: {o[live].sum()/o.sum()*100:.2f}% of total")
    del c, sp, o, S
