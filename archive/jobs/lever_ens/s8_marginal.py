"""THE 0.15-BAND TEST.
Base = k4 (the calibrated production-harness ensemble: the 4 UT members, uniform pixel-mean).
For every other model variant X, measure the score of mean(k4 members + X) at uniform 1/5 weight.
Marginal delta vs base, plotted against X's own single score and its residual decorrelation.
Also: same test with a 6-member 'production-analogue' base (mixed families)."""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens")
import harnlm as H
torch.backends.cudnn.benchmark = True
H.init()
singles = json.load(open(os.path.join(H.OUT, "singles.json")))
corr = {r["name"]: r for r in json.load(open(os.path.join(H.OUT, "corr.json")))}

K4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
POOL = [n for n in H.NAMES if n not in ("sh3", "k4")]
res = {}
t0 = time.time()

for tag, base in [("k4", K4)]:
    d0 = H.score(base, per_image=True)
    res[f"BASE_{tag}"] = {k: v for k, v in d0.items() if k != "per"}
    print(f"BASE {tag} ({len(base)}): SCORE {d0['score']:.4f}  PSNR {d0['psnr']:.4f} "
          f"SSIM {d0['ssim']:.4f} LPIPS {d0['lpips']:.4f}", flush=True)
    print(f"\n{'add':20s} {'single':>8s} {'gap':>7s} {'corr':>6s} {'k+1 score':>10s} {'delta':>8s} "
          f"{'dPSNR':>7s} {'dSSIM':>8s} {'dLPIPS':>8s}", flush=True)
    rows = []
    for x in POOL:
        if x in base:
            continue
        d = H.score(base + [x], per_image=True)
        r = dict(add=x, single=singles[x]["score"], gap=singles[x]["score"] - singles["gsplatB11ut60k"]["score"],
                 corr_k4=corr[x]["corr_k4"], score=d["score"], delta=d["score"] - d0["score"],
                 dpsnr=d["psnr"] - d0["psnr"], dssim=d["ssim"] - d0["ssim"], dlpips=d["lpips"] - d0["lpips"])
        rows.append(r)
        print(f"{x:20s} {r['single']:8.4f} {r['gap']:+7.3f} {r['corr_k4']:6.3f} {r['score']:10.4f} "
              f"{r['delta']:+8.4f} {r['dpsnr']:+7.4f} {r['dssim']:+8.5f} {r['dlpips']:+8.5f}", flush=True)
        json.dump({**res, tag: rows}, open(os.path.join(H.OUT, "marginal.json"), "w"), indent=1)
    res[tag] = rows
json.dump(res, open(os.path.join(H.OUT, "marginal.json"), "w"), indent=1)
print("DONE", time.time() - t0, flush=True)
