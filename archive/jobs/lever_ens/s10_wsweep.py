"""Weight sweep for ONE added member on top of the calibrated k4 base.
Directly answers: 'a weaker but decorrelated member -- does it help, and at what weight?'
and calibrates the production 'mip3d member at w=0.2' choice.
Per-image metrics saved so 2-fold CV of the argmax can be done offline."""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens")
import harnlm as H
torch.backends.cudnn.benchmark = True
H.init()
singles = json.load(open(os.path.join(H.OUT, "singles.json")))
corr = {r["name"]: r for r in json.load(open(os.path.join(H.OUT, "corr.json")))}

K4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
ADD = ["e17visnorm", "e16app", "gsplatB8pure", "gsplatB1", "sh2", "sh0", "gsplatB6bilagrid"]
WS = [0.05, 0.10, 0.15, 0.20, 0.25, 1 / 3]

t0 = time.time()
base = H.score(K4, per_image=True)
print(f"BASE k4 = {base['score']:.4f}", flush=True)
out = {"base": {k: v for k, v in base.items() if k != "per"}, "base_per": base["per"], "rows": []}
for x in ADD:
    print(f"\n{x}  single {singles[x]['score']:.4f} (gap {singles[x]['score']-75.9644:+.3f})  "
          f"corr_k4 {corr[x]['corr_k4']:.3f}  resRMS_ratio {corr[x]['res_rms']/0.0609:.2f}", flush=True)
    for w in WS:
        ww = [(1 - w) / 4] * 4 + [w]
        d = H.score(K4 + [x], weights=ww, per_image=True)
        r = dict(add=x, w=w, score=d["score"], delta=d["score"] - base["score"],
                 psnr=d["psnr"], ssim=d["ssim"], lpips=d["lpips"], per=d["per"])
        out["rows"].append(r)
        print(f"   w={w:.3f}  SCORE {d['score']:8.4f}  delta {d['score']-base['score']:+8.4f}  "
              f"PSNR {d['psnr']:7.4f} SSIM {d['ssim']:.4f} LPIPS {d['lpips']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(H.OUT, "wsweep.json"), "w"))
print("DONE", time.time() - t0, flush=True)
