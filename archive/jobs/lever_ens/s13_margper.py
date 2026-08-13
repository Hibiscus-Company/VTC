"""Re-score add-one-to-k4 at uniform weight, KEEPING per-image metrics so the
'does this addition help' decision can be 2-fold cross-validated and bootstrapped."""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens")
import harnlm as H
torch.backends.cudnn.benchmark = True
H.init()
K4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
POOL = [n for n in H.NAMES if n not in ("sh3", "k4") and n not in K4]
t0 = time.time()
out = {}
d0 = H.score(K4, per_image=True)
out["__base__"] = d0
print(f"base {d0['score']:.4f}", flush=True)
for x in POOL:
    d = H.score(K4 + [x], per_image=True)
    out[x] = d
    print(f"{x:20s} {d['score']:8.4f} delta {d['score']-d0['score']:+.4f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(os.path.join(H.OUT, "margper.json"), "w"))
print("DONE", time.time() - t0, flush=True)
