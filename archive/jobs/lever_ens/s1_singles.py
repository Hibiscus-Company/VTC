import os, sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens")
import harn

shape = harn.init()
print("cache", shape, flush=True)
res = {}
t0 = time.time()
for i, n in enumerate(harn.NAMES):
    d = harn.score([i], per_image=True)
    res[n] = d
    print(f"{n:20s} PSNR {d['psnr']:7.4f}  SSIM {d['ssim']:.4f}  LPIPS {d['lpips']:.4f}  SCORE {d['score']:8.4f}  ({time.time()-t0:.0f}s)", flush=True)

# duplicate check: max abs pixel diff between every pair (uint8), cheap on GPU
R = harn.R()
V = R.shape[0]
print("\n--- pairwise mean |diff| (uint8 levels) ---", flush=True)
D = np.zeros((V, V))
for a in range(V):
    for b in range(a + 1, V):
        d = (R[a].float() - R[b].float()).abs().mean().item()
        D[a, b] = D[b, a] = d
np.save(os.path.join(harn.OUT, "pairdiff.npy"), D)
for a in range(V):
    print(f"{harn.NAMES[a]:20s} " + " ".join(f"{D[a,b]:5.2f}" for b in range(V)), flush=True)

with open(os.path.join(harn.OUT, "singles.json"), "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per"} for k, v in res.items()}, f, indent=1)
np.save(os.path.join(harn.OUT, "singles_per.npy"), res, allow_pickle=True)
print("DONE", time.time() - t0)
