import os, sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/bkai/.claude/jobs/1c9cf7e9/tmp/lever_ens")
import harn
torch.backends.cudnn.benchmark = True
harn.init()
N = harn.NAMES
singles = json.load(open(os.path.join(harn.OUT, "singles.json")))

# candidate pool: 20 real model variants. sh3 is a BIT-EXACT duplicate of gsplatB11ut60k -> excluded.
# k4 is a derived ensemble -> excluded.
POOL = [n for n in N if n not in ("sh3", "k4")]
print("pool", len(POOL), POOL, flush=True)

t0 = time.time()
cache = {}


def sc(members):
    key = tuple(sorted(members))
    if key not in cache:
        cache[key] = harn.score(list(key), per_image=True)
    return cache[key]


log = {"greedy": [], "rank": []}

# ---------- A. rank-order path: add in descending single-model score ----------
order = sorted(POOL, key=lambda n: -singles[n]["score"])
print("\n=== RANK-ORDER PATH (add best-single-score first) ===", flush=True)
for k in range(1, len(order) + 1):
    d = sc(order[:k])
    log["rank"].append(dict(k=k, added=order[k - 1], members=order[:k],
                            psnr=d["psnr"], ssim=d["ssim"], lpips=d["lpips"], score=d["score"]))
    print(f"k={k:2d} +{order[k-1]:18s} PSNR {d['psnr']:7.4f} SSIM {d['ssim']:.4f} "
          f"LPIPS {d['lpips']:.4f} SCORE {d['score']:8.4f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(log, open(os.path.join(harn.OUT, "greedy.json"), "w"), indent=1)

# ---------- B. greedy forward selection maximizing the real score ----------
print("\n=== GREEDY FORWARD SELECTION ===", flush=True)
cur = []
rest = list(POOL)
while rest:
    best = None
    for c in rest:
        d = sc(cur + [c])
        if best is None or d["score"] > best[1]["score"]:
            best = (c, d)
    c, d = best
    cur = cur + [c]
    rest.remove(c)
    log["greedy"].append(dict(k=len(cur), added=c, members=list(cur), single=singles[c]["score"],
                              psnr=d["psnr"], ssim=d["ssim"], lpips=d["lpips"], score=d["score"]))
    print(f"k={len(cur):2d} +{c:18s} (single {singles[c]['score']:7.4f}) PSNR {d['psnr']:7.4f} "
          f"SSIM {d['ssim']:.4f} LPIPS {d['lpips']:.4f} SCORE {d['score']:8.4f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(log, open(os.path.join(harn.OUT, "greedy.json"), "w"), indent=1)

np.save(os.path.join(harn.OUT, "cache_per.npy"), cache, allow_pickle=True)
print("DONE", time.time() - t0, "evals", len(cache), flush=True)
