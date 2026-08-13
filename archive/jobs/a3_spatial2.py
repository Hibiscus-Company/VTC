"""A3 (c) deeper: what does the spatial error follow? tile-level regression of LPIPS
density on GT texture/edge/depth-discontinuity proxies; bad8 vs good20 maps; quick-look PNGs."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_common import *  # noqa
import numpy as np
from scipy import stats
import cv2

OUT = "/mnt/d/avv/r42_bonsai78/a3_diag"
RENDER = "/mnt/d/avv/r36_shape/sr01/eval_png"
GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
co = json.load(open(f"{OUT}/bonsai_sr01_tile_coords.json"))
ys, xs, T = co["ys"], co["xs"], co["tile"]
tl = np.load(f"{OUT}/bonsai_sr01_tiles_lpips.npy")   # 28,9,15
t1 = np.load(f"{OUT}/bonsai_sr01_tiles_l1.npy")
rows = sorted(json.load(open(f"{OUT}/bonsai_sr01_perframe.json")), key=lambda r: r["frame"])
order = [r["stem"] for r in rows]
pairs = {s: (rp, gp) for s, rp, gp in pair_list(RENDER, GT)}
# a3_perframe iterated sorted(listdir) == sorted stems == frame order for bonsai
assert order == sorted(order)

# GT texture features per tile
feat = {k: [] for k in ["gtvar", "gtgrad", "gtlap", "rnvar", "rngrad", "dvar"]}
for s in order:
    rp, gp = pairs[s]
    g = gray(load(gp)); r = gray(load(rp))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    gm = np.hypot(gx, gy)
    rx = cv2.Sobel(r, cv2.CV_32F, 1, 0, 3); ry = cv2.Sobel(r, cv2.CV_32F, 0, 1, 3)
    rm = np.hypot(rx, ry)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    for name, M in [("gtvar", None), ("gtgrad", gm), ("gtlap", lap), ("rngrad", rm)]:
        if M is None:
            feat[name].append([[g[y:y+T, x:x+T].var() for x in xs] for y in ys])
        else:
            feat[name].append([[M[y:y+T, x:x+T].mean() for x in xs] for y in ys])
    feat["rnvar"].append([[r[y:y+T, x:x+T].var() for x in xs] for y in ys])
F = {k: np.array(v, np.float32) for k, v in feat.items() if v}

lp_all = tl.mean(0).ravel()
print("=== tile-level drivers of LPIPS density (135 tiles, 28-frame means) ===")
for k in ["gtvar", "gtgrad", "gtlap", "rngrad"]:
    m = F[k].mean(0).ravel()
    print(f"  LPIPS density vs mean GT {k:7s}: spearman {stats.spearmanr(m, lp_all)[0]:+.3f}")
# sharpness DEFICIT per tile (render gradient / GT gradient)
defi = (F["rngrad"].mean(0) / np.maximum(F["gtgrad"].mean(0), 1e-8)).ravel()
print(f"  LPIPS density vs tile sharpness ratio (rn_grad/gt_grad): "
      f"spearman {stats.spearmanr(defi, lp_all)[0]:+.3f}")
print(f"  tile sharpness ratio: min {defi.min():.3f} med {np.median(defi):.3f} max {defi.max():.3f}")

# per-tile pooled over all 28 frames (n=28*135) -- within-frame variation too
lpf = tl.reshape(-1); gvf = F["gtgrad"].reshape(-1)
print(f"  pooled (28x135=3780) LPIPS vs GT gradient: spearman {stats.spearmanr(gvf, lpf)[0]:+.3f}")

bad, good = np.arange(8), np.arange(8, 28)
tb, tg = tl[bad].mean(0), tl[good].mean(0)
print(f"\n  mean LPIPS density  first8 {tb.mean():.4f}  last20 {tg.mean():.4f}")
print(f"  spatial correlation of the two maps: pearson "
      f"{stats.pearsonr(tb.ravel(), tg.ravel())[0]:+.3f}")
for nm, t in [("first8", tb), ("last20", tg)]:
    f = np.sort(t.ravel())[::-1]
    k = int(round(0.10 * f.size))
    print(f"  {nm}: worst 10% of tiles carry {100*f[:k].sum()/f.sum():.1f}% of LPIPS; "
          f"rows(top->bot) " + " ".join(f"{v:.3f}" for v in t.mean(1)))
np.save(f"{OUT}/bonsai_sr01_tilemean_lpips_first8.npy", tb)
np.save(f"{OUT}/bonsai_sr01_tilemean_lpips_last20.npy", tg)
np.save(f"{OUT}/bonsai_sr01_tile_gtgrad.npy", F["gtgrad"].mean(0))
np.save(f"{OUT}/bonsai_sr01_tile_rngrad.npy", F["rngrad"].mean(0))
np.save(f"{OUT}/bonsai_sr01_tile_gtvar.npy", F["gtvar"].mean(0))

# quick-look PNGs of the pixel maps
for nm in ["mean_lpips_map", "mean_l1_map"]:
    m = np.load(f"{OUT}/bonsai_sr01_{nm}.npy").astype(np.float32)
    v = np.clip((m - np.percentile(m, 1)) / (np.percentile(m, 99) - np.percentile(m, 1)), 0, 1)
    cv2.imwrite(f"{OUT}/bonsai_sr01_{nm}.png",
                cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
print(f"\nwrote quick-look PNGs + tile feature npys to {OUT}")
