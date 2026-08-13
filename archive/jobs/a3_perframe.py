"""A3 (a)(b)(c): per-frame metrics, GT sharpness, per-pixel error maps, per-tile LPIPS.
CPU ONLY. LPIPS scalar is computed with the EXACT eval_score.py definition
(lpips.LPIPS(net='vgg'), spatial_average over layers); the spatial map is the same
quantity before spatial averaging, bilinearly upsampled -- so tile sums attribute the
scalar LPIPS to image regions."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_common import *  # noqa
import numpy as np
import torch
import torch.nn.functional as F
import lpips as lpips_pkg

RENDER = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/avv/r36_shape/sr01/eval_png"
GT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/avv/evalsplit/bonsai/eval_gt"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/mnt/d/avv/r42_bonsai78/a3_diag"
TAG = sys.argv[4] if len(sys.argv) > 4 else "bonsai_sr01"
TILE = 128
os.makedirs(OUT, exist_ok=True)

model = lpips_pkg.LPIPS(net="vgg").to(DEV).eval()


def lpips_both(in0, in1):
    """returns (scalar exactly as eval_score.py, per-pixel map summing to ~scalar)"""
    a, b = model.scaling_layer(in0), model.scaling_layer(in1)
    o0, o1 = model.net.forward(a), model.net.forward(b)
    val = 0.0
    spat = None
    for kk in range(model.L):
        f0 = lpips_pkg.normalize_tensor(o0[kk])
        f1 = lpips_pkg.normalize_tensor(o1[kk])
        lin = model.lins[kk]((f0 - f1) ** 2)              # 1,1,h,w
        val = val + float(lin.mean())
        up = F.interpolate(lin, size=in0.shape[2:], mode="bilinear", align_corners=False)
        spat = up if spat is None else spat + up
    return val, spat[0, 0]


def tile_starts(n, t):
    s = list(range(0, n - t + 1, t))
    if s[-1] != n - t:
        s.append(n - t)
    return s


pairs = pair_list(RENDER, GT)
print(f"{len(pairs)} pairs", flush=True)

rows, tiles_lp, tiles_l1 = [], [], []
acc_l1 = acc_lp = None
ys = xs = None
t0 = time.time()
with torch.no_grad():
    for i, (stem, rp, gp) in enumerate(pairs):
        r = load(rp).to(DEV)
        g = load(gp).to(DEV)
        assert r.shape == g.shape, (stem, r.shape, g.shape)
        mse = ((r - g) ** 2).mean().item()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        ss = float(repo_ssim(r, g))
        lp, lp_map = lpips_both(r * 2 - 1, g * 2 - 1)
        lpm = lp_map.numpy().astype(np.float64)
        l1map = (r - g).abs().mean(1)[0].numpy().astype(np.float64)
        H, W = l1map.shape
        if acc_l1 is None:
            acc_l1 = np.zeros((H, W)); acc_lp = np.zeros((H, W))
            ys = tile_starts(H, TILE); xs = tile_starts(W, TILE)
        acc_l1 += l1map
        acc_lp += lpm
        tiles_lp.append([[lpm[y:y+TILE, x:x+TILE].mean() for x in xs] for y in ys])
        tiles_l1.append([[l1map[y:y+TILE, x:x+TILE].mean() for x in xs] for y in ys])
        gg = gray(g); rg = gray(r)
        rows.append(dict(stem=stem, frame=int(stem.split("_")[-1]) if stem.split("_")[-1].isdigit() else -1,
                         psnr=psnr, ssim=ss, lpips=lp, lpips_spatialmean=float(lpm.mean()),
                         score=score_from(psnr, ss, lp), l1=float(l1map.mean()),
                         gt_lapvar=lapvar(gg), rn_lapvar=lapvar(rg),
                         gt_mean=float(gg.mean()), rn_mean=float(rg.mean()),
                         gt_std=float(gg.std()), rn_std=float(rg.std())))
        print(f"[{i+1}/{len(pairs)}] {stem} P{psnr:.3f} S{ss:.4f} L{lp:.4f} "
              f"sc{rows[-1]['score']:.3f} gtlapv{rows[-1]['gt_lapvar']*1e4:.2f}e-4 "
              f"rnlapv{rows[-1]['rn_lapvar']*1e4:.2f}e-4 ({time.time()-t0:.0f}s)", flush=True)

n = len(pairs)
np.save(f"{OUT}/{TAG}_mean_l1_map.npy", (acc_l1 / n).astype(np.float32))
np.save(f"{OUT}/{TAG}_mean_lpips_map.npy", (acc_lp / n).astype(np.float32))
np.save(f"{OUT}/{TAG}_tiles_lpips.npy", np.array(tiles_lp, dtype=np.float32))
np.save(f"{OUT}/{TAG}_tiles_l1.npy", np.array(tiles_l1, dtype=np.float32))
np.save(f"{OUT}/{TAG}_tile_grid.npy", np.array([ys, ], dtype=object), allow_pickle=True)
with open(f"{OUT}/{TAG}_tile_coords.json", "w") as f:
    json.dump(dict(tile=TILE, ys=ys, xs=xs, H=int(acc_l1.shape[0]), W=int(acc_l1.shape[1])), f)
with open(f"{OUT}/{TAG}_perframe.json", "w") as f:
    json.dump(rows, f, indent=1)

P = float(np.mean([r["psnr"] for r in rows])); S = float(np.mean([r["ssim"] for r in rows]))
L = float(np.mean([r["lpips"] for r in rows]))
print(f"AGG n={n} PSNR {P:.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {score_from(P,S,L):.4f}", flush=True)
