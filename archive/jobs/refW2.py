#!/usr/bin/env python
"""REFUTE-W2: pool robustness of the winsor nudge at the PRODUCTION k (=10).

PoolC is DISJOINT from the k=10 production-shaped pool used in refW.py (no member in common).
Same chain: mean -> uint8 -> production restore(lam=1,k=10) -> gauss1 field x1.30 -> JPEG q100/ss2.
"""
import io, os, sys, time, json
import numpy as np
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from energy_restore import restore
from fieldlib import LooPool, gauss_smooth, upsample, warp

Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(2)
SHIP = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG, GAIN, DEV = "HCM0181", 1.30, os.environ.get("REFW_DEV", "cuda:1")
GTD = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
D = lambda m: f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"

# DISJOINT from refW's P10
PC = ["gsplatB1", "gsplatB2", "gsplatB3", "gsplatB4warm", "gsplatB5affine",
      "gsplatB6bilagrid", "gsplatB7ppisp2", "sh0", "sh1", "sh2"]


def r8(t):
    return (t.clamp(0, 1) * 255.0 + 0.5).floor().clamp(0, 255) / 255.0


def gap(X):
    S, _ = torch.sort(X, dim=0)
    k = X.shape[0]
    return (S[1] - S[0]) + (S[k - 2] - S[k - 1])


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(DEV).eval()
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(D(m), s + ".png")) for m in PC))
    c = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in c["HW"]]
    lens = upsample(gauss_smooth(LooPool(c["s8"]).pooled("median"), 1), H, W, "cubic") * GAIN

    ARMS = ["mC10", "wxC10_t0.5", "antiC10"]
    per = {a: [] for a in ARMS}
    t0 = time.time()
    for n, s in enumerate(stems):
        X = torch.stack([torch.from_numpy(
            np.asarray(Image.open(os.path.join(D(m), s + ".png")).convert("RGB"),
                       np.float32) / 255.0).permute(2, 0, 1) for m in PC], 0).to(DEV)
        mem = [X[i:i + 1] for i in range(10)]
        m = r8(X.mean(0))
        g = gap(X)
        gt = torch.from_numpy(np.asarray(Image.open(os.path.join(GTD, gt_by[s]))
                                         .convert("RGB"), np.float32) / 255.0
                              ).permute(2, 0, 1).unsqueeze(0).to(DEV)
        build = {"mC10": m,
                 "wxC10_t0.5": r8(m + (0.5 / 10) * g),
                 "antiC10": r8(m - (0.5 / 10) * g)}
        for a in ARMS:
            o = restore(build[a].unsqueeze(0), mem, 1.0, 10).clamp(0, 1)
            x = np.clip(warp(np.ascontiguousarray(o[0].permute(1, 2, 0).cpu().numpy()),
                             lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIP)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(DEV)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(float(((r - gt) ** 2).mean()), 1e-12))
                S = float(repo_ssim(r, gt))
                L = float(vgg(r * 2 - 1, gt * 2 - 1).item())
            per[a].append((P, S, L, 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.))))
        del X, mem, gt
        if n % 15 == 0:
            print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    A = {a: np.array(per[a]) for a in ARMS}
    json.dump({a: A[a].tolist() for a in ARMS}, open(f"{HERE}/refW2.json", "w"))
    print(f"\nREFUTE-W2  {TAG}  DISJOINT PoolC  k=10  n={N}")
    print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    for a in ARMS:
        mm = A[a].mean(0)
        print(f"{a:>12} {mm[3]:9.4f} {mm[0]:8.4f} {mm[1]:8.5f} {mm[2]:8.5f}")
    for a in ARMS[1:]:
        d = A[a][:, 3] - A["mC10"][:, 3]
        se = d.std(ddof=1) / np.sqrt(N)
        print(f"{a+' - mC10':>22} {d.mean():+9.4f} {se:7.4f} t={d.mean()/se:+6.2f} "
              f"{int((d>0).sum())}/{N}")


if __name__ == "__main__":
    main()
