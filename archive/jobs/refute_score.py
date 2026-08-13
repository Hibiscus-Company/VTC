#!/usr/bin/env python
"""INDEPENDENT RE-MEASUREMENT of the "field gain 1.30 -> 1.50" claim.

Production harness: HCM0181, REAL test poses, REAL test GT, models trained on 100% of train.
FULL shipped chain per arm: k4 pixel mean -> energy restore(lam=1.0,k=4) -> warp(field*g,
lanczos4) -> JPEG q100/ss2/optimize/progressive -> decode -> score.  Mean+restore shared.

Arms:
  b9_1.00     control anchor -- MUST reproduce fieldrefit/combo.py's 78.38774
  b9_1.24     the B9 field scaled to the SAME absolute magnitude as b11_1.50.  If score is a
              pure function of applied |f|, this ties b11_1.50 and the "model class" framing
              is just a magnitude re-normalisation.
  b11_1.30    the r29 BASELINE (what the private set ships today)
  b11_1.50    THE ACTUAL PROPOSAL (the claim's CLAIMED DELTA 0.0619 is its 1.55 arm, not this)
  b11a_1.30   same, but the field pooled over ALL 120 train views -- this is the EXACT
  b11a_1.50   configuration of the shipped private fields (60k/8M model, 120 view-pairs),
              whereas the claim's b11 arms pool only 60.
"""
import io, os, sys, time, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
OUT = os.path.join(HERE, "refute")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from energy_restore import restore                                     # noqa: E402
from fieldlib import upsample, warp                                    # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(4)
Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
POOL = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
D = lambda m: f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
SPEC = [("b9_1.00", "B9", 1.00), ("b9_1.24", "B9", 1.2401),
        ("b11_1.30", "B11", 1.30), ("b11_1.50", "B11", 1.50),
        ("b11a_1.30", "B11all", 1.30), ("b11a_1.50", "B11all", 1.50)]
ARMS = [a for a, _, _ in SPEC]
REF = "b11_1.30"


def ldt(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def report(per, acc, N, final):
    res = {}
    for a in ARMS:
        P, S, L = acc[a] / N
        res[a] = dict(score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)),
                      psnr=P, ssim=S, lpips=L)
    json.dump({"n": N, "final": final, "res": res, "per": per},
              open(os.path.join(OUT, "res_score.json"), "w"), indent=1)
    print(f"\n[n={N}] INDEPENDENT gain re-measurement, {TAG}, full shipped chain")
    print(f"{'arm':>10} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs b11_1.30':>12} {'t':>8} {'win/N':>7}")
    r0 = np.array(per[REF])
    for a in ARMS:
        v = np.array(per[a]) - r0
        t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != REF and v.std() > 0 else float("nan")
        ts = f"{t:8.2f}" if t == t else f"{'--':>8}"
        r_ = res[a]
        print(f"{a:>10} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:7.4f} {r_['lpips']:8.4f} "
              f"{r_['score']-res[REF]['score']:+12.4f} {ts} {int((v>0).sum()):3d}/{N}", flush=True)


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    dirs = [D(m) for m in POOL]
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))

    H, W = 989, 1320
    base = {}
    for t, fn in (("B9", "flow_HCM0181_B9.npz"), ("B11", "flow_HCM0181_B11.npz"),
                  ("B11all", "flow_HCM0181_B11all.npz")):
        z = np.load(os.path.join(HERE, "fieldrefit", fn))
        base[t] = np.median(z["s8"].astype(np.float32), 0)
        print(f"  {t}: {z['s8'].shape[0]} views  mean|f| "
              f"{np.linalg.norm(base[t],axis=2).mean():.4f}")
    F = {a: upsample(base[t] * c, H, W, "cubic") for a, t, c in SPEC}
    print("applied mean |field| px: " + "  ".join(
        f"{a} {np.linalg.norm(F[a],axis=2).mean():.4f}" for a in ARMS), flush=True)

    acc = {a: np.zeros(3) for a in ARMS}
    per = {a: [] for a in ARMS}
    t0 = time.time()
    for n, s_ in enumerate(stems):
        mem = [ldt(os.path.join(d, s_ + ".png")) for d in dirs]
        ens = torch.stack(mem).mean(0)
        er = restore(ens, mem, 1.0, len(mem)).clamp(0, 1)[0].permute(1, 2, 0).numpy()
        del mem, ens
        g = torch.from_numpy(
            np.asarray(Image.open(os.path.join(gtd, gt_by[s_])).convert("RGB"),
                       dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)

        def chain(a):
            x = np.clip(warp(er, F[a], "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                              dtype=np.float32) / 255.0

        with ThreadPoolExecutor(max_workers=6) as ex:
            jj = dict(zip(ARMS, ex.map(chain, ARMS)))
        for a in ARMS:
            r = torch.from_numpy(np.ascontiguousarray(jj[a])).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, g))
                L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            acc[a] += (P, S, L)
            per[a].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
            del r
        del g
        if n % 15 == 14 or n == len(stems) - 1:
            print(f"  {n+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
            report(per, acc, n + 1, final=(n == len(stems) - 1))
    print(f"\ndone {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
