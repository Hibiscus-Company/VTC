#!/usr/bin/env python
"""IS THERE SHAPE IN THE REGISTRATION RESIDUAL, OR ONLY MAGNITUDE?  4 arms, one shared restore.

Production harness: HCM0181, REAL test poses, REAL test GT, models trained on 100% of train.
FULL shipped chain per arm, identical to fieldrefit/combo.py so the numbers are directly
comparable to its B9 78.3877 / s1.30 78.5277:
    k4 pixel mean -> energy restore(lam=1.0,k=4) -> warp(field, lanczos4) -> JPEG q100/ss2
    optimize/progressive -> decode -> score.  Mean+restore computed ONCE per image.

  base    f1                      the shipped train-fitted median field
  s       1.3185 * f1             SCALE ONLY, at the oracle projection alpha* -- the operating
                                  point (an independent replication of the +0.14 scale win)
  cum2s   1.2264 * cum2           the train-only ITERATED field (f1 (+) f2, second-order fit on
                                  the residual after the first field), rescaled to EXACTLY the
                                  same mean magnitude as arm s.  arm s vs arm cum2s is therefore
                                  a pure SHAPE comparison with magnitude held fixed -- the
                                  control that decides whether iterating extracts structure.
  cum2    cum2                    the iterated field raw, i.e. what a train-only pipeline could
                                  ship on the private set with NO calibration constant at all.
"""
import io, os, sys, time, json, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
OUT = os.path.join(HERE, "resid2")
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
ARMS = ["base", "s", "cum2s", "cum2"]
ALPHA = 1.31846


def ldt(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def report(per, acc, N, out, final):
    res = {}
    for a in ARMS:
        P, S, L = acc[a] / N
        res[a] = dict(score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)),
                      psnr=P, ssim=S, lpips=L)
    json.dump({"n": N, "final": final, "res": res, "per": per}, open(out, "w"), indent=1)
    print(f"\n[n={N}] SHAPE-vs-MAGNITUDE, {TAG}, full shipped chain")
    print(f"{'arm':>7} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs base':>8} {'vs s':>8} {'t vs s':>8} {'win/N':>7}")
    b = np.array(per["base"]); s = np.array(per["s"])
    for a in ARMS:
        v = np.array(per[a]) - s
        t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != "s" and v.std() > 0 else float("nan")
        ts = f"{t:8.2f}" if t == t else f"{'--':>8}"
        r_ = res[a]
        print(f"{a:>7} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:7.4f} {r_['lpips']:8.4f} "
              f"{r_['score']-res['base']['score']:+8.4f} {r_['score']-res['s']['score']:+8.4f} "
              f"{ts} {int((v>0).sum()):3d}/{N}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "res_score2.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    dirs = [D(m) for m in POOL]
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
    if args.limit:
        stems = stems[:args.limit]

    I = np.load(os.path.join(OUT, f"iter_{TAG}_gsplatB9ut.npz"))
    f1g, cum2g = I["f1"], I["cum2"]
    H, W = 989, 1320
    m1 = np.linalg.norm(f1g, axis=2).mean()
    m2 = np.linalg.norm(cum2g, axis=2).mean()
    k = ALPHA * m1 / m2                       # rescale cum2 to arm s's mean magnitude
    F = {"base": upsample(f1g, H, W, "cubic"),
         "s": upsample(f1g * ALPHA, H, W, "cubic"),
         "cum2s": upsample(cum2g * k, H, W, "cubic"),
         "cum2": upsample(cum2g, H, W, "cubic")}
    mag = lambda f: np.linalg.norm(f, axis=2).mean()
    print("mean |field| px: " + "  ".join(f"{a} {mag(F[a]):.4f}" for a in ARMS))
    print(f"cum2 rescale factor k={k:.4f}  (arm s and arm cum2s magnitudes matched to "
          f"{mag(F['s']):.4f} vs {mag(F['cum2s']):.4f})", flush=True)

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

        with ThreadPoolExecutor(max_workers=4) as ex:
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
        if n % 5 == 4 or n == len(stems) - 1:
            print(f"  {n+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
            report(per, acc, n + 1, args.out, final=(n == len(stems) - 1))
    print(f"\nwrote {args.out}   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
