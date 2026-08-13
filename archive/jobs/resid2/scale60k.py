#!/usr/bin/env python
"""THE SHIPPED FIELD GAIN IS CALIBRATED ON THE WRONG MODEL CLASS.

The 1.30 amplitude gain in build_r29.sh was measured against fields fit on HCM0181_gsplatB9ut,
which utfix.sh shows is the OLD 30k iters / 5M cap recipe.  Every field the private set actually
warps with is fit on /mnt/d/avv/r2r9/models/<T>_ut42/train_png, and all four private member
recipes read `--iters 60000 --cap_max 8000000` -- the PRODUCTION recipe, i.e. the same class as
HCM0181_gsplatB11ut60k, not B9.

That matters because the gain is an ABSORPTION correction, and a more converged model absorbs
more.  Measured against the test-pose oracle on HCM0181:
      field fit on      mean|f|   alpha*   alpha*|f|   R2 at own best scale
      B9   (30k/5M)      0.2112   1.3185     0.2785          0.9570
      B11  (60k/8M)      0.1746   1.5396     0.2689          0.9315
alpha*|f| is invariant to 1.5% while alpha moves 17%: the two fields chase the SAME target and
differ only in how much of it their model already swallowed.  The private fields' magnitudes
(0.156-0.160 px) sit 0.823x the public B9 fields -- and B11/B9 is 0.827.  So the private set is
being warped by B11-class fields multiplied by a B9-class constant.

Arms, full shipped chain (k4 mean -> restore lam=1.0 k=4 -> warp lanczos4 -> JPEG q100/ss2):
  b9_1.00   the unscaled B9 field                     (anchors to combo.py's 78.3877)
  b9_1.32   B9 field at its own alpha*                (the best configuration measured so far)
  b11_1.30  B11 field x 1.30                          <- WHAT THE PRIVATE SET SHIPS TODAY
  b11_1.45 / b11_1.55                                 <- the B11 alpha* is 1.5396
  b11_1.70  overshoot control, bounds the optimum
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
SPEC = [("b9_1.00", "B9", 1.00), ("b9_1.32", "B9", 1.3185),
        ("b11_1.30", "B11", 1.30), ("b11_1.45", "B11", 1.45),
        ("b11_1.55", "B11", 1.55), ("b11_1.70", "B11", 1.70)]
ARMS = [a for a, _, _ in SPEC]
REF = "b11_1.30"


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
    print(f"\n[n={N}] FIELD GAIN vs MODEL CLASS, {TAG}, full shipped chain")
    print(f"{'arm':>9} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs b11_1.30':>12} {'t':>8} {'win/N':>7}")
    r0 = np.array(per[REF])
    for a in ARMS:
        v = np.array(per[a]) - r0
        t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != REF and v.std() > 0 else float("nan")
        ts = f"{t:8.2f}" if t == t else f"{'--':>8}"
        r_ = res[a]
        print(f"{a:>9} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:7.4f} {r_['lpips']:8.4f} "
              f"{r_['score']-res[REF]['score']:+12.4f} {ts} {int((v>0).sum()):3d}/{N}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "res_scale60k.json"))
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

    H, W = 989, 1320
    base = {}
    for t in ("B9", "B11"):
        z = np.load(os.path.join(HERE, "fieldrefit", f"flow_{TAG}_{t}.npz"))
        base[t] = np.median(z["s8"].astype(np.float32), 0)
    F = {a: upsample(base[t] * c, H, W, "cubic") for a, t, c in SPEC}
    print("mean |field| px: " + "  ".join(
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
        if n % 10 == 9 or n == len(stems) - 1:
            print(f"  {n+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
            report(per, acc, n + 1, args.out, final=(n == len(stems) - 1))
    print(f"\nwrote {args.out}   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
