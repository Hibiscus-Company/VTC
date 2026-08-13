#!/usr/bin/env python
"""FIT-TARGET MISMATCH: the lens field is fit on ONE member's train renders but applied to the
ENSEMBLE (after energy restore).  Does fitting it on the ensemble instead buy anything?

Production harness: HCM0181, 60 REAL test poses, REAL test GT, models trained on 100% of train.
FULL shipped chain per arm:  k4 pixel mean -> energy restore(lam=1.0,k=4) -> warp(field,lanczos4)
-> JPEG q100/subsampling2/optimize/progressive -> decode -> score.

The mean and the restore are IDENTICAL across arms, so they are computed ONCE per image and only
the warp differs (machine discipline: no shared work inside the arm loop).

ARMS
  none        no field                                        reference
  B9          fit on member gsplatB9ut's 60 train renders     <- the SHIPPED recipe (1 member)
  B11         fit on member gsplatB11ut60k's SAME 60 views    <- CONTROL: does member identity matter?
  avgfield    0.5*(B9 field + B11 field)                      <- CONTROL: average FIELDS, not targets
  M2          fit on mean(B9,B11) train renders               <- TREATMENT: ensemble fit target
  M2ER        fit on energy-restored mean(B9,B11)             <- TREATMENT: exact chain input
  M2_mean     M2 target with mean pooling (estimator recheck)
  B11_n30 / B11_n120   view-DEPTH curve of the fit pool (B11_n60 == B11)
"""
import io, os, sys, time, json, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from PIL import Image

cv2.setNumThreads(1)          # threading happens at the arm level instead
torch.set_num_threads(8)

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from energy_restore import restore
from fieldlib import upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
FR = os.path.join(HERE, "fieldrefit")
D = lambda m: f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
POOL = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]


def ldt(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def report(arms, per, acc, N, out, final):
    res = {}
    for a in arms:
        P, S, L = acc[a] / N
        res[a] = dict(score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)),
                      psnr=P, ssim=S, lpips=L)
    json.dump({"n": N, "final": final, "res": res, "per": per}, open(out, "w"), indent=1)
    if not final:
        return
    print(f"\nFIT-TARGET MISMATCH, {TAG}, n={N}, FULL shipped chain "
          f"(k4 mean -> restore lam=1.0 -> warp lanczos4 -> JPEG q100/ss2)")
    print(f"{'arm':>10} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs none':>8} {'vs B9':>8} {'paired t':>9} {'win/N':>7}")
    b9 = np.array(per["B9"])
    for a in arms:
        v = np.array(per[a]) - b9
        t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != "B9" and v.std() > 0 else float("nan")
        ts = f"{t:9.2f}" if t == t else f"{'--':>9}"
        r_ = res[a]
        print(f"{a:>10} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:7.4f} {r_['lpips']:8.4f} "
              f"{r_['score']-res['none']['score']:+8.4f} {r_['score']-res['B9']['score']:+8.4f} "
              f"{ts} {int((v>0).sum()):3d}/{N}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(FR, "res_fittarget.json"))
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

    # ---------------- fields: built ONCE, full resolution
    Z = {t: np.load(os.path.join(FR, f"flow_{TAG}_{t}.npz")) for t in ("B9", "B11", "M2", "M2ER")}
    ZA = np.load(os.path.join(FR, f"flow_{TAG}_B11all.npz"))
    H, W = [int(x) for x in Z["B9"]["HW"]]
    med = lambda st: np.median(st.astype(np.float32), 0)

    F, SM = {}, {}
    for t in ("B9", "B11", "M2", "M2ER"):
        SM[t] = med(Z[t]["s8"])
        F[t] = upsample(SM[t], H, W, "cubic")
    F["avgfield"] = 0.5 * (F["B9"] + F["B11"])
    F["M2_mean"] = upsample(Z["M2"]["s8"].astype(np.float32).mean(0), H, W, "cubic")

    # view-depth curve, all from the same 120-view B11 stack
    fit_stems = [str(s) for s in Z["B9"]["stems"]]
    all_stems = [str(s) for s in ZA["stems"]]
    idx60 = [all_stems.index(s) for s in fit_stems if s in all_stems]
    rng = np.random.RandomState(0)
    for n in (30, 120):
        sel = np.arange(len(all_stems)) if n >= len(all_stems) else \
            rng.choice(len(all_stems), n, replace=False)
        F[f"B11_n{n}"] = upsample(med(ZA["s8"][sel]), H, W, "cubic")
    # sanity: B11 restricted to the 60 B9 views, rebuilt from the 120-stack -> must match F['B11']
    chk = upsample(med(ZA["s8"][np.array(idx60)]), H, W, "cubic")
    print(f"B11(60 from 120-stack) vs B11(own stack): mean |d| "
          f"{np.linalg.norm(chk - F['B11'], axis=2).mean():.5f} px")

    # sanity: does our recomputed B9 field reproduce the PRODUCTION cache?
    cache = np.load(os.path.join(HERE, "lens", "cache", f"pub_{TAG}.npz"))
    d = np.linalg.norm(med(cache["s8"]) - SM["B9"], axis=2)
    print(f"recomputed-B9 vs production cache field: mean |d| {d.mean():.5f} px  "
          f"max {d.max():.5f} px")

    # MAGNITUDE CONTROLS: every alternative target yields a SMALLER field than B9's
    # (mean|f| 0.175-0.192 vs 0.211).  Rescaling them to B9's mean magnitude separates
    # "the fit target changed the field's STRUCTURE" from "the fit target changed its SIZE".
    mag = lambda f: np.linalg.norm(f, axis=2).mean()
    F["M2_sc"] = F["M2"] * (mag(F["B9"]) / mag(F["M2"]))
    F["B11_sc"] = F["B11"] * (mag(F["B9"]) / mag(F["B11"]))

    print(f"\n{'field':>10} {'mean|f| px':>11} {'p95|f|':>8} {'mean|f-B9| px':>14}")
    for k in ("B9", "B11", "avgfield", "M2", "M2ER", "M2_mean", "B11_n30", "B11_n120",
              "M2_sc", "B11_sc"):
        m = np.linalg.norm(F[k], axis=2)
        dd = np.linalg.norm(F[k] - F["B9"], axis=2).mean()
        print(f"{k:>10} {m.mean():11.4f} {np.percentile(m,95):8.4f} {dd:14.4f}")

    arms = ["none", "B9", "B11", "M2", "M2ER", "M2_sc", "B11_sc", "B11_n120"]
    acc = {a: np.zeros(3) for a in arms}
    per = {a: [] for a in arms}
    t0 = time.time()
    for n, s in enumerate(stems):
        mem = [ldt(os.path.join(d, s + ".png")) for d in dirs]
        ens = torch.stack(mem).mean(0)
        # SHARED across all arms: pixel mean + energy restore, computed ONCE
        er = restore(ens, mem, 1.0, len(mem)).clamp(0, 1)[0].permute(1, 2, 0).numpy()
        del mem, ens
        g = torch.from_numpy(
            np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                       dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)

        def chain(a):
            """warp -> shipped JPEG round trip. Pure CPU, so run the arms in parallel."""
            x = er if a == "none" else np.clip(warp(er, F[a], "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                              dtype=np.float32) / 255.0

        with ThreadPoolExecutor(max_workers=min(len(arms), 5)) as ex:
            jj = dict(zip(arms, ex.map(chain, arms)))
        for a in arms:
            j = jj[a]
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, g))
                L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            acc[a] += (P, S, L)
            per[a].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
            del r
        del g
        if n % 5 == 4 or n == len(stems) - 1:
            # partial dump: this machine is congested, so results must be readable mid-run
            print(f"  {n+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
            report(arms, per, acc, n + 1, args.out, final=False)

    report(arms, per, acc, len(stems), args.out, final=True)
    print(f"\nwrote {args.out}   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
