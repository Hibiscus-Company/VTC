#!/usr/bin/env python
"""FIT-TARGET MISMATCH, ISOLATED FROM MAGNITUDE -- the control combo.py was missing.

combo.py already showed that swapping the field's fit target from ONE member (B9, the shipped
recipe) to the 2-member ENSEMBLE MEAN (M2) costs -0.0859 at gain 1.0.  But every alternative
target yields a SMALLER field (mean|f|: B9 0.2112, M2 0.1897, B11 0.1746) and the field is known
to UNDERSHOOT -- production ships an amplitude gain of 1.30.  So a target that shrinks the field
is penalised for its SIZE, and the fit-target question is confounded.

This script asks the production question instead: at the SHIPPED amplitude (B9 x 1.30), does the
fit target's STRUCTURE matter at all?  Every treatment field is rescaled so its mean |f| equals
the shipped field's, which is exactly the gain re-tuning production would do anyway.

Production harness: HCM0181, 60 REAL test poses, REAL test GT, models trained on 100% of train.
FULL shipped chain per arm: k4 pixel mean -> energy restore(lam=1.0,k=4) -> warp(field,lanczos4)
-> JPEG q100/ss2/optimize/progressive -> decode -> score.  Mean+restore computed ONCE per image
and shared by all arms (machine discipline).

ARMS
  B9            shipped field, gain 1.00                       ties back to combo.log (78.3877)
  B9x130        shipped field, gain 1.30                       <- r29 REFERENCE (combo.log 78.5277)
  M2x130        ensemble-target field x 1.30, RAW magnitude    naive production swap (confounded)
  M2m130        ensemble-target field, magnitude-MATCHED       <- TREATMENT, structure only
  M2ERm130      fit on energy-restored ensemble, matched       <- TREATMENT, exact chain input
  B11m130       a DIFFERENT single member, matched             <- CONTROL: member identity
  B11n30m130    same member, 30-view fit pool, matched         <- pool DEPTH
  B11n120m130   same member, 120-view fit pool, matched        <- pool DEPTH (60 = B11m130)
"""
import io, os, sys, time, json, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from energy_restore import restore                       # noqa: E402
from fieldlib import upsample, warp                      # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
FR = os.path.join(HERE, "fieldrefit")
D = lambda m: f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
POOL = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
ARMS = ["B9", "B9x130", "M2x130", "M2m130", "M2ERm130", "B11m130",
        "B11n30m130", "B11n120m130"]
REF = "B9x130"


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
    if not final:
        return
    print(f"\nFIT-TARGET AT MATCHED AMPLITUDE, {TAG}, n={N}, FULL shipped chain "
          f"(k4 mean -> restore lam=1.0 -> warp lanczos4 -> JPEG q100/ss2)")
    print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs B9x130':>10} {'paired t':>9} {'win/N':>7}")
    b = np.array(per[REF])
    for a in ARMS:
        v = np.array(per[a]) - b
        t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != REF and v.std() > 0 else float("nan")
        ts = f"{t:9.2f}" if t == t else f"{'--':>9}"
        r_ = res[a]
        print(f"{a:>12} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:7.4f} {r_['lpips']:8.4f} "
              f"{r_['score']-res[REF]['score']:+10.4f} {ts} {int((v>0).sum()):3d}/{N}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(FR, "res_target2.json"))
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

    med = lambda st: np.median(st.astype(np.float32), 0)
    Z = {t: np.load(os.path.join(FR, f"flow_{TAG}_{t}.npz")) for t in ("B9", "B11", "M2", "M2ER")}
    ZA = np.load(os.path.join(FR, f"flow_{TAG}_B11all.npz"))
    H, W = [int(x) for x in Z["B9"]["HW"]]
    S8 = {t: med(Z[t]["s8"]) for t in Z}
    rng = np.random.RandomState(0)
    for n in (30, 120):
        sel = (np.arange(ZA["s8"].shape[0]) if n >= ZA["s8"].shape[0]
               else rng.choice(ZA["s8"].shape[0], n, replace=False))
        S8[f"B11n{n}"] = med(ZA["s8"][sel])

    mag = lambda f: float(np.linalg.norm(f, axis=2).mean())
    base = S8["B9"]
    tgt = mag(base) * 1.30                       # the shipped amplitude
    F = {"B9": base.copy(), "B9x130": base * 1.30, "M2x130": S8["M2"] * 1.30}
    for a, k in (("M2m130", "M2"), ("M2ERm130", "M2ER"), ("B11m130", "B11"),
                 ("B11n30m130", "B11n30"), ("B11n120m130", "B11n120")):
        F[a] = S8[k] * (tgt / mag(S8[k]))
    print(f"{'arm':>12} {'mean|f| px':>11} {'p95|f|':>8} {'cos(.,B9)':>10} "
          f"{'mean|f-ref| px':>15}")
    ref = F[REF]
    for a in ARMS:
        f = F[a]
        c = float((f * base).sum() / np.sqrt((f * f).sum() * (base * base).sum()))
        m = np.linalg.norm(f, axis=2)
        print(f"{a:>12} {m.mean():11.4f} {np.percentile(m,95):8.4f} {c:10.4f} "
              f"{np.linalg.norm(f-ref,axis=2).mean():15.4f}", flush=True)
    FU = {a: upsample(F[a], H, W, "cubic") for a in ARMS}

    acc = {a: np.zeros(3) for a in ARMS}
    per = {a: [] for a in ARMS}
    t0 = time.time()
    for n, s in enumerate(stems):
        mem = [ldt(os.path.join(d, s + ".png")) for d in dirs]
        ens = torch.stack(mem).mean(0)
        er = restore(ens, mem, 1.0, len(mem)).clamp(0, 1)[0].permute(1, 2, 0).numpy()
        del mem, ens
        g = torch.from_numpy(
            np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                       dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)

        def chain(a):
            x = np.clip(warp(er, FU[a], "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                              dtype=np.float32) / 255.0

        with ThreadPoolExecutor(max_workers=8) as ex:
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
            report(per, acc, n + 1, args.out, final=False)
    report(per, acc, len(stems), args.out, final=True)
    print(f"\nwrote {args.out}   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
