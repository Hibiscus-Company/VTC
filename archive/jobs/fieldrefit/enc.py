#!/usr/bin/env python
"""SMOOTHING x ENCODE (ship=q100/ss2, 98=q98/ss0).  shape.py found that gaussian-smoothing the fitted field on its ds=8
grid, at FIXED mean amplitude, beats the shipped field: sigma 1.0 +0.0115 (t=16.4, 59/60),
sigma 2.0 +0.0146 (t=9.3, 55/60).  Both magnitude reprofiling (-0.026..-0.107) and radial gain
(-0.009, -0.002) were negative, so smoothing is the only shape axis with anything in it.

sigma 2.0 was the largest value tried and still rising, so the optimum is unlocated.  It must be
interior: at sigma -> infinity the field collapses to a pure translation, and a pure-translation
field has already been measured at -0.0002 (i.e. worthless) against the full field's ~+1.4.

Two forms per sigma:
  smX     smoothed then RENORMALISED to the shipped mean |f| -- structure only, the clean isolation
  smXraw  smoothed, then the production gain 1.30 -- what shipping it would actually do
(smoothing shrinks mean |f| by <1%, so the two should agree; smXraw is the ship-form check.)

Production harness: HCM0181, 60 REAL test poses, REAL test GT.  FULL shipped chain per arm:
k4 pixel mean -> energy restore(lam=1.0,k=4) -> warp(field,lanczos4) -> JPEG q100/ss2 -> score.
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
torch.set_num_threads(4)
Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
FR = os.path.join(HERE, "fieldrefit")
D = lambda m: f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
POOL = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
SIGS = [2.0]
ARMS = ["ship", "sm2", "ship98", "sm2_98"]
REF = "ship"


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
    print(f"\nSMOOTHING x ENCODE (ship=q100/ss2, 98=q98/ss0), {TAG}, n={N}, FULL shipped chain")
    print(f"{'arm':>8} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs ship':>9} {'paired t':>9} {'win/N':>7}")
    b = np.array(per[REF])
    for a in ARMS:
        v = np.array(per[a]) - b
        t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != REF and v.std() > 0 else float("nan")
        ts = f"{t:9.2f}" if t == t else f"{'--':>9}"
        r_ = res[a]
        print(f"{a:>8} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:7.4f} {r_['lpips']:8.4f} "
              f"{r_['score']-res[REF]['score']:+9.4f} {ts} {int((v>0).sum()):3d}/{N}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(FR, "res_enc.json"))
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

    Z = np.load(os.path.join(FR, f"flow_{TAG}_B9.npz"))
    H, W = [int(x) for x in Z["HW"]]
    base = np.median(Z["s8"].astype(np.float32), 0)
    mag = lambda f: float(np.linalg.norm(f, axis=2).mean())
    tgt = mag(base) * 1.30
    gs = lambda f, s: np.stack([cv2.GaussianBlur(f[..., c], (0, 0), s) for c in range(2)], -1)

    F = {"ship": base * 1.30}
    for s in SIGS:
        g_ = gs(base, s)
        F[f"sm{s:g}"] = g_ * (tgt / mag(g_))
    F["ship98"] = F["ship"]; F["sm2_98"] = F["sm2"]
    ENC = {"ship": SHIPPED, "sm2": SHIPPED,
           "ship98": dict(quality=98, subsampling=0, optimize=True, progressive=True),
           "sm2_98": dict(quality=98, subsampling=0, optimize=True, progressive=True)}

    print(f"{'arm':>8} {'mean|f|':>8} {'p95|f|':>8} {'cos(.,ship)':>12} {'mean|f-ship|':>13}")
    for a in ARMS:
        f = F[a]
        c = float((f * F['ship']).sum() /
                  np.sqrt((f * f).sum() * (F['ship'] * F['ship']).sum()))
        mg = np.linalg.norm(f, axis=2)
        print(f"{a:>8} {mg.mean():8.4f} {np.percentile(mg,95):8.4f} {c:12.5f} "
              f"{np.linalg.norm(f-F['ship'],axis=2).mean():13.4f}", flush=True)
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
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **ENC[a])
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
        if n % 10 == 9 or n == len(stems) - 1:
            print(f"  {n+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
            report(per, acc, n + 1, args.out, final=False)
    report(per, acc, len(stems), args.out, final=True)
    print(f"\nwrote {args.out}   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
