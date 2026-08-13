#!/usr/bin/env python
"""IS THE 1.30 AMPLITUDE GAIN THE RIGHT *SHAPE*?  A pure-shape probe at FIXED mean amplitude.

r29 ships field -> 1.30*field, one global scalar.  If the train-fit shrinkage is not uniform
across the frame, a scalar is the wrong correction and the residual is free score.  Every arm
below is renormalised to the SAME mean |f| as the shipped 1.30*B9 field, so amplitude is held
constant and only the field's SHAPE varies -- the control the scalar sweep never had.

  ship        1.30 * B9                                     <- r29 REFERENCE
  cmp0.15     magnitude-compressed: f*(|f|/m)^-0.15, renorm  tail pulled in
  cmp0.30     f*(|f|/m)^-0.30, renorm                        tail pulled in harder
  exp0.15     f*(|f|/m)^+0.15, renorm                        tail pushed out (opposite sign)
  radin       radial gain 1 -/+ 0.25 favouring the CENTRE, renorm
  radout      radial gain favouring the EDGE, renorm
  sm1.0       gaussian-smoothed on the ds8 grid (sigma 1.0), renorm
  sm2.0       gaussian-smoothed (sigma 2.0), renorm

Production harness: HCM0181, 60 REAL test poses, REAL test GT.  FULL shipped chain per arm.
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
ARMS = ["ship", "cmp0.15", "cmp0.30", "exp0.15", "radin", "radout", "sm1.0", "sm2.0"]
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
    print(f"\nFIELD SHAPE AT FIXED AMPLITUDE, {TAG}, n={N}, FULL shipped chain")
    print(f"{'arm':>9} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs ship':>9} {'paired t':>9} {'win/N':>7}")
    b = np.array(per[REF])
    for a in ARMS:
        v = np.array(per[a]) - b
        t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != REF and v.std() > 0 else float("nan")
        ts = f"{t:9.2f}" if t == t else f"{'--':>9}"
        r_ = res[a]
        print(f"{a:>9} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:7.4f} {r_['lpips']:8.4f} "
              f"{r_['score']-res[REF]['score']:+9.4f} {ts} {int((v>0).sum()):3d}/{N}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(FR, "res_shape.json"))
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
    ren = lambda f: f * (tgt / mag(f))

    m = np.linalg.norm(base, axis=2, keepdims=True)
    mm = max(mag(base), 1e-8)
    h, w, _ = base.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.sqrt(((xx - (w - 1) / 2) / ((w - 1) / 2)) ** 2 +
                 ((yy - (h - 1) / 2) / ((h - 1) / 2)) ** 2)[..., None]
    rn = rr / rr.max()
    gs = lambda f, s: np.stack([cv2.GaussianBlur(f[..., c], (0, 0), s) for c in range(2)], -1)

    F = {"ship": base * 1.30}
    for b_ in (0.15, 0.30):
        F[f"cmp{b_:.2f}"] = ren(base * np.power(np.maximum(m, 1e-6) / mm, -b_))
    F["exp0.15"] = ren(base * np.power(np.maximum(m, 1e-6) / mm, 0.15))
    F["radin"] = ren(base * (1.0 + 0.25 * (0.5 - rn)))
    F["radout"] = ren(base * (1.0 + 0.25 * (rn - 0.5)))
    F["sm1.0"] = ren(gs(base, 1.0))
    F["sm2.0"] = ren(gs(base, 2.0))

    print(f"{'arm':>9} {'mean|f|':>8} {'p95|f|':>8} {'cos(.,ship)':>12} {'mean|f-ship|':>13}")
    for a in ARMS:
        f = F[a]
        c = float((f * F['ship']).sum() /
                  np.sqrt((f * f).sum() * (F['ship'] * F['ship']).sum()))
        mg = np.linalg.norm(f, axis=2)
        print(f"{a:>9} {mg.mean():8.4f} {np.percentile(mg,95):8.4f} {c:12.5f} "
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
        if n % 10 == 9 or n == len(stems) - 1:
            print(f"  {n+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
            report(per, acc, n + 1, args.out, final=False)
    report(per, acc, len(stems), args.out, final=True)
    print(f"\nwrote {args.out}   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
