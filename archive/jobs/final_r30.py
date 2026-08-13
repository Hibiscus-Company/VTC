#!/usr/bin/env python
"""DECISIVE r30 ARM TEST: do q98/ss0 and field gain 1.45 stack on top of the ALREADY-BUILT
k=10 + gauss1 chain (sub_round30_k10.zip)?

Chain emulated exactly: 10-member uniform mean (kcurve2 ORDER[:10]) -> restore(lam=1.0, k=10)
-> gauss1-smoothed B11-CLASS field x gain -> JPEG -> decode -> score vs real test GT.
B11 class because refute_class.log measured the PRIVATE fields' own model class directly:
fields fit on 30k/5M renders are 1.1486x the fields fit on 60k/8M renders (corr 0.987-0.993,
5/5 towers), and the private fields are fit on <T>_ut42 which train_args.txt shows is
`--iters 60000 --cap_max 8000000`. So the shipped 1.30 undershoots by ~1.149x -> ~1.49.
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
from energy_restore import restore                                    # noqa: E402
from fieldlib import gauss_smooth, upsample, warp                     # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(4)
Image.MAX_IMAGE_PIXELS = None
TAG = "HCM0181"
ORDER = ["gsplatB11ut60k", "sh3", "m31b_nolpips", "m31b_taillpips", "gsplatB10ut8M",
         "gsplatB12ut8Ms7", "e17visnorm", "e15ceil95", "gsplatB9ut", "gsplatB8pure"]
D = lambda m: f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
Q100 = dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q98 = dict(quality=98, subsampling=0, optimize=True, progressive=True)

#       name           gain  encode
SPEC = [("k10_ref",    1.30, Q100),   # == sub_round30_k10.zip
        ("g145",       1.45, Q100),
        ("g150",       1.50, Q100),
        ("q98",        1.30, Q98),
        ("g145_q98",   1.45, Q98),    # <- the proposal
        ("g150_q98",   1.50, Q98),
        ("g160_q98",   1.60, Q98)]    # overshoot control
ARMS = [a for a, *_ in SPEC]
REF = "k10_ref"


def ld(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),
                                       dtype=np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)


def report(per, acc, nb, N, out):
    res = {}
    for a in ARMS:
        P, S, L = acc[a] / N
        res[a] = dict(score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.)),
                      psnr=P, ssim=S, lpips=L, mb=nb[a] / 1e6)
    json.dump({"n": N, "res": res, "per": per}, open(out, "w"), indent=1)
    r0 = np.array(per[REF])
    print(f"\n[n={N}] r30 FINAL ARMS on the k=10 + gauss1 chain, {TAG}, real test GT")
    print(f"{'arm':>10} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs k10':>9} "
          f"{'t':>7} {'win/N':>7} {'MB/60':>8}")
    for a in ARMS:
        v = np.array(per[a]) - r0
        t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != REF and v.std() > 0 else float("nan")
        ts = f"{t:7.2f}" if t == t else f"{'--':>7}"
        r_ = res[a]
        print(f"{a:>10} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:8.5f} {r_['lpips']:8.5f} "
              f"{r_['score']-res[REF]['score']:+9.4f} {ts} {int((v>0).sum()):3d}/{N} "
              f"{r_['mb']:8.2f}", flush=True)
    s = {a: res[a]["score"] - res[REF]["score"] for a in ARMS}
    tot = s["q98"] + s["g145"]
    print(f"  ADDITIVITY: q98 {s['q98']:+.4f} + g145 {s['g145']:+.4f} = {tot:+.4f}  "
          f"vs JOINT {s['g145_q98']:+.4f}  (ratio {s['g145_q98']/tot if abs(tot)>1e-9 else float('nan'):.2f})",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "final_r30.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(D(m), s + ".png")) for m in ORDER))
    if args.limit:
        stems = stems[:args.limit]
    H, W = 989, 1320
    z = np.load(os.path.join(HERE, "fieldrefit", f"flow_{TAG}_B11.npz"))
    b11g1 = gauss_smooth(np.median(z["s8"].astype(np.float32), 0), 1)
    F = {g: upsample(b11g1 * g, H, W, "cubic") for _, g, _ in SPEC}
    print(f"pool k={len(ORDER)}  n={len(stems)}  field mean|f| px: " +
          "  ".join(f"x{g} {np.linalg.norm(F[g],axis=2).mean():.4f}" for g in sorted(F)), flush=True)

    acc = {a: np.zeros(3) for a in ARMS}
    nb = {a: 0 for a in ARMS}
    per = {a: [] for a in ARMS}
    t0 = time.time()
    for n, s_ in enumerate(stems):
        mem = [ld(os.path.join(D(m), s_ + ".png")).to(dev) for m in ORDER]
        ens = torch.stack(mem).mean(0)
        with torch.no_grad():
            er = restore(ens, mem, 1.0, len(mem)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        del mem, ens
        g_ = torch.from_numpy(np.asarray(Image.open(os.path.join(gtd, gt_by[s_])).convert("RGB"),
                                         dtype=np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(dev)

        def chain(i):
            a, gain, enc = SPEC[i]
            x = np.clip(warp(er, F[gain], "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **enc)
            by = b.getvalue()
            return len(by), np.asarray(Image.open(io.BytesIO(by)).convert("RGB"),
                                       dtype=np.float32) / 255.

        with ThreadPoolExecutor(max_workers=7) as ex:
            jj = dict(zip(ARMS, ex.map(chain, range(len(SPEC)))))
        for a in ARMS:
            L_, j = jj[a]
            nb[a] += L_
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1. / max(((r - g_) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, g_))
                L = float(vgg(r * 2 - 1, g_ * 2 - 1).item())
            acc[a] += (P, S, L)
            per[a].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.)))
            del r
        del g_
        if n % 15 == 14 or n == len(stems) - 1:
            print(f"  {n+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
            report(per, acc, nb, n + 1, args.out)
    print(f"\nwrote {args.out}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
