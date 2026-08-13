#!/usr/bin/env python
"""FIELD SCALE THROUGH THE r28 CHAIN.

WHY. The field is fit on TRAIN renders against TRAIN photos, but applied at TEST poses. The model
is fit to those train photos, so its train renders are BETTER registered than its test renders --
the train-fitted field therefore UNDERSHOOTS the test displacement. Two prior sweeps agree:
  res/k4scale.json  HCM0181 k4 ensemble, real test GT, shipped JPEG, lanczos4 warp, NO energy
                    restore:  s1.0 +1.5063, s1.15 +1.5969, s1.30 +1.6389, s1.45 +1.6332
                    -> s1.30 is +0.1326 over the shipped s1.0
  res/scale2.json   the other public towers, single member, PNG: optimum 1.30-1.45, 5/5 positive
while the GT-free LOO-TRAIN selection rule (looscale.py) picks 1.00 on 5/5 public AND 5/5 private
towers -- because it evaluates the field on the very train photos it was fit against, which is the
regime whose registration is already tight. That disagreement is why 1.0 ships.

WHAT IS NEW HERE. Every one of those numbers predates r28's ENERGY RESTORE, which amplifies the
finest Laplacian band before the warp. Misregistered high frequency costs more than misregistered
low frequency, so the restore operator can move the scale optimum -- and it sits between the
ensemble and the field in the shipped chain, so this is the only regime that decides r29.

Chain per arm: k4 mean -> restore(lam=1.0,k=4) -> warp(field*s, lanczos4) -> JPEG q100/ss2.
The mean and the restore are shared across arms and computed once per image.
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
from energy_restore import restore
from fieldlib import upsample, warp

cv2.setNumThreads(1)
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
FR = os.path.join(HERE, "fieldrefit")
D = lambda m: f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"
POOL = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
SCALES = [1.0, 1.30, 1.45]   # 3 arms only: the GPU is shared with 18 other processes


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
    print(f"\nFIELD SCALE THROUGH THE r28 CHAIN, {TAG}, n={N}")
    print("(k4 mean -> energy restore lam=1.0 -> warp field*s lanczos4 -> JPEG q100/ss2)")
    print(f"{'arm':>8} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs none':>8} {'vs s1.0':>8} {'paired t':>9} {'win/N':>7}")
    b = np.array(per["s1.0"])
    for a in arms:
        v = np.array(per[a]) - b
        t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != "s1.0" and v.std() > 0 else float("nan")
        ts = f"{t:9.2f}" if t == t else f"{'--':>9}"
        r_ = res[a]
        print(f"{a:>8} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:7.4f} {r_['lpips']:8.4f} "
              f"{'':>8} {r_['score']-res['s1.0']['score']:+8.4f} "
              f"{ts} {int((v>0).sum()):3d}/{N}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(FR, "res_scale.json"))
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

    # the PRODUCTION field: median ds8 pooled over the B9ut member's train renders. Verified
    # elsewhere in this job to be byte-identical to lens/cache/pub_HCM0181.npz.
    cache = np.load(os.path.join(HERE, "lens", "cache", f"pub_{TAG}.npz"))
    H, W = [int(x) for x in cache["HW"]]
    base = upsample(np.median(cache["s8"].astype(np.float32), 0), H, W, "cubic")
    F = {f"s{s:g}": base * s for s in SCALES}
    arms = [f"s{s:g}" for s in SCALES]

    acc = {a: np.zeros(3) for a in arms}
    per = {a: [] for a in arms}
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
            x = er if a == "none" else np.clip(warp(er, F[a], "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                              dtype=np.float32) / 255.0

        with ThreadPoolExecutor(max_workers=5) as ex:
            jj = dict(zip(arms, ex.map(chain, arms)))
        for a in arms:
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
            report(arms, per, acc, n + 1, args.out, final=False)
    report(arms, per, acc, len(stems), args.out, final=True)
    print(f"\nwrote {args.out}   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
