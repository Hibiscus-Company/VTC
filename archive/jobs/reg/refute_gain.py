#!/usr/bin/env python
"""INDEPENDENT re-measurement of the "scale the lens field by 1.30" claim.

Differences from the claimant's gainchain.py, on purpose:
  * warp uses the PRODUCTION function  fit_field.apply_field  (repo), not a hand-rolled remap
  * energy restore uses the CANONICAL  energy_restore.restore, not an inline copy
  * adds a NO-FIELD arm (the claimant had no s=0 control, so nothing in his run proves the
    field is even net-positive inside this chain)
  * uses ALL 60 test views, not the first 45 of the sorted list
  * dumps per-view scores so the paired test / bootstrap / subset stability can be checked
"""
import os, sys, io, time, json, argparse
import numpy as np
import torch
import cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, os.path.join("/mnt/c/Users/BKAI/an_plaza2/FastGS", "gsplat_track"))
sys.path.insert(0, HERE)

from fit_field import apply_field as PROD_APPLY_FIELD      # noqa: E402  (production warp)
from energy_restore import restore as PROD_RESTORE         # noqa: E402  (production operator)

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(2)
torch.set_num_threads(2)

JPG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
GAINS = [1.00, 1.15, 1.30, 1.45]

SCENES = {
    "HCM0181": dict(
        mem=[f"/mnt/d/avv/output/HCM0181_{v}/test_poses_renders_png"
             for v in ("gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7")],
        gt="/mnt/d/avv/data/phase1/public_set/HCM0181/test/images",
        flow=f"{HERE}/flow/HCM0181.npz", combiner="r28w", restore=True),
}
for s in ("HCM0193", "HCM0204", "hcm0031", "hcm0034"):
    SCENES[s] = dict(mem=[f"/mnt/d/avv/output/{s}_gsplatB9ut/test_poses_renders_png"],
                     gt=f"/mnt/d/avv/data/phase1/public_set/{s}/test/images",
                     flow=f"{HERE}/flow/{s}.npz", combiner="single", restore=False)


def load(p, dev):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)


def r8(t):
    return (t.clamp(0, 1) * 255.0 + 0.5).floor().clamp(0, 255) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="HCM0181")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--no_lpips", action="store_true",
                    help="PSNR+SSIM only (both tiny on GPU); VGG-LPIPS needs GB of VRAM and "
                         "thrashes into host memory while production training holds the card")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    C = SCENES[args.scene]
    dev = args.dev
    if dev == "cpu":
        torch.set_num_threads(6)

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(C["gt"])}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in C["mem"]))
    stems = stems[:args.n]

    med = np.median(np.load(C["flow"])["ds8"].astype(np.float32), axis=0)
    arms = ["none"] + [f"s{g:.2f}" for g in GAINS]
    FLD = {f"s{g:.2f}": (med * g).astype(np.float32) for g in GAINS}

    from utils.loss_utils import ssim as repo_ssim
    vgg = None
    if not args.no_lpips:
        import lpips as lpips_pkg
        vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    per = {a: [] for a in arms}
    t0 = time.time()
    for c, s in enumerate(stems):
        mem = [load(os.path.join(d, s + ".png"), dev) for d in C["mem"]]
        g = load(os.path.join(C["gt"], gt_by[s]), dev)
        if C["combiner"] == "r28w":                       # r28 tower combiner
            e_p = r8(r8((mem[0] + mem[1] + mem[2]) / 3.0) * 0.8 + 0.2 * mem[3])
        else:
            e_p = r8(mem[0])
        if C["restore"]:
            base = PROD_RESTORE(e_p, mem, args.lam, len(mem)).clamp(0, 1)
        else:
            base = e_p.clamp(0, 1)
        base_u8 = (base[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        del mem, e_p, base
        bf = np.ascontiguousarray(base_u8.astype(np.float32) / 255.0)
        for a in arms:
            x = bf if a == "none" else np.clip(PROD_APPLY_FIELD(bf, FLD[a]), 0, 1)
            u8 = (x * 255.0 + 0.5).astype(np.uint8)
            b = io.BytesIO(); Image.fromarray(u8).save(b, "JPEG", **JPG)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(float(((r - g) ** 2).mean()), 1e-12))
                S = float(repo_ssim(r, g))
                L = float(vgg(r * 2 - 1, g * 2 - 1).item()) if vgg is not None else 0.0
            # with --no_lpips the 4th column is the FIDELITY-ONLY partial score
            # 0.6*PSNR + 30*SSIM, i.e. the blended score holding LPIPS fixed.
            sc = (100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
                  if vgg is not None else 0.6 * P + 30.0 * S)
            per[a].append((P, S, L, sc))
        del g
        if c % 5 == 0:
            print(f"  {c+1}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    n = len(stems)
    A = {a: np.array(per[a]) for a in arms}
    ref = A["s1.00"][:, 3]                                 # PRODUCTION control
    print(f"\n{args.scene}  n={n}  combiner={C['combiner']}  restore={C['restore']} "
          f"lam={args.lam}  (ref = s1.00 = PRODUCTION)")
    print(f"{'arm':>8} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} "
          f"{'d(prod)':>9} {'se':>7} {'t':>7} {'wins':>7}")
    res = {}
    for a in arms:
        m = A[a].mean(0); d = A[a][:, 3] - ref
        se = d.std(ddof=1) / np.sqrt(n) if a != "s1.00" else 0.0
        t = d.mean() / se if se > 0 else 0.0
        res[a] = dict(score=float(m[3]), psnr=float(m[0]), ssim=float(m[1]), lpips=float(m[2]),
                      d=float(d.mean()), se=float(se), t=float(t), win=int((d > 0).sum()),
                      per_view=[float(v) for v in A[a][:, 3]])
        print(f"{a:>8} {m[3]:9.4f} {m[0]:8.4f} {m[1]:8.5f} {m[2]:8.5f} "
              f"{d.mean():+9.4f} {se:7.4f} {t:7.2f} {int((d>0).sum()):3d}/{n}")
    json.dump(dict(scene=args.scene, n=n, lam=args.lam, stems=stems, res=res),
              open(args.out, "w"), indent=1)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
