#!/usr/bin/env python
"""PHASE B: a LEVEL-1 energy-restore term with its own lambda, scored on the production harness
through the FULL SHIPPED CHAIN (mean -> restore -> median lens field lanczos4 -> JPEG q100/ss2/
optimize/progressive) against real HCM0181 test GT.

Shared computation (member Laplacian pyramids, the level-0 and level-1 r maps) is computed ONCE
per image and reused by every arm.  Pyramid/warp/JPEG on CPU; only LPIPS-VGG touches the GPU.

Arms
  base            k-member pixel mean, no operator
  L0              shipped operator, level 0 only, lam=1.0                    <- REFERENCE
  L0+L1 mu=x      shipped level 0 plus a level-1 term with gain mu
  L1only mu=1.0   level-1 term ALONE                                          <- isolation control
  L0+L1const      level-1 term with the r1 map replaced by its own scalar mean <- geography control
"""
import os, sys, io, json, argparse
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.dirname(HERE)
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, JOB)
sys.path.insert(0, os.path.join(JOB, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(4)

SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
ROOT = "/mnt/d/avv/output"
POOL8 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
         "m31b_nolpips", "e17visnorm", "gsplatB8pure", "gsplatB4warm"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
NLEV, WIN, CLAMP = 5, 3, 4.0


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def rmap(Lmean, Lmems, k, win=WIN, clamp=CLAMP):
    Eb = boxf((Lmean ** 2).sum(1, keepdim=True), win)
    V = 0.0
    for ml in Lmems:
        V = V + boxf(((ml - Lmean) ** 2).sum(1, keepdim=True), win)
    V = V / len(Lmems) * (k / (k - 1.0))
    return torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--nimg", type=int, default=60)
    ap.add_argument("--tag", default="k8")
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    from fieldlib import LooPool, upsample, warp
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    pool = POOL8[:args.k]
    dirs = [os.path.join(ROOT, "HCM0181_" + v, "test_poses_renders_png") for v in pool]
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
    stems = stems[:args.nimg]
    cache = np.load(os.path.join(JOB, "lens/cache/pub_HCM0181.npz"))
    lens = upsample(LooPool(cache["s8"]).pooled("median"),
                    *[int(x) for x in cache["HW"]], "cubic")

    ARMS = [("base", 0.0, 0.0, "raw"),
            ("L0", 1.0, 0.0, "raw"),
            ("L0+L1_0.5", 1.0, 0.5, "raw"),
            ("L0+L1_1.0", 1.0, 1.0, "raw"),
            ("L0+L1_2.0", 1.0, 2.0, "raw"),
            ("L1only_1.0", 0.0, 1.0, "raw"),
            ("L0+L1c_1.0", 1.0, 1.0, "const")]
    acc = {a[0]: [0.0, 0.0, 0.0] for a in ARMS}
    nb = {a[0]: 0 for a in ARMS}
    print(f"k={args.k} pool={pool}\nn={len(stems)} arms={[a[0] for a in ARMS]}", flush=True)

    for c, st in enumerate(stems):
        # ---- shared computation, ONCE per image ----
        mlaps, mres, msz = [], [], None
        for d in dirs:
            L, R, S = lap_pyr(load(os.path.join(d, st + ".png")), NLEV, _K)
            mlaps.append(L); mres.append(R); msz = S
        laps = [torch.stack([m[l] for m in mlaps]).mean(0) for l in range(NLEV)]
        res = torch.stack(mres).mean(0)
        r0 = rmap(laps[0], [m[0] for m in mlaps], args.k)
        r1 = rmap(laps[1], [m[1] for m in mlaps], args.k)
        r1c = r1.mean().expand_as(r1)
        del mlaps, mres
        g = load(os.path.join(GTD, gt_by[st])).to(dev)

        for name, lam0, mu1, m1 in ARMS:
            out = list(laps)
            if lam0:
                out[0] = laps[0] * (1.0 + lam0 * (r0 - 1.0))
            if mu1:
                rr = r1c if m1 == "const" else r1
                out[1] = laps[1] * (1.0 + mu1 * (rr - 1.0))
            x = lap_recon(out, res, msz, _K).clamp(0, 1)[0].permute(1, 2, 0).numpy()
            x = np.clip(warp(x, lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
            nb[name] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float32) / 255.
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                acc[name][0] += 10 * np.log10(1.0 / max(float(((r - g) ** 2).mean()), 1e-12))
                acc[name][1] += float(repo_ssim(r, g))
                acc[name][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
            del r
        del g, laps, res, r0, r1, r1c
        if c % 5 == 0:
            print(f"  {c}/{len(stems)}", flush=True)

    n = len(stems)
    print(f"\nFULL SHIPPED CHAIN, HCM0181 real test GT, k={args.k}, n={n}")
    print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs base':>9} {'vs L0':>8} {'MB/60':>7}")
    sc = {}
    for name, *_ in ARMS:
        P, S, L = (x / n for x in acc[name])
        sc[name] = (100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)), P, S, L)
    for name, *_ in ARMS:
        s, P, S, L = sc[name]
        print(f"{name:>12} {s:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} "
              f"{s-sc['base'][0]:+9.4f} {s-sc['L0'][0]:+8.4f} {nb[name]/1e6:7.2f}")
    json.dump({k: dict(score=v[0], psnr=v[1], ssim=v[2], lpips=v[3], mb=nb[k] / 1e6)
               for k, v in sc.items()},
              open(os.path.join(HERE, f"p2_{args.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
