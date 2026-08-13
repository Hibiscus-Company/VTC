#!/usr/bin/env python
"""FREEBIE HUNT: encode + intermediate-quantization arms, production harness, full shipped chain.

Chain replicated exactly as r29 ships it for a tower:
    members(PNG u8) -> weighted mean(float32) -> ROUND to u8 (ensemble_renders.py --png_dir)
      -> energy_restore(lam=1.0) -> ROUND to u8 (energy_restore --apply)
      -> apply_field(lanczos4, field*1.30) -> ROUND to u8 (apply_field.py)
      -> PIL JPEG quality=100 subsampling=2 optimize=True progressive=True

ARMS
  prod_*      : identical pixels, different JPEG encoder settings  (Q sweep, subsampling, keep_rgb,
                progressive/optimize)
  b1/b2/b3    : the SAME encoder settings as the baseline, with one/both intermediate uint8
                round-trips removed (b3 = single quantization, at the JPEG boundary only)
Baseline arm = prod_q100ss2 = exactly what ships.
"""
import argparse, io, os, sys, json, time
import numpy as np
import torch
from PIL import Image, ImageFile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K          # noqa: E402
from fieldlib import LooPool, upsample, warp              # noqa: E402
from energy_restore import restore                        # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ImageFile.MAXBLOCK = 1 << 26   # PIL default 65k buffer overflows on q100/ss0 tiles
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)

POOLS = {
    "HCM0181": ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
                "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
                "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
                "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"],
    "HCM0193": ["/mnt/d/avv/output/HCM0193_gsplatB9ut/test_poses_renders_png"],
    "HCM0204": ["/mnt/d/avv/output/HCM0204_gsplatB9ut/test_poses_renders_png"],
    "hcm0031": ["/mnt/d/avv/output/hcm0031_gsplatB9ut/test_poses_renders_png"],
    "hcm0034": ["/mnt/d/avv/output/hcm0034_gsplatB9ut/test_poses_renders_png"],
}

# (name, source, jpeg kwargs)  source in {prod,b1,b2,b3}
def build_arms(full):
    A = [("prod_q100ss2", "prod", dict(SHIPPED)),
         ("prod_q99ss2",  "prod", dict(SHIPPED, quality=99)),
         ("prod_q98ss2",  "prod", dict(SHIPPED, quality=98)),
         ("prod_q96ss2",  "prod", dict(SHIPPED, quality=96)),
         ("prod_q100ss1", "prod", dict(SHIPPED, subsampling=1)),
         ("prod_q100ss0", "prod", dict(SHIPPED, subsampling=0)),
         ("prod_rgb",     "prod", dict(SHIPPED, subsampling=0, keep_rgb=True)),
         ("prod_noprog",  "prod", dict(SHIPPED, progressive=False)),
         ("prod_noopt",   "prod", dict(SHIPPED, optimize=False)),
         ]
    if full:
        A += [("b1_noensround", "b1", dict(SHIPPED)),
              ("b2_noerround",  "b2", dict(SHIPPED)),
              ("b3_allfloat",   "b3", dict(SHIPPED))]
    return A


def u8(x):
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="HCM0181")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=1.30)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    tag = args.scene
    MEM = POOLS[tag]
    full = len(MEM) > 1                       # only the 4-member pool has an ensemble/ER stage
    arms = build_arms(full)
    gtd = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    if args.n:
        stems = stems[:args.n]

    cache = np.load(f"{HERE}/lens/cache/pub_{tag}.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"),
                    *[int(x) for x in cache["HW"]], "cubic") * args.gain

    acc = {a[0]: np.zeros(3) for a in arms}
    nb = {a[0]: 0 for a in arms}
    per = {a[0]: [] for a in arms}
    identical = {}
    t0 = time.time()
    for c, s in enumerate(stems):
        mem_np = [np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                             dtype=np.float32) / 255.0 for d in MEM]
        mem = [torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev) for a in mem_np]
        ens_f = torch.stack(mem).mean(0)                       # float mean, never written
        ens_q = torch.from_numpy(u8(ens_f[0].permute(1, 2, 0).cpu().numpy()).astype(np.float32)
                                 / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)   # png_ens

        srcs = {}
        if full:
            er_from_q = restore(ens_q, mem, args.lam, len(mem)).clamp(0, 1)   # production
            er_from_f = restore(ens_f, mem, args.lam, len(mem)).clamp(0, 1)
            eq = er_from_q[0].permute(1, 2, 0).cpu().numpy()
            ef = er_from_f[0].permute(1, 2, 0).cpu().numpy()
            prod_pre = u8(eq).astype(np.float32) / 255.0        # png_er (production round #2)
            srcs["prod"] = u8(np.clip(warp(prod_pre, lens, "lanczos"), 0, 1))
            srcs["b1"] = u8(np.clip(warp(u8(ef).astype(np.float32) / 255.0, lens, "lanczos"), 0, 1))
            srcs["b2"] = u8(np.clip(warp(eq, lens, "lanczos"), 0, 1))
            srcs["b3"] = u8(np.clip(warp(ef, lens, "lanczos"), 0, 1))
        else:
            base = ens_q[0].permute(1, 2, 0).cpu().numpy()
            srcs["prod"] = u8(np.clip(warp(base, lens, "lanczos"), 0, 1))

        g = torch.from_numpy(
            np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                       dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)

        ref_dec = None
        for name, src, kw in arms:
            b = io.BytesIO()
            Image.fromarray(srcs[src]).save(b, "JPEG", **kw)
            raw = b.getvalue()
            nb[name] += len(raw)
            j = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255.0
            if name == "prod_q100ss2":
                ref_dec = j
            elif name in ("prod_noprog", "prod_noopt"):
                identical[name] = identical.get(name, True) and bool((j == ref_dec).all())
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, g))
                L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            acc[name] += (P, S, L)
            per[name].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
        if c % 10 == 0:
            print(f"  {c}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    n = len(stems)
    base = np.array(per["prod_q100ss2"])
    res = {}
    print(f"\n=== {tag}  n={n}  full_chain={full}  lam={args.lam} gain={args.gain} ===")
    print(f"{'arm':>16} {'SCORE':>9} {'dSCORE':>9} {'t':>6} {'wins':>7} "
          f"{'PSNR':>8} {'SSIM':>7} {'LPIPS':>7} {'MB':>7} {'dMB%':>7}")
    for name, _, _ in arms:
        P, S, L = acc[name] / n
        sc = float(np.mean(per[name]))
        d = np.array(per[name]) - base
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n)) if d.std(ddof=1) > 1e-12 else 0.0
        res[name] = dict(score=sc, d=sc - base.mean(), t=float(t),
                         wins=int((d > 0).sum()), psnr=P, ssim=S, lpips=L,
                         bytes=nb[name], dbytes_pct=100.0 * (nb[name] - nb["prod_q100ss2"])
                         / nb["prod_q100ss2"])
        print(f"{name:>16} {sc:9.4f} {sc-base.mean():+9.4f} {t:6.2f} {res[name]['wins']:3d}/{n:<3d} "
              f"{P:8.4f} {S:7.4f} {L:7.4f} {nb[name]/1e6:7.2f} {res[name]['dbytes_pct']:+7.2f}")
    print("\npixel-identity to baseline:", identical)
    json.dump(dict(scene=tag, n=n, full=full, res=res, identical=identical),
              open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
