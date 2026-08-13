#!/usr/bin/env python
"""PRODUCTION OPERATOR: restore the finest-band energy that pixel-mean ensembling destroys.

Averaging k members kills local high-frequency energy wherever they disagree. The deficit is not
uniform -- it is exactly the ensemble's own disagreement map. So rescale the FINEST Laplacian band
of the ensemble mean by

    r = sqrt(1 + (k/(k-1)) * mean_{i in S} E(L0_i - L0_mean) / E(L0_mean))      clamped to 4
    out = mean + lam * (r - 1) * L0(mean)

with E(.) a 3x3 box of the RGB-summed square. This is NOT a sharpening filter: the boost lands
where averaging destroyed energy (corr(r, log E_mean) = -0.63), not on strong edges, and the
spatially-shuffled control scores -0.98 while the reverse-ordered control scores -2.49.

k/(k-1) is the unbiased correction for estimating the member spread from deviations about the
mean the members themselves formed. S may be a strict subset: since
    mean_i E_i = E_mean + mean_i E(L_i - L_mean)
one member is enough (measured: DEV-1 +0.1542 vs FULL +0.1505 at lam=0.75 -- no new rendering).

Inserted BETWEEN the ensemble and the lens field. lam is capped by the 350 MB zip, not by score:
added high frequency costs JPEG bytes, and lam>=1.0 does not fit.

Two modes:
  --validate   reproduce the production-harness number on HCM0181 through the full shipped chain
               (mean -> restore -> median field lanczos4 -> JPEG q100/ss2) against real test GT
  --apply      write restored PNGs for a production scene
"""
import argparse, io, os, sys
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K

Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)


def load(p, dev):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)


def restore(ens, members, lam, k, win=3, nlev=5, clamp=4.0):
    """ens [1,3,H,W] float in [0,1]; members list of [1,3,H,W]; k = TRUE member count of ens"""
    K = _K.to(ens.device)
    laps, res, sizes = lap_pyr(ens, nlev, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    if not members:
        raise SystemExit("need at least one member render for the disagreement map")
    V = 0.0
    for m in members:
        ml = lap_pyr(m, nlev, K)[0][0]
        V = V + boxf(((ml - L0) ** 2).sum(1, keepdim=True), win)
    V = V / len(members) * (k / (k - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)
    out = [L0 * (1.0 + lam * (r - 1.0))] + laps[1:]
    return lap_recon(out, res, sizes, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("validate", "apply"), required=True)
    ap.add_argument("--ens_dir")
    ap.add_argument("--member_dirs", nargs="*", default=[])
    ap.add_argument("--out_dir")
    # --- DEADBAND FIX (freeze-day audit) -------------------------------------------------
    # ens_dir holds round(mean)*255 as uint8, i.e. an EXACT INTEGER per pixel. restore() then
    # adds a sub-LSB correction d, and the uint8 write computes round(n + d) = n + round(d),
    # so every pixel with |d| < 0.5 LSB is thresholded to ZERO. Measured on the shipped bytes:
    # 49% of the intended correction destroyed on chair, 79% on bonsai; the towers ship at
    # 0.13-0.58 LSB delivered, inside the damaged regime. Harness cost is 1% at lam=1.0 but
    # 43% at lam=0.25 -- it bites hardest exactly where our private pool operates.
    # Fix: rebuild the SAME mean in float32 from the dirs it was formed from and round ONCE,
    # at the end. No re-rendering needed. Gated below by asserting round(float mean) matches
    # the existing ens_dir png everywhere except exact .5 ties.
    ap.add_argument("--mean_from_dirs", nargs="*", default=[],
                    help="rebuild ens in float from these dirs instead of reading the rounded "
                         "ens_dir png (video: the k member dirs; towers: r22 png_ens, mip3d, "
                         "r28 member with --mean_weights 4 1 1)")
    ap.add_argument("--mean_weights", nargs="*", type=float, default=[],
                    help="weights for --mean_from_dirs (default uniform); normalised internally")
    ap.add_argument("--mean_tol", type=float, default=0.08,
                    help="max fraction of pixels allowed to differ from ens_dir by 1 LSB "
                         "(exact .5 ties only: 1/8 of pixels at k=8, 1/6 at 4:1:1, 0 at k=7)")
    ap.add_argument("--k", type=float, required=True, help="TRUE member count behind ens_dir")
    ap.add_argument("--lam", type=float, default=0.75)
    ap.add_argument("--win", type=int, default=3)
    ap.add_argument("--device", default=None,
                    help="default: cpu for --apply, cuda for --validate. apply defaults to CPU "
                         "because production runs alongside 8M-gaussian training that is already "
                         "using 14 of 16 GB -- a few hundred MB from this would OOM it.")
    args = ap.parse_args()
    dev = args.device or ("cpu" if args.mode == "apply"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.mode == "apply":
        os.makedirs(args.out_dir, exist_ok=True)
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.ens_dir)
                       if f.lower().endswith(".png"))
        for md in args.member_dirs:
            have = {os.path.splitext(f)[0] for f in os.listdir(md)}
            missing = [s for s in stems if s not in have]
            assert not missing, f"{md} missing {len(missing)} stems e.g. {missing[:2]}"
        if args.mean_from_dirs:
            for md in args.mean_from_dirs:
                have = {os.path.splitext(f)[0] for f in os.listdir(md)}
                missing = [s for s in stems if s not in have]
                assert not missing, f"mean_from {md} missing {len(missing)} e.g. {missing[:2]}"
            w = args.mean_weights or [1.0] * len(args.mean_from_dirs)
            assert len(w) == len(args.mean_from_dirs), "mean_weights/mean_from_dirs length"
            w = torch.tensor(w, dtype=torch.float64); w = w / w.sum()
            print(f"DEADBAND FIX: rebuilding ens in float from {len(args.mean_from_dirs)} dir(s) "
                  f"weights={[round(float(x),4) for x in w]}")
        worst = 0.0
        for s in stems:
            if args.mean_from_dirs:
                mm = [load(os.path.join(md, s + ".png"), dev).double()
                      for md in args.mean_from_dirs]
                e = sum(float(wi) * mi for wi, mi in zip(w, mm))
                # GATE: the float mean must reproduce the shipped png_ens everywhere except
                # exact .5 ties, otherwise the dirs/weights do not describe that ensemble.
                ref = load(os.path.join(args.ens_dir, s + ".png"), dev).double()
                q = torch.floor(e * 255.0 + 0.5) / 255.0
                bad = float((torch.abs(q - ref) > 1.5 / 255.0).float().mean())
                diff = float((torch.abs(q - ref) > 0.5 / 255.0).float().mean())
                assert bad == 0.0, (f"{s}: {bad:.4%} of pixels differ from ens_dir by >1 LSB -- "
                                    f"--mean_from_dirs/--mean_weights do not describe this ensemble")
                assert diff <= args.mean_tol, (f"{s}: {diff:.4%} of pixels differ by 1 LSB, "
                                               f"above --mean_tol {args.mean_tol:.2%}")
                worst = max(worst, diff)
                e = e.float()
            else:
                e = load(os.path.join(args.ens_dir, s + ".png"), dev)
            ms = [load(os.path.join(md, s + ".png"), dev) for md in args.member_dirs]
            o = restore(e, ms, args.lam, args.k, args.win).clamp(0, 1)
            a = (o[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(a).save(os.path.join(args.out_dir, s + ".png"))
        print(f"restored {len(stems)} images (lam={args.lam}, k={args.k}, "
              f"{len(args.member_dirs)} member(s)"
              + (f", FLOAT MEAN, worst tie-frac {worst:.4%}" if args.mean_from_dirs else "")
              + f") -> {args.out_dir}")
        return

    # ---------------- validate on the production harness, full shipped chain ----------------
    from utils.loss_utils import ssim as repo_ssim
    from fieldlib import LooPool, upsample, warp
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    tag = "HCM0181"
    MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
           "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
           "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
           "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
    gtd = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    cache = np.load(f"{HERE}/lens/cache/pub_{tag}.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"),
                    *[int(x) for x in cache["HW"]], "cubic")

    def enc(x):
        b = io.BytesIO()
        Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            b, "JPEG", **SHIPPED_JPEG)
        return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                          dtype=np.float32) / 255.0

    arms = ["base"] + [f"lam{l}" for l in (0.5, 0.75, 1.0)] + ["dev1_lam0.75"]
    acc = {a: [0.0, 0.0, 0.0] for a in arms}
    nbytes = {a: 0 for a in arms}
    for c, s in enumerate(stems):
        mem = [load(os.path.join(d, s + ".png"), dev) for d in MEM]
        ens = torch.stack(mem).mean(0)
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                                        dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        outs = {"base": ens}
        for l in (0.5, 0.75, 1.0):
            outs[f"lam{l}"] = restore(ens, mem, l, len(mem), args.win)
        outs["dev1_lam0.75"] = restore(ens, [mem[0]], 0.75, len(mem), args.win)
        for a in arms:
            x = outs[a].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            x = np.clip(warp(x, lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
            nbytes[a] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                acc[a][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[a][1] += float(repo_ssim(r, g))
                acc[a][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
        if c % 15 == 0:
            print(f"  {c}/{len(stems)}", flush=True)

    n = len(stems)
    print(f"\nFULL SHIPPED CHAIN (mean -> restore -> median field lanczos4 -> JPEG q100/ss2), "
          f"{tag}, n={n}")
    print(f"{'arm':>14} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'vs base':>9} {'MB/60':>8}")
    base = None
    for a in arms:
        P, S, L = (x / n for x in acc[a])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        if a == "base":
            base = sc
        print(f"{a:>14} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {sc-base:+9.4f} "
              f"{nbytes[a]/1e6:8.2f}")


if __name__ == "__main__":
    main()
