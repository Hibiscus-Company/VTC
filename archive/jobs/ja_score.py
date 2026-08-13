#!/usr/bin/env python
"""jpegalloc STAGE 2: score-vs-bytes of the encode ladder, PRODUCTION HARNESS, FULL SHIPPED CHAIN.

Chain per image (hoisted -- computed ONCE, then encoded 11 ways):
    4-member pixel mean -> energy restore (lam, k=4) -> median lens field x1.30, INTER_LANCZOS4
    -> [ENCODE ARM] -> decode -> PSNR / repo SSIM / LPIPS-vgg vs REAL test GT

CONTROL: arm "png" is bit-exact lossless. It is the only arm with zero encode loss, so
(arm - png) isolates what the encoder does, and (q100ss0 - png) is the pure DCT-rounding term
while (q100ss2 - q100ss0) is the pure chroma-subsampling term. Bytes are counted for every arm
including png, so every arm is a point on the score-vs-bytes curve.
"""
import argparse, io, os, sys, json, time
import numpy as np
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from energy_restore import restore
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIP = dict(optimize=True, progressive=True)

ARMS = [("q100ss2", 100, 2), ("q99ss2", 99, 2), ("q98ss2", 98, 2), ("q97ss2", 97, 2),
        ("q100ss1", 100, 1), ("q99ss1", 99, 1), ("q98ss1", 98, 1),
        ("q100ss0", 100, 0), ("q99ss0", 99, 0), ("q98ss0", 98, 0), ("q97ss0", 97, 0),
        ("q96ss0", 96, 0), ("png", None, None)]

MEMS = {
    "HCM0181": ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
                "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
                "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
                "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"],
    "HCM0193": ["/mnt/d/avv/output/HCM0193_gsplatB9ut/test_poses_renders_png"],
    "HCM0204": ["/mnt/d/avv/output/HCM0204_gsplatB9ut/test_poses_renders_png"],
    "hcm0031": ["/mnt/d/avv/output/hcm0031_gsplatB9ut/test_poses_renders_png"],
    "hcm0034": ["/mnt/d/avv/output/hcm0034_gsplatB9ut/test_poses_renders_png"],
}


def enc_dec(x, q, ss):
    """x float HxWx3 in [0,1] -> (decoded float, nbytes)."""
    im = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
    b = io.BytesIO()
    if q is None:
        im.save(b, "PNG", optimize=True)
    else:
        im.save(b, "JPEG", quality=q, subsampling=ss, **SHIP)
    raw = b.getvalue()
    d = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(d), len(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="HCM0181")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=1.30)
    ap.add_argument("--field", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    tag = args.scene
    MEM = [d for d in MEMS[tag] if os.path.isdir(d)]
    gtd = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    if args.limit:
        stems = stems[::max(1, len(stems) // args.limit)][:args.limit]
    print(f"{tag}: {len(MEM)} members, {len(stems)} poses, lam={args.lam} "
          f"field={args.field} gain={args.gain}", flush=True)

    lens = None
    if args.field:
        c = np.load(f"{HERE}/lens/cache/pub_{tag}.npz")
        lens = upsample(LooPool(c["s8"]).pooled("median"),
                        *[int(x) for x in c["HW"]], "cubic") * args.gain

    acc = {a: [0.0, 0.0, 0.0] for a, _, _ in ARMS}
    nb = {a: 0 for a, _, _ in ARMS}
    per = {a: [] for a, _, _ in ARMS}   # per-image (P,S,L) for PAIRED statistics
    t0 = time.time()
    for ci, s in enumerate(stems):
        mem = []
        for d in MEM:
            a = np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                           dtype=np.float32) / 255.0
            mem.append(torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev))
        ens = torch.stack(mem).mean(0)
        # ---- hoisted: operator + field computed ONCE for all arms ----
        if len(mem) > 1 and args.lam > 0:
            ens = restore(ens, mem, args.lam, len(mem), 3)
        x = ens.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        if lens is not None:
            x = np.clip(warp(x, lens, "lanczos"), 0, 1)
        g = torch.from_numpy(
            np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                       dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        for name, q, ss in ARMS:
            d, n = enc_dec(x, q, ss)
            nb[name] += n
            r = torch.from_numpy(d).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                p = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                ssv = float(repo_ssim(r, g))
                lv = float(vgg(r * 2 - 1, g * 2 - 1).item())
            acc[name][0] += p; acc[name][1] += ssv; acc[name][2] += lv
            per[name].append((p, ssv, lv))
        if ci % 10 == 0:
            print(f"  {ci}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    n = len(stems)
    res = {}
    print(f"\n=== {tag} FULL SHIPPED CHAIN, n={n}, lam={args.lam}, field={args.field} ===")
    print(f"{'arm':>9} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} "
          f"{'d_vs_q100ss2':>13} {'MB/scene':>9} {'dMB':>8} {'d/MB':>9}")
    base = bb = None
    for name, _, _ in ARMS:
        P, S, L = (v / n for v in acc[name])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        mb = nb[name] / 1e6
        if name == "q100ss2":
            base, bb = sc, mb
        dm = mb - bb
        res[name] = dict(score=sc, psnr=P, ssim=S, lpips=L, mb=mb, d=sc - base, dmb=dm)
        rate = (sc - base) / dm if abs(dm) > 1e-9 else float("nan")
        print(f"{name:>9} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {sc-base:+13.4f} "
              f"{mb:9.2f} {dm:+8.2f} {rate:+9.4f}")
    # -------- PAIRED statistics: same source pixels in every arm, so difference-of-means
    # has far smaller variance than either mean. t on the per-image score difference.
    A = {a: np.asarray(per[a]) for a, _, _ in ARMS}
    def img_score(v):
        return 100 * (0.4 * (1 - v[:, 2]) + 0.3 * v[:, 1] + 0.3 * np.minimum(v[:, 0] / 50, 1))
    b = img_score(A["q100ss2"])
    print(f"\nPAIRED vs q100ss2 (n={n})")
    print(f"{'arm':>9} {'mean_d':>9} {'se':>8} {'t':>7} {'wins':>7} "
          f"{'dPSNR':>8} {'dSSIM':>9} {'dLPIPS':>10}")
    for name, _, _ in ARMS:
        if name == "q100ss2":
            continue
        d = img_score(A[name]) - b
        se = d.std(ddof=1) / np.sqrt(n)
        dd = A[name] - A["q100ss2"]
        res[name]["se"] = se
        res[name]["t"] = d.mean() / se if se > 0 else 0.0
        print(f"{name:>9} {d.mean():+9.4f} {se:8.4f} {d.mean()/se if se>0 else 0:7.2f} "
              f"{int((d>0).sum()):>4}/{n:<3} {dd[:,0].mean():+8.4f} {dd[:,1].mean():+9.5f} "
              f"{dd[:,2].mean():+10.5f}")
    json.dump(res, open(f"{HERE}/ja_score_{tag}{args.tag}.json", "w"), indent=1)
    np.savez(f"{HERE}/ja_per_{tag}{args.tag}.npz", **{a: A[a] for a, _, _ in ARMS})
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
