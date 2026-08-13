#!/usr/bin/env python
"""PHASE B (parallel, CPU-ONLY -- does not touch the GPU, which is running production training).

Same measurement as p2_chain.py: level-1 energy-restore term with its own gain, scored on the
production harness (HCM0181 real test poses vs real test GT) through the FULL SHIPPED CHAIN
    k-member pixel mean -> restore -> median lens field warped INTER_LANCZOS4 -> JPEG q100/ss2
Per-image work (pyramids, r maps, all arms, warp, JPEG round-trip, PSNR/SSIM/LPIPS-vgg) runs in a
worker process; shared computation is hoisted so every arm reuses ONE set of member pyramids and
ONE pair of r maps.
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
Image.MAX_IMAGE_PIXELS = None

SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
ROOT = "/mnt/d/avv/output"
POOL8 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
         "m31b_nolpips", "e17visnorm", "gsplatB8pure", "gsplatB4warm"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
NLEV, WIN, CLAMP = 5, 3, 4.0

ARMS = [("base", 0.0, 0.0, "raw"),
        ("lam0.75", 0.75, 0.0, "raw"),
        ("lam1.00", 1.00, 0.0, "raw"),
        ("lam1.25", 1.25, 0.0, "raw"),
        ("lam1.50", 1.50, 0.0, "raw")]

_G = {}


def _init(k, hw, s8):
    torch.set_num_threads(int(os.environ.get("WT", "3")))
    from utils.loss_utils import ssim as repo_ssim
    from fieldlib import LooPool, upsample, warp
    from lapfuse import lap_pyr, lap_recon, boxf, _K
    import lpips as lpips_pkg
    _G.update(k=k, ssim=repo_ssim, warp=warp,
              lens=upsample(LooPool(s8).pooled("median"), hw[0], hw[1], "cubic"),
              lap_pyr=lap_pyr, lap_recon=lap_recon, boxf=boxf, K=_K,
              vgg=lpips_pkg.LPIPS(net="vgg", verbose=False).eval())
    _G["dirs"] = [os.path.join(ROOT, "HCM0181_" + v, "test_poses_renders_png") for v in POOL8[:k]]


def _load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def _rmap(Lmean, Lmems, k):
    boxf = _G["boxf"]
    Eb = boxf((Lmean ** 2).sum(1, keepdim=True), WIN)
    V = 0.0
    for ml in Lmems:
        V = V + boxf(((ml - Lmean) ** 2).sum(1, keepdim=True), WIN)
    V = V / len(Lmems) * (k / (k - 1.0))
    return torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=CLAMP)


def work(job):
    st, gtf = job
    k = _G["k"]
    lap_pyr, lap_recon, K = _G["lap_pyr"], _G["lap_recon"], _G["K"]
    mlaps, mres, msz = [], [], None
    for d in _G["dirs"]:
        L, R, S = lap_pyr(_load(os.path.join(d, st + ".png")), NLEV, K)
        mlaps.append(L); mres.append(R); msz = S
    laps = [torch.stack([m[l] for m in mlaps]).mean(0) for l in range(NLEV)]
    res = torch.stack(mres).mean(0)
    r0 = _rmap(laps[0], [m[0] for m in mlaps], k)
    r1 = _rmap(laps[1], [m[1] for m in mlaps], k)
    r1c = r1.mean().expand_as(r1)
    del mlaps, mres
    g = _load(os.path.join(GTD, gtf))
    out = {}
    for name, lam0, mu1, m1 in ARMS:
        o = list(laps)
        if lam0:
            o[0] = laps[0] * (1.0 + lam0 * (r0 - 1.0))
        if mu1:
            o[1] = laps[1] * (1.0 + mu1 * ((r1c if m1 == "const" else r1) - 1.0))
        x = lap_recon(o, res, msz, K).clamp(0, 1)[0].permute(1, 2, 0).numpy()
        x = np.clip(_G["warp"](x, _G["lens"], "lanczos"), 0, 1)
        b = io.BytesIO()
        Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
        nby = len(b.getvalue())
        j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float32) / 255.
        r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            P = 10 * np.log10(1.0 / max(float(((r - g) ** 2).mean()), 1e-12))
            S = float(_G["ssim"](r, g))
            L = float(_G["vgg"](r * 2 - 1, g * 2 - 1).item())
        out[name] = (P, S, L, nby)
    return st, out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--nimg", type=int, default=24)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="k8")
    args = ap.parse_args()

    dirs = [os.path.join(ROOT, "HCM0181_" + v, "test_poses_renders_png") for v in POOL8[:args.k]]
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
    jobs = [(s, gt_by[s]) for s in stems[args.skip:args.skip + args.nimg]]
    cache = np.load(os.path.join(JOB, "lens/cache/pub_HCM0181.npz"))
    hw = [int(x) for x in cache["HW"]]
    print(f"k={args.k} n={len(jobs)} workers={args.workers} arms={[a[0] for a in ARMS]}", flush=True)

    def score(P, S, L):
        return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))

    per = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.k, hw, cache["s8"])) as pool:
        for c, (st, o) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            per[st] = o
            run = {a: np.mean([score(*per[s][a][:3]) for s in per]) for a, *_ in ARMS}
            print(f"  {c+1}/{len(jobs)} {st}  " +
                  "  ".join(f"{a}:{run[a]-run['lam1.00']:+.4f}" for a, *_ in ARMS), flush=True)
            json.dump({"n": len(per),
                       "per_image": {s: {a: list(v) for a, v in per[s].items()} for s in per}},
                      open(os.path.join(HERE, f"p3_{args.tag}_partial.json"), "w"))

    n = len(per)
    print(f"\nFULL SHIPPED CHAIN (mean -> restore -> median field lanczos4 -> JPEG q100/ss2), "
          f"HCM0181 real test GT, k={args.k}, n={n}")
    print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs base':>9} {'vs L0':>9} {'paired se':>10} {'MB/60':>7}")

    agg, pim = {}, {}
    for name, *_ in ARMS:
        P = np.mean([per[s][name][0] for s in per])
        S = np.mean([per[s][name][1] for s in per])
        L = np.mean([per[s][name][2] for s in per])
        agg[name] = (score(P, S, L), P, S, L, sum(per[s][name][3] for s in per) / 1e6 * (60.0 / n))
        pim[name] = np.array([score(*per[s][name][:3]) for s in per])
    for name, *_ in ARMS:
        s, P, S, L, mb = agg[name]
        d = pim[name] - pim["lam1.00"]
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        print(f"{name:>12} {s:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} "
              f"{s-agg['base'][0]:+9.4f} {s-agg['lam1.00'][0]:+9.4f} {se:10.4f} {mb:7.2f}")
    json.dump({"agg": {k: list(v) for k, v in agg.items()},
               "per_image": {s: {a: list(v) for a, v in per[s].items()} for s in per}},
              open(os.path.join(HERE, f"p3_{args.tag}.json"), "w"))


if __name__ == "__main__":
    main()
