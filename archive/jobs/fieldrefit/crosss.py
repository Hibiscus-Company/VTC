#!/usr/bin/env python
"""CROSS-SCENE GENERALISATION of field smoothing.  One scene is not a result.

HCM0181 is the only public tower with a multi-member pool at test poses, so the ensemble+restore
harness runs there alone.  The other four public towers each have one render variant
(_gsplatB9ut) and real test GT, which is enough to ask the only question that matters for
shipping: does gaussian-smoothing the fitted field help on EVERY tower, or only on HCM0181?

Chain per arm: single-member render -> warp(field, lanczos4) -> JPEG q100/ss2/optimize/
progressive -> decode -> score.  (No ensemble mean and no energy restore: k=1 here.  The field
operator is the last stage of the chain either way, and it is the only thing that varies.)

Fields come from the production flow cache lens/cache/pub_<scene>.npz (DIS MEDIUM, clip 6, ds=8,
fit on TRAIN photos only), pooled with the shipped MEDIAN estimator, then x1.30 as r29 ships.
Smoothed arms are renormalised to the shipped mean |f| so only SHAPE varies.
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
from fieldlib import upsample, warp                      # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(4)
Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
SCENES = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
FR = os.path.join(HERE, "fieldrefit")
SIGS = [2.0, 3.0, 4.0, 6.0]
ARMS = ["none", "ship"] + [f"sm{s:g}" for s in SIGS]
REF = "ship"


def sc(P, S, L):
    return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(FR, "res_cross.json"))
    args = ap.parse_args()
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()
    gs = lambda f, s: np.stack([cv2.GaussianBlur(f[..., c], (0, 0), s) for c in range(2)], -1)
    mag = lambda f: float(np.linalg.norm(f, axis=2).mean())

    OUT = {}
    for TAG in SCENES:
        gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
        rd = f"/mnt/d/avv/output/{TAG}_gsplatB9ut/test_poses_renders_png"
        gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
        stems = sorted(s for s in gt_by if os.path.exists(os.path.join(rd, s + ".png")))
        if args.limit:
            stems = stems[:args.limit]
        Z = np.load(os.path.join(HERE, "lens", "cache", f"pub_{TAG}.npz"))
        H, W = [int(x) for x in Z["HW"]]
        base = np.median(Z["s8"].astype(np.float32), 0)
        tgt = mag(base) * 1.30
        F = {"none": np.zeros_like(base), "ship": base * 1.30}
        for s in SIGS:
            g_ = gs(base, s)
            F[f"sm{s:g}"] = g_ * (tgt / mag(g_))
        FU = {a: upsample(F[a], H, W, "cubic") for a in ARMS}

        acc = {a: np.zeros(3) for a in ARMS}
        per = {a: [] for a in ARMS}
        t0 = time.time()
        for n, s in enumerate(stems):
            img = np.asarray(Image.open(os.path.join(rd, s + ".png")).convert("RGB"),
                             dtype=np.float32) / 255.0
            g = torch.from_numpy(
                np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                           dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)

            def chain(a):
                x = img if a == "none" else np.clip(warp(img, FU[a], "lanczos"), 0, 1)
                b = io.BytesIO()
                Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
                return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                                  dtype=np.float32) / 255.0

            with ThreadPoolExecutor(max_workers=6) as ex:
                jj = dict(zip(ARMS, ex.map(chain, ARMS)))
            for a in ARMS:
                r = torch.from_numpy(np.ascontiguousarray(jj[a])).permute(2, 0, 1) \
                    .unsqueeze(0).to(dev)
                with torch.no_grad():
                    P = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                    S = float(repo_ssim(r, g))
                    L = float(vgg(r * 2 - 1, g * 2 - 1).item())
                acc[a] += (P, S, L)
                per[a].append(sc(P, S, L))
                del r
            del g
            if n % 20 == 19 or n == len(stems) - 1:
                print(f"  {TAG} {n+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
        N = len(stems)
        res = {a: dict(zip(("psnr", "ssim", "lpips"), (acc[a] / N).tolist())) for a in ARMS}
        for a in ARMS:
            res[a]["score"] = sc(res[a]["psnr"], res[a]["ssim"], res[a]["lpips"])
        OUT[TAG] = {"n": N, "res": res, "per": per}
        json.dump(OUT, open(args.out, "w"), indent=1)
        b = np.array(per[REF])
        print(f"\n== {TAG} (n={N}) single member -> warp -> JPEG q100/ss2 ==")
        print(f"{'arm':>6} {'SCORE':>9} {'vs ship':>9} {'paired t':>9} {'win/N':>7}")
        for a in ARMS:
            v = np.array(per[a]) - b
            t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != REF and v.std() > 0 \
                else float("nan")
            ts = f"{t:9.2f}" if t == t else f"{'--':>9}"
            print(f"{a:>6} {res[a]['score']:9.4f} {res[a]['score']-res[REF]['score']:+9.4f} "
                  f"{ts} {int((v>0).sum()):3d}/{N}", flush=True)

    print("\n=== ACROSS ALL 5 PUBLIC TOWERS (unweighted scene mean, vs ship) ===")
    print(f"{'arm':>6} {'mean d':>9} {'scenes>0':>9}   per-scene")
    for a in ARMS:
        d = [OUT[t]["res"][a]["score"] - OUT[t]["res"][REF]["score"] for t in OUT]
        print(f"{a:>6} {np.mean(d):+9.4f} {sum(x > 0 for x in d):5d}/{len(d)}   "
              + "  ".join(f"{t}:{x:+.4f}" for t, x in zip(OUT, d)))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
