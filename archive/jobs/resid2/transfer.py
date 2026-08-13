#!/usr/bin/env python
"""DOES THE FIELD-SCALE WIN TRANSFER OFF HCM0181, AND WHICH CONSTANT SHOULD SHIP?

The scale correction has only ever been scored on HCM0181, whose own oracle projection is
alpha*=1.3185 -- i.e. the one scene where the hand-picked 1.30 is already right.  The other four
public towers project at 1.352 / 1.413 / 1.431 / 1.339 (reg/alpha_scenes.json), so a single
shipped constant is an extrapolation and needs a transfer control on scenes it was not read off.

These four scenes have exactly one render variant each, so the ensemble mean is the member and
the energy restore is the identity (its deviation term is zero for k=1); the chain below is
therefore the full shipped chain for a one-member pool:  render -> warp(field, lanczos4) ->
JPEG q100/ss2 optimize progressive -> decode -> score.

Arms: the shipped unscaled field, the constant the leaderboard candidate uses (1.30), and the
energy-weighted population optimum over the five public alphas (1.37).
"""
import io, os, sys, time, json, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
OUT = os.path.join(HERE, "resid2")
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, os.path.join(HERE, "lens"))
from fieldlib import upsample, warp                                    # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(3)
Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
SCALES = [1.00, 1.30, 1.37]
ARMS = [f"s{c:.2f}" for c in SCALES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="HCM0193,HCM0204,hcm0031,hcm0034")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(OUT, "res_transfer.json"))
    args = ap.parse_args()
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    allres, t0 = {}, time.time()
    for tag in args.scenes.split(","):
        rd = f"/mnt/d/avv/output/{tag}_gsplatB9ut/test_poses_renders_png"
        gtd = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
        gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
        stems = sorted(s for s in gt_by if os.path.exists(os.path.join(rd, s + ".png")))
        if args.limit:
            stems = stems[:args.limit]
        f1 = np.median(np.load(os.path.join(HERE, "flow", f"{tag}.npz"))["ds8"]
                       .astype(np.float32), 0)
        probe = np.asarray(Image.open(os.path.join(rd, stems[0] + ".png")).convert("RGB"))
        H, W, _ = probe.shape
        F = {a: upsample(f1 * c, H, W, "cubic") for a, c in zip(ARMS, SCALES)}
        acc = {a: np.zeros(3) for a in ARMS}
        per = {a: [] for a in ARMS}
        for n, s_ in enumerate(stems):
            r = np.asarray(Image.open(os.path.join(rd, s_ + ".png")).convert("RGB"),
                           dtype=np.float32) / 255.0
            g = torch.from_numpy(
                np.asarray(Image.open(os.path.join(gtd, gt_by[s_])).convert("RGB"),
                           dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)

            def chain(a):
                x = np.clip(warp(r, F[a], "lanczos"), 0, 1)
                b = io.BytesIO()
                Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
                return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                                  dtype=np.float32) / 255.0

            with ThreadPoolExecutor(max_workers=3) as ex:
                jj = dict(zip(ARMS, ex.map(chain, ARMS)))
            for a in ARMS:
                y = torch.from_numpy(np.ascontiguousarray(jj[a])).permute(2, 0, 1) \
                    .unsqueeze(0).to(dev)
                with torch.no_grad():
                    P = 10 * np.log10(1.0 / max(((y - g) ** 2).mean().item(), 1e-12))
                    S = float(repo_ssim(y, g))
                    L = float(vgg(y * 2 - 1, g * 2 - 1).item())
                acc[a] += (P, S, L)
                per[a].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
                del y
            del g
        N = len(stems)
        res = {}
        for a in ARMS:
            P, S, L = acc[a] / N
            res[a] = dict(score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)),
                          psnr=P, ssim=S, lpips=L)
        b = np.array(per["s1.00"])
        print(f"\n{tag}  n={N}  ({time.time()-t0:.0f}s)")
        for a in ARMS:
            v = np.array(per[a]) - b
            t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != "s1.00" else float("nan")
            print(f"  {a:>6} {res[a]['score']:9.4f}  vs s1.00 {res[a]['score']-res['s1.00']['score']:+8.4f}"
                  f"  t {t:7.2f}  win {int((v>0).sum()):3d}/{N}", flush=True)
        allres[tag] = {"n": N, "res": res, "per": per}
        json.dump(allres, open(args.out, "w"), indent=1)

    print("\n=== POOLED over scenes (unweighted mean of per-scene scores) ===")
    for a in ARMS:
        m = np.mean([allres[t]["res"][a]["score"] for t in allres])
        m0 = np.mean([allres[t]["res"]["s1.00"]["score"] for t in allres])
        print(f"  {a:>6} {m:9.4f}   vs s1.00 {m-m0:+8.4f}")
    print(f"wrote {args.out}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
