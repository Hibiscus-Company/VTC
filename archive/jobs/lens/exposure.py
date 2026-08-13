#!/usr/bin/env python
"""HOW MUCH SCORE IS LOCKED UP IN PER-IMAGE EXPOSURE / WHITE BALANCE?

A drone on an orbit runs auto-exposure and auto-WB, so consecutive photos of the same tower are
not photometrically identical. 3DGS has no per-image photometric parameter, so it fits the
AVERAGE, and every test photo is offset from that average by its own exposure state. That offset
is a global, low-dimensional error sitting on top of an otherwise correct render, and it costs
PSNR (directly), SSIM (luminance term) and LPIPS.

This script measures the ORACLE: the best per-image photometric correction, fitted against real
test GT. That is a DIAGNOSTIC on public_set (the sanctioned local-scoring GT), not a shippable
operator -- it sizes the prize and nothing more.

Why this oracle is not the per-view-field mirage: that one fitted a dense 123x165x2 field from one
noisy DIS flow field, so it could fit its own noise. This fits 1, 3 or 6 numbers from ~1.3M pixels.
It cannot noise-fit. If the oracle is large, the only question left is whether the per-image
parameter is PREDICTABLE without GT -- and the filenames carry a capture sequence number, with
40/60 private test frames having both immediate sequence-neighbours in the train set.

Also reports, for free, the LEGAL half: the same fit done on TRAIN renders vs TRAIN photos, which
is what a predictor would interpolate from, plus a leave-one-out test of how well a train image's
own coefficient is predicted by its sequence neighbours.
"""
import argparse, io, json, os, re, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)

RENDERS = {
    "HCM0181": "/mnt/d/avv/prodharness/k4/png",
    "HCM0193": "/mnt/d/avv/output/HCM0193_gsplatB9ut/test_poses_renders_png",
    "HCM0204": "/mnt/d/avv/output/HCM0204_gsplatB9ut/test_poses_renders_png",
    "hcm0031": "/mnt/d/avv/output/hcm0031_gsplatB9ut/test_poses_renders_png",
    "hcm0034": "/mnt/d/avv/output/hcm0034_gsplatB9ut/test_poses_renders_png",
}
TRAIN_RENDERS = {t: f"/mnt/d/avv/output/{t}_gsplatB9ut/train_renders" for t in RENDERS}


def seq_of(stem):
    m = re.match(r"DJI_(\d{14})_(\d+)_V", stem)
    return int(m.group(2)) if m else None


def fit(r, g, mode):
    """least-squares photometric map render->photo. r,g are HxWx3 float."""
    if mode == "global_gain":
        a = float((r * g).sum() / max((r * r).sum(), 1e-9))
        return np.full(3, a, np.float64), np.zeros(3)
    A, B = np.zeros(3), np.zeros(3)
    for c in range(3):
        x, y = r[..., c].ravel(), g[..., c].ravel()
        if mode == "gain":
            A[c] = float(x @ y / max(x @ x, 1e-9))
        else:  # gain+bias
            n = x.size
            sx, sy, sxx, sxy = x.sum(), y.sum(), x @ x, x @ y
            den = n * sxx - sx * sx
            A[c] = (n * sxy - sx * sy) / max(den, 1e-9)
            B[c] = (sy - A[c] * sx) / n
    return A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="HCM0181,HCM0193,HCM0204,hcm0031,hcm0034")
    ap.add_argument("--out", default=f"{HERE}/res_exposure.json")
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    def ld(p):
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

    def enc(img):
        b = io.BytesIO()
        Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            b, "JPEG", **SHIPPED_JPEG)
        return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                          dtype=np.float32) / 255.0

    ARMS = ("base", "global_gain", "gain", "gainbias")
    out = {}
    for tag in args.scenes.split(","):
        t0 = time.time()
        gtd = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
        gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
        rd = RENDERS[tag]
        stems = sorted(s for s in gt_by if os.path.exists(os.path.join(rd, s + ".png")))
        cache = np.load(f"{HERE}/cache/pub_{tag}.npz")
        lens = upsample(LooPool(cache["s8"]).pooled("median"),
                        *[int(x) for x in cache["HW"]], "cubic")

        acc = {a: [0.0, 0.0, 0.0] for a in ARMS}
        coefs = {}
        for c, s in enumerate(stems):
            img = np.clip(warp(ld(os.path.join(rd, s + ".png")), lens, "lanczos"), 0, 1)
            gt = ld(os.path.join(gtd, gt_by[s]))
            g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(dev)
            for a in ARMS:
                if a == "base":
                    im = img
                else:
                    A, B = fit(img, gt, a)
                    im = np.clip(img * A[None, None, :] + B[None, None, :], 0, 1)
                    if a == "gain":
                        coefs[s] = A.tolist()
                r = torch.from_numpy(np.ascontiguousarray(enc(im))).permute(
                    2, 0, 1).unsqueeze(0).to(dev)
                with torch.no_grad():
                    acc[a][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                    acc[a][1] += float(repo_ssim(r, g))
                    acc[a][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
            if c % 20 == 0:
                print(f"  [{tag}] {c}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

        n = len(stems)
        res = {}
        for a in ARMS:
            P, S, L = (x / n for x in acc[a])
            res[a] = dict(score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)),
                          psnr=P, ssim=S, lpips=L)
        C = np.array([coefs[s] for s in stems])
        res["gain_stats"] = dict(mean=C.mean(0).tolist(), std=C.std(0).tolist(),
                                 lo=C.min(0).tolist(), hi=C.max(0).tolist())
        res["gains"] = {s: coefs[s] for s in stems}
        out[tag] = res
        print(f"\n[{tag}] n={n}  per-image ORACLE gain, fitted vs real test GT")
        print(f"   mean {np.round(C.mean(0),4)}  sd {np.round(C.std(0),4)}  "
              f"range {np.round(C.min(0),3)}..{np.round(C.max(0),3)}")
        print(f"   {'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8}  {'vs base':>8}")
        for a in ARMS:
            r_ = res[a]
            print(f"   {a:>12} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:7.4f} "
                  f"{r_['lpips']:8.4f}  {r_['score']-res['base']['score']:+8.4f}")
        # is the coefficient PREDICTABLE from sequence neighbours? (uses no GT of the target)
        sq = np.array([seq_of(s) for s in stems], dtype=float)
        ok = ~np.isnan(sq)
        if ok.sum() > 5:
            o = np.argsort(sq[ok]); q = sq[ok][o]; V = C[ok][o]
            pred = np.stack([np.interp(q, np.delete(q, i), np.delete(V[:, c], i))
                             for c in range(3) for i in [0]], axis=-1) if False else None
            # leave-one-out linear interpolation from the OTHER test frames, per channel
            err, spread = [], []
            for i in range(1, len(q) - 1):
                p = [np.interp(q[i], np.delete(q, i), np.delete(V[:, c], i)) for c in range(3)]
                err.append(np.abs(np.array(p) - V[i])); spread.append(np.abs(V[i] - V.mean(0)))
            err, spread = np.array(err).mean(0), np.array(spread).mean(0)
            print(f"   LOO interpolation from sequence neighbours: |err| {np.round(err,4)} "
                  f"vs |deviation from mean| {np.round(spread,4)}  "
                  f"-> explains {np.round(100*(1-err/np.maximum(spread,1e-9)),1)}%")
        print(f"[{tag}] {time.time()-t0:.0f}s\n", flush=True)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    tags = args.scenes.split(",")
    print(f"{'arm':>12} {'MEAN':>9} {'vs base':>9}")
    for a in ARMS:
        m = np.mean([out[t][a]["score"] for t in tags])
        b = np.mean([out[t]["base"]["score"] for t in tags])
        print(f"{a:>12} {m:9.4f} {m-b:+9.4f}")


if __name__ == "__main__":
    main()
