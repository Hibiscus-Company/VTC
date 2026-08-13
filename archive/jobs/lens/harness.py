#!/usr/bin/env python
"""PRODUCTION-REGIME test of lens-field variants: fields fitted on TRAIN photos only
(Rule 10), applied at REAL test poses, scored against REAL test GT.

HCM0181 uses the 4-member pixel-mean ensemble (the production combiner); the other four
public towers use their single gsplatB9ut render.  Optionally re-encodes through the
shipped JPEG (q100, subsampling 2) before scoring, because the record shows the encode is
load-bearing and can flip a post-processing operator's sign.
"""
import argparse, io, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fieldlib import LooPool, PolyModel, gauss_smooth, upsample, warp
from lovo import parse_variants

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))

RENDERS = {
    "HCM0181": ["/mnt/d/avv/prodharness/k4/png"],
    "HCM0193": ["/mnt/d/avv/output/HCM0193_gsplatB9ut/test_poses_renders_png"],
    "HCM0204": ["/mnt/d/avv/output/HCM0204_gsplatB9ut/test_poses_renders_png"],
    "hcm0031": ["/mnt/d/avv/output/hcm0031_gsplatB9ut/test_poses_renders_png"],
    "hcm0034": ["/mnt/d/avv/output/hcm0034_gsplatB9ut/test_poses_renders_png"],
}
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--scenes", default="HCM0181,HCM0193,HCM0204,hcm0031,hcm0034")
    ap.add_argument("--jpeg", action="store_true", help="score after the shipped JPEG encode")
    ap.add_argument("--est", default="", help="flow-estimator cache prefix: '', dis0_, dis2x_, lk_")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    variants = parse_variants([l.strip() for l in open(args.variants)
                               if l.strip() and not l.startswith("#")])
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    acc = {v["name"]: {} for v in variants}
    acc["none"] = {}
    for tag in args.scenes.split(","):
        t0 = time.time()
        gtd = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
        gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
        dirs = RENDERS[tag]
        stems = sorted(s for s in gt_by
                       if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
        cache = np.load(f"{HERE}/cache/{args.est}pub_{tag}.npz")
        H, W = [int(x) for x in cache["HW"]]
        pools, polys, fields = {}, {}, {}
        for v in variants:
            d = v["ds"]
            if d not in pools:
                pools[d] = LooPool(cache[f"s{d}"])
            f = pools[d].pooled(v["pool"])          # full train stack: test poses are disjoint
            if v["smooth"] > 0:
                f = gauss_smooth(f, v["smooth"])
            if v["poly"]:
                key = (v["poly"], d)
                if key not in polys:
                    fh, fw = cache[f"s{d}"].shape[1:3]
                    polys[key] = PolyModel(v["poly"], fh, fw, H, W)
                fields[v["name"]] = polys[key].full_field(f)
            else:
                fields[v["name"]] = upsample(f, H, W, v["up"])
        del pools, polys

        def enc(img):
            if not args.jpeg:
                return img
            b = io.BytesIO()
            Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
                b, "JPEG", **SHIPPED_JPEG)
            return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                              dtype=np.float32) / 255.0

        def score(img, g):
            r = torch.from_numpy(np.ascontiguousarray(np.clip(img, 0, 1))).permute(
                2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                p = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                s = float(repo_ssim(r, g))
                l = float(vgg(r * 2 - 1, g * 2 - 1).item())
            return p, s, l

        per = {k: [0.0, 0.0, 0.0] for k in acc}
        for c, s in enumerate(stems):
            img = np.mean([np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                                      dtype=np.float32) / 255.0 for d in dirs], axis=0)
            g = torch.from_numpy(
                np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                           dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            p, ss, l = score(enc(img), g)
            per["none"][0] += p; per["none"][1] += ss; per["none"][2] += l
            for v in variants:
                im = np.clip(warp(img, fields[v["name"]], v["remap"]), 0, 1)
                p, ss, l = score(enc(im), g)
                per[v["name"]][0] += p; per[v["name"]][1] += ss; per[v["name"]][2] += l
            if c % 20 == 0:
                print(f"  [{tag}] {c}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
        n = len(stems)
        for k in acc:
            P, S, L = (x / n for x in per[k])
            acc[k][tag] = dict(score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)),
                               psnr=P, ssim=S, lpips=L, n=n)
        print(f"[{tag}] done {time.time()-t0:.0f}s", flush=True)

    with open(args.out, "w") as fh:
        json.dump(acc, fh, indent=1)
    names = ["none"] + [v["name"] for v in variants]
    base = "median_ds8" if "median_ds8" in acc else names[0]
    tags = args.scenes.split(",")
    print(f"\n{'variant':>22} {'MEAN':>9} {'d(base)':>8}  " + "  ".join(f"{t:>9}" for t in tags))
    for k in names:
        m = np.mean([acc[k][t]["score"] for t in tags])
        b = np.mean([acc[base][t]["score"] for t in tags])
        print(f"{k:>22} {m:9.4f} {m-b:+8.4f}  " +
              "  ".join(f"{acc[k][t]['score']-acc[base][t]['score']:+9.4f}" for t in tags))


if __name__ == "__main__":
    main()
