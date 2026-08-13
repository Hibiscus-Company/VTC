#!/usr/bin/env python
"""Leave-one-view-out comparison of lens-field variants, project scorer (PSNR/SSIM/LPIPS-vgg).

For each held-out train view i the field is pooled over every view EXCEPT i, so no view ever
contributes to the field that corrects it.  All variants see the SAME flow stack and the SAME
held-out views, so the comparison is paired.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fieldlib import LooPool, PolyModel, gauss_smooth, upsample, warp

Image.MAX_IMAGE_PIXELS = None

SCENES = {
    "priv": [(t, f"/mnt/d/avv/r2r9/models/{t}_ut42/train_png",
              f"/mnt/d/avv/data/phase1/private_set2/{t}/train/images")
             for t in ("HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674")],
    "chairbonsai": [("chair", "/mnt/d/avv/r2r9/models/chair_ut42/train_png",
                     "/mnt/d/avv/data/phase1/private_set2/chair/train/images"),
                    ("bonsai", "/mnt/d/avv/r2r8/models/bonsai_ut42/train_png",
                     "/mnt/d/avv/data/phase1/private_set2/bonsai/train/images")],
    "pub": [(t, f"/mnt/d/avv/output/{t}_gsplatB9ut/train_renders",
             f"/mnt/d/avv/data/phase1/public_set/{t}/train/images")
            for t in ("HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034")],
}


def parse_variants(spec):
    """each variant: name|pool|ds|smooth|poly|up|remap   (poly=0 -> free-form grid)"""
    out = []
    for line in spec:
        n, pool, ds, sm, poly, up, rm = line.split("|")
        out.append(dict(name=n, pool=pool, ds=int(ds), smooth=float(sm),
                        poly=int(poly), up=up, remap=rm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="priv", choices=("priv", "pub", "chairbonsai"))
    ap.add_argument("--scenes", default="")
    ap.add_argument("--n_eval", type=int, default=40)
    ap.add_argument("--variants", required=True, help="file with one variant spec per line")
    ap.add_argument("--est", default="", help="flow-estimator cache prefix: '', dis0_, dis2x_, lk_")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    variants = parse_variants([l.strip() for l in open(args.variants)
                               if l.strip() and not l.startswith("#")])
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    scenes = SCENES[args.set]
    if args.scenes:
        keep = set(args.scenes.split(","))
        scenes = [s for s in scenes if s[0] in keep]

    acc = {v["name"]: {} for v in variants}
    acc["none"] = {}
    for tag, rdir, gdir in scenes:
        t0 = time.time()
        cache = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/cache/"
                        f"{args.est}{args.set}_{tag}.npz")
        stems = list(cache["stems"])
        H, W = [int(x) for x in cache["HW"]]
        pools, polys = {}, {}
        need_ds = sorted({v["ds"] for v in variants})
        for d in need_ds:
            pools[d] = LooPool(cache[f"s{d}"])
        for v in variants:
            if v["poly"]:
                key = (v["poly"], v["ds"])
                if key not in polys:
                    fh, fw = cache[f"s{v['ds']}"].shape[1:3]
                    polys[key] = PolyModel(v["poly"], fh, fw, H, W)
        N = len(stems)
        idx = np.random.RandomState(0).permutation(N)[:args.n_eval]
        gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gdir)}
        rd_by = {os.path.splitext(f)[0]: f for f in os.listdir(rdir)}

        def score(img, g):
            r = torch.from_numpy(np.ascontiguousarray(np.clip(img, 0, 1))).permute(2, 0, 1
                                                                                   ).unsqueeze(0).to(dev)
            with torch.no_grad():
                p = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                s = float(repo_ssim(r, g))
                l = float(vgg(r * 2 - 1, g * 2 - 1).item())
            return p, s, l

        per = {k: [0.0, 0.0, 0.0] for k in acc}
        for c, i in enumerate(idx):
            stem = str(stems[i])
            img = np.asarray(Image.open(os.path.join(rdir, rd_by[stem])).convert("RGB"),
                             dtype=np.float32) / 255.0
            g = torch.from_numpy(
                np.asarray(Image.open(os.path.join(gdir, gt_by[stem])).convert("RGB"),
                           dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            p, s, l = score(img, g)
            per["none"][0] += p; per["none"][1] += s; per["none"][2] += l
            base_field = {}
            for v in variants:
                k = (v["pool"], v["ds"])
                if k not in base_field:
                    base_field[k] = pools[v["ds"]].pooled(v["pool"], i)
                f = base_field[k]
                if v["smooth"] > 0:
                    f = gauss_smooth(f, v["smooth"])
                if v["poly"]:
                    fu = polys[(v["poly"], v["ds"])].full_field(f)
                else:
                    fu = upsample(f, H, W, v["up"])
                im = np.clip(warp(img, fu, v["remap"]), 0, 1)
                p, s, l = score(im, g)
                per[v["name"]][0] += p; per[v["name"]][1] += s; per[v["name"]][2] += l
            if c % 10 == 0:
                print(f"  [{tag}] {c}/{len(idx)}  {time.time()-t0:.0f}s", flush=True)
        n = len(idx)
        for k in acc:
            P, S, L = (x / n for x in per[k])
            sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
            acc[k][tag] = dict(score=sc, psnr=P, ssim=S, lpips=L, n=n)
        print(f"[{tag}] done {time.time()-t0:.0f}s", flush=True)
        del pools, polys

    with open(args.out, "w") as fh:
        json.dump(acc, fh, indent=1)
    names = ["none"] + [v["name"] for v in variants]
    base = "median_ds8" if "median_ds8" in acc else names[0]
    tags = [s[0] for s in scenes]
    print(f"\n{'variant':>22} {'MEAN':>9} {'d(base)':>8}  " +
          "  ".join(f"{t:>9}" for t in tags))
    for k in names:
        m = np.mean([acc[k][t]["score"] for t in tags])
        b = np.mean([acc[base][t]["score"] for t in tags])
        print(f"{k:>22} {m:9.4f} {m-b:+8.4f}  " +
              "  ".join(f"{acc[k][t]['score']-acc[base][t]['score']:+9.4f}" for t in tags))


if __name__ == "__main__":
    main()
