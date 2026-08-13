#!/usr/bin/env python
"""Hypothesis (d): is the lens field really per-SCENE, or is the per-scene fit just noise
around one shared camera field?

For each scene we score, on held-out views, its OWN median field against (i) no field,
(ii) the field of every OTHER scene, (iii) the mean of all OTHER scenes' fields
(leave-one-scene-out pooled field, so nothing is fitted on the scene it corrects).
If the pooled foreign field matches or beats the own field, per-scene fitting is
over-fitting and we should pool across scenes.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fieldlib import upsample, warp
from lovo import SCENES

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="priv", choices=("priv", "pub"))
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--n_eval", type=int, default=25)
    ap.add_argument("--remap", default="cubic")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    scenes = SCENES[args.set]
    tags = [s[0] for s in scenes]
    F, stacks = {}, {}
    for tag, _, _ in scenes:
        c = np.load(f"{HERE}/cache/{args.set}_{tag}.npz")
        st = c[f"s{args.ds}"].astype(np.float32)
        stacks[tag] = st
        F[tag] = np.median(st, axis=0)
        H, W = [int(x) for x in c["HW"]]

    print("field correlation between scenes (dx / dy):")
    print("        " + "".join(f"{t:>16}" for t in tags))
    for a in tags:
        row = ""
        for b in tags:
            cx = np.corrcoef(F[a][..., 0].ravel(), F[b][..., 0].ravel())[0, 1]
            cy = np.corrcoef(F[a][..., 1].ravel(), F[b][..., 1].ravel())[0, 1]
            row += f"{cx:+7.2f}/{cy:+6.2f}"
        print(f"{a:>8}" + row)
    print("\nfield magnitude (mean|d| px):",
          {t: round(float(np.linalg.norm(F[t], axis=2).mean()), 4) for t in tags})

    acc = {}
    for tag, rdir, gdir in scenes:
        t0 = time.time()
        c = np.load(f"{HERE}/cache/{args.set}_{tag}.npz")
        stems = [str(x) for x in c["stems"]]
        N = len(stems)
        idx = np.random.RandomState(0).permutation(N)[:args.n_eval]
        gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gdir)}
        rd_by = {os.path.splitext(f)[0]: f for f in os.listdir(rdir)}
        others = [t for t in tags if t != tag]
        cand = {"none": None,
                "own_LOO": "LOO",
                "pooled_foreign": np.mean([F[t] for t in others], axis=0),
                "median_foreign": np.median(np.stack([F[t] for t in others]), axis=0)}
        for t in others:
            cand[f"from_{t}"] = F[t]
        # own field, leave-one-view-out, so "own" is not scored in-sample
        st = stacks[tag]
        sums = st.sum(axis=0)
        per = {k: [0.0, 0.0, 0.0] for k in cand}
        for i in idx:
            stem = stems[i]
            img = np.asarray(Image.open(os.path.join(rdir, rd_by[stem])).convert("RGB"),
                             dtype=np.float32) / 255.0
            g = torch.from_numpy(
                np.asarray(Image.open(os.path.join(gdir, gt_by[stem])).convert("RGB"),
                           dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            own = np.median(np.delete(st, i, axis=0), axis=0)
            for k, f in cand.items():
                ff = own if k == "own_LOO" else f
                im = img if ff is None else np.clip(
                    warp(img, upsample(ff, H, W, "cubic"), args.remap), 0, 1)
                r = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).unsqueeze(0).to(dev)
                with torch.no_grad():
                    per[k][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                    per[k][1] += float(repo_ssim(r, g))
                    per[k][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
        n = len(idx)
        acc[tag] = {}
        for k in cand:
            P, S, L = (x / n for x in per[k])
            acc[tag][k] = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        print(f"[{tag}] {time.time()-t0:.0f}s  " +
              "  ".join(f"{k}={acc[tag][k]-acc[tag]['own_LOO']:+.4f}"
                        for k in ("none", "pooled_foreign", "median_foreign")), flush=True)

    with open(args.out, "w") as fh:
        json.dump(acc, fh, indent=1)
    keys = ["none", "own_LOO", "pooled_foreign", "median_foreign"] + [f"from_{t}" for t in tags]
    print(f"\n{'field used':>18} {'MEAN d(own)':>12}  " + "  ".join(f"{t:>9}" for t in tags))
    for k in keys:
        vals = [acc[t][k] - acc[t]["own_LOO"] for t in tags if k in acc[t]]
        cells = "  ".join(f"{acc[t][k]-acc[t]['own_LOO']:+9.4f}" if k in acc[t] else f"{'-':>9}"
                          for t in tags)
        print(f"{k:>18} {np.mean(vals):+12.4f}  {cells}")


if __name__ == "__main__":
    main()
