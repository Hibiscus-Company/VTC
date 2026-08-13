#!/usr/bin/env python
"""Two-stage field: fit a field, warp the train renders with it, re-fit a RESIDUAL field on
the already-corrected renders, and ship the composition.

Rationale: our displacements (~0.2 px) sit an order of magnitude below what DIS is built to
resolve, so a single pass may leave a systematic residual that a second pass -- now operating
on nearly-aligned images, where the linearisation is much better -- can still see.

HONEST PROTOCOL: 2-fold cross-fit.  Both stages are fitted on one half of the train views and
scored on the other half, so nothing the score sees ever contributed to either field.
"""
import argparse, os, sys, time
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fieldlib import upsample, warp
from lovo import SCENES

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="priv")
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--remap", default="lanczos")
    ap.add_argument("--n_eval", type=int, default=25)
    ap.add_argument("--stages", type=int, default=3)
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def flow(gt_u8, rn_u8, H, W, ds):
        fl = np.clip(dis.calc(gt_u8, rn_u8, None), -6.0, 6.0)
        return cv2.resize(fl, (W // ds, H // ds), interpolation=cv2.INTER_AREA)

    rows = []
    for tag, rdir, gdir in SCENES[args.set]:
        t0 = time.time()
        cache = np.load(f"{HERE}/cache/{args.set}_{tag}.npz")
        stems = [str(x) for x in cache["stems"]]
        H, W = [int(x) for x in cache["HW"]]
        st1 = cache[f"s{args.ds}"].astype(np.float32)
        gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gdir)}
        rd_by = {os.path.splitext(f)[0]: f for f in os.listdir(rdir)}
        N = len(stems)
        perm = np.random.RandomState(0).permutation(N)
        folds = [perm[:N // 2], perm[N // 2:]]

        def load(i):
            s = stems[i]
            r = np.asarray(Image.open(os.path.join(rdir, rd_by[s])).convert("RGB"),
                           dtype=np.float32) / 255.0
            g = np.asarray(Image.open(os.path.join(gdir, gt_by[s])).convert("RGB"),
                           dtype=np.float32) / 255.0
            return r, g

        acc = {k: [0.0, 0.0, 0.0] for k in ["none"] + [f"stage{k}" for k in range(1, args.stages + 1)]}
        n_scored = 0
        for fi, fit_idx in enumerate(folds):
            ev_idx = folds[1 - fi][:args.n_eval // 2 + 1]
            # ---- build the cascade on the FIT half only
            fields = [np.median(st1[fit_idx], axis=0)]
            cur = {i: None for i in fit_idx}
            for st in range(2, args.stages + 1):
                acc_s = []
                for i in fit_idx:
                    r, g = load(i)
                    fu = upsample(sum_fields(fields, H, W), H, W, "cubic")
                    rw = np.clip(warp(r, fu, args.remap), 0, 1)
                    rg = (cv2.cvtColor(rw, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
                    gg = (cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
                    acc_s.append(flow(gg, rg, H, W, args.ds))
                fields.append(np.median(np.stack(acc_s), axis=0))
                print(f"  [{tag}] fold{fi} stage{st} residual mean|d| "
                      f"{np.linalg.norm(fields[-1], axis=2).mean():.4f} px", flush=True)
            # ---- score on the held-out half
            cum = [fields[0]]
            for st in range(1, args.stages):
                cum.append(cum[-1] + fields[st])
            fus = [upsample(c, H, W, "cubic") for c in cum]
            for i in ev_idx:
                r, g = load(i)
                gt = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(dev)

                def sc(x):
                    t = torch.from_numpy(np.ascontiguousarray(np.clip(x, 0, 1))).permute(
                        2, 0, 1).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        return (10 * np.log10(1.0 / max(((t - gt) ** 2).mean().item(), 1e-12)),
                                float(repo_ssim(t, gt)),
                                float(vgg(t * 2 - 1, gt * 2 - 1).item()))

                p, s, l = sc(r)
                acc["none"][0] += p; acc["none"][1] += s; acc["none"][2] += l
                for k, fu in enumerate(fus, start=1):
                    p, s, l = sc(warp(r, fu, args.remap))
                    acc[f"stage{k}"][0] += p; acc[f"stage{k}"][1] += s; acc[f"stage{k}"][2] += l
                n_scored += 1
        out = {}
        for k, (P, S, L) in acc.items():
            P, S, L = P / n_scored, S / n_scored, L / n_scored
            out[k] = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        rows.append((tag, out))
        print(f"[{tag}] n={n_scored} {time.time()-t0:.0f}s  " +
              "  ".join(f"{k}={v-out['stage1']:+.4f}" for k, v in out.items()), flush=True)

    keys = ["none"] + [f"stage{k}" for k in range(1, args.stages + 1)]
    print(f"\n{'model':>10} {'MEAN d(stage1)':>15}  " + "  ".join(f"{t:>9}" for t, _ in rows))
    for k in keys:
        d = [o[k] - o["stage1"] for _, o in rows]
        print(f"{k:>10} {np.mean(d):+15.4f}  " + "  ".join(f"{x:+9.4f}" for x in d))


def sum_fields(fields, H, W):
    s = fields[0].copy()
    for f in fields[1:]:
        s = s + f
    return s


if __name__ == "__main__":
    main()
