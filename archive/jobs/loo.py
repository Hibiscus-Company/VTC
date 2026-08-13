"""Leave-one-view-out on TRAIN renders (the protocol that selected the median estimator).

Field pooled over the other 59 views, applied to the held-out view's train render, scored
against that view's train photo.  Secondary evidence: it is the only protocol available on the
graded private_set2 scenes, but it is the noisy-render regime, so the harness outranks it.
"""
import os, sys, json, time
import numpy as np
import cv2
import torch
import fieldlib as F

cv2.setNumThreads(8)
torch.set_num_threads(8)
RES = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/res"
os.makedirs(RES, exist_ok=True)

SRC = {"pub": ("/mnt/d/avv/output/{s}_gsplatB9ut/train_renders",
               "/mnt/d/avv/data/phase1/public_set/{s}/train/images", F.PUB, ""),
       "priv": ("/mnt/d/avv/r2r9/models/{s}_ut42/train_png",
                "/mnt/d/avv/data/phase1/private_set2/{s}/train/images", F.PRIV, "P_")}


def variants(stacks, i):
    """fields fit WITHOUT view i"""
    V = {"none": None}
    for ds in (4, 8, 16):
        st = np.delete(stacks[ds], i, axis=0)
        for est in ("mean", "median"):
            V[f"g{ds}_{est}"] = F.pool(st, est)
    med4 = F.pool(np.delete(stacks[4], i, axis=0), "median")
    for kind in ("affine", "poly2", "poly3", "brown"):
        V[f"p_{kind}"], _ = F.fit_parametric(med4, kind, out_hw=(989, 1320))
    return V


def main():
    which = sys.argv[1]
    rdt, gdt, scenes, pref = SRC[which]
    if len(sys.argv) > 2:
        scenes = sys.argv[2].split(",")
    sc = F.Scorer("cuda")
    out = {}
    for s in scenes:
        stacks = {ds: F.load_stack(pref + s, ds)[0] for ds in (4, 8, 16)}
        stems = F.load_stack(pref + s, 8)[1]
        rd, gd = rdt.format(s=s), gdt.format(s=s)
        gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gd)}
        acc, n = {}, 0
        t0 = time.time()
        for i, stem in enumerate(stems):
            rp = os.path.join(rd, stem + ".png")
            if not os.path.exists(rp):
                continue
            r = F.load_u8(rp)
            g = F.load_u8(os.path.join(gd, gt_by[stem]))
            H, W, _ = r.shape
            gt_t = F.to_t(g, sc.dev)
            V = variants(stacks, i)
            for k, fld in V.items():
                y = r if fld is None else F.warp_u8(r, *F.make_maps(fld, H, W), cv2.INTER_LANCZOS4)
                acc.setdefault(k, np.zeros(3))
                acc[k] += np.array(sc(y, gt_t))
            n += 1
            if i % 20 == 0:
                print(f"  {s} {i}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
        out[s] = {k: dict(zip(("psnr", "ssim", "lpips"), (acc[k] / n).tolist()),
                          n=n, score=F.score(*(acc[k] / n))) for k in acc}
        base = out[s]["none"]
        print(f"\n== LOO {which} {s} (n={n}) ==")
        for k in acc:
            d = out[s][k]
            print(f"  {k:14s} P {d['psnr']:7.4f}  S {d['ssim']:.5f}  L {d['lpips']:.5f}  "
                  f"score {d['score']:8.4f}  d {d['score']-base['score']:+.4f}", flush=True)
        json.dump(out, open(f"{RES}/loo_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
