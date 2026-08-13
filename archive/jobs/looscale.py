"""Leave-one-view-out SCALE selection on TRAIN renders.

The scale is a fitted parameter, so it must be chosen without test GT.  This is the legal
selection rule: for held-out view i, pool the field over the other n-1 views, sweep the scale,
score against view i's own TRAIN photo.  Run it on the 5 public towers to check that the rule
picks the same scale the real test GT prefers, then on private_set2 to get the shipping value.
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
SCALES = [0.0, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0]

SRC = {"pub": ("/mnt/d/avv/output/{s}_gsplatB9ut/train_renders",
               "/mnt/d/avv/data/phase1/public_set/{s}/train/images", F.PUB, ""),
       "priv": ("/mnt/d/avv/r2r9/models/{s}_ut42/train_png",
                "/mnt/d/avv/data/phase1/private_set2/{s}/train/images", F.PRIV, "P_")}


def main():
    which = sys.argv[1]
    rdt, gdt, scenes, pref = SRC[which]
    if len(sys.argv) > 2:
        scenes = sys.argv[2].split(",")
    sc = F.Scorer("cuda")
    out = {}
    for s in scenes:
        stack, stems = F.load_stack(pref + s, 8)
        rd, gd = rdt.format(s=s), gdt.format(s=s)
        gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gd)}
        acc, n = {a: np.zeros(3) for a in SCALES}, 0
        t0 = time.time()
        for i, stem in enumerate(stems):
            rp = os.path.join(rd, stem + ".png")
            if not os.path.exists(rp) or stem not in gt_by:
                continue
            r = F.load_u8(rp)
            g = F.load_u8(os.path.join(gd, gt_by[stem]))
            H, W, _ = r.shape
            gt_t = F.to_t(g, sc.dev)
            fld = F.pool(np.delete(stack, i, axis=0), "median")
            for a in SCALES:
                y = r if a == 0.0 else F.warp_u8(r, *F.make_maps(fld * a, H, W),
                                                 cv2.INTER_LANCZOS4)
                acc[a] += np.array(sc(y, gt_t))
            n += 1
            if i % 20 == 0:
                print(f"  {s} {i}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
        out[s] = {str(a): dict(zip(("psnr", "ssim", "lpips"), (acc[a] / n).tolist()),
                               n=n, score=F.score(*(acc[a] / n))) for a in SCALES}
        base = out[s]["0.0"]
        print(f"\n== LOO-train {which} {s} (n={n}) ==")
        for a in SCALES:
            d = out[s][str(a)]
            print(f"  scale {a:4.2f}  P {d['psnr']:7.4f}  S {d['ssim']:.5f}  L {d['lpips']:.5f}"
                  f"  score {d['score']:8.4f}  d {d['score']-base['score']:+.4f}", flush=True)
        json.dump(out, open(f"{RES}/looscale_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
