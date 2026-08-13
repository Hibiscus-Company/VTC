"""Is the optimal field scale a property of the FIELD ESTIMATE or of the member that made
the render?  Sweep the scale on several independently-trained HCM0181 test-pose render
variants, using (a) the B9ut-fitted field on all of them -- which is what production does,
one field for the whole ensemble -- and (b) B11ut60k's own field on its own renders.
"""
import os, sys, json, time
import numpy as np
import cv2
import torch
from PIL import Image
import fieldlib as F

cv2.setNumThreads(6)
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
RES = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/res"
GT = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
SCALES = [0.0, 1.0, 1.15, 1.3, 1.45, 1.6]
VARIANTS = ["gsplatB1", "gsplatB8pure", "gsplatB11ut60k", "m31b_nolpips", "gsplatB12ut8Ms7"]


def own_field():
    """fit a field on B11ut60k's own train renders (its own model, own train photos)"""
    sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track")
    from fit_field import fit_field
    return fit_field("/mnt/d/avv/output/HCM0181_gsplatB11ut60k/train_renders",
                     "/mnt/d/avv/data/phase1/public_set/HCM0181/train/images",
                     ds=8, estimator="median")


def run(sc, ren_dir, field, tag, out):
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GT)}
    files = sorted(f for f in os.listdir(ren_dir) if f.lower().endswith((".png", ".jpg")))
    files = [f for f in files if os.path.splitext(f)[0] in gt_by]
    acc = {a: np.zeros(3) for a in SCALES}
    maps = {}
    for f in files:
        r = F.load_u8(os.path.join(ren_dir, f))
        g = F.load_u8(os.path.join(GT, gt_by[os.path.splitext(f)[0]]))
        H, W, _ = r.shape
        gt_t = F.to_t(g, sc.dev)
        for a in SCALES:
            if a == 0.0:
                y = r
            else:
                if a not in maps:
                    maps[a] = F.make_maps(field * a, H, W)
                y = F.warp_u8(r, *maps[a], cv2.INTER_LANCZOS4)
            acc[a] += np.array(sc(y, gt_t))
    n = len(files)
    base = F.score(*(acc[0.0] / n))
    best = max(SCALES[1:], key=lambda a: F.score(*(acc[a] / n)))
    print(f"\n-- {tag} (n={n}) --")
    for a in SCALES:
        P, S, L = acc[a] / n
        s = F.score(P, S, L)
        out[f"{tag}_{a}"] = dict(psnr=P, ssim=S, lpips=L, score=s, n=n)
        print(f"   scale {a:4.2f}  P {P:7.4f}  S {S:.5f}  L {L:.5f}  score {s:8.4f}"
              f"  d {s-base:+.4f}", flush=True)
    print(f"   best scale {best}")


if __name__ == "__main__":
    sc = F.Scorer("cuda")
    st, _ = F.load_stack("HCM0181", 8)
    fb9 = F.pool(st, "median")
    out = {}
    for v in VARIANTS:
        d = f"/mnt/d/avv/output/HCM0181_{v}/test_poses_renders_png"
        if not os.path.isdir(d):
            d = f"/mnt/d/avv/output/HCM0181_{v}/test_poses_renders"
        if not os.path.isdir(d):
            print("skip", v); continue
        run(sc, d, fb9, f"{v}|B9field", out)
    f11 = own_field()
    print("B11 own field mean|d|", np.linalg.norm(f11, axis=2).mean())
    run(sc, "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png", f11,
        "gsplatB11ut60k|ownfield", out)
    json.dump(out, open(f"{RES}/variants.json", "w"), indent=1)
