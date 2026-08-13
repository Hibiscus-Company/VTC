#!/usr/bin/env python
"""IS THE PIXEL-MEAN ENSEMBLE BLURRING ITSELF BY AVERAGING MISREGISTERED MEMBERS?

The lens field exists because a render is displaced against the photo by a fixed sub-pixel
field, and correcting it was worth +0.73 on the leaderboard. Every ensemble member is a
separately-trained model, so each has its OWN slightly different geometry -- and therefore its
own displacement. We average them on a shared pixel grid without ever checking that they agree.
If member-to-member displacement is of the same order as the render-to-photo displacement
(~0.10 px), the pixel mean has been quietly low-passing itself on every round since r5, and the
fix costs no GPU and no training.

RULE 10: this uses NO ground truth of any kind. Member-to-member flow is render-against-render.
It is legal at test poses, which is where it is measured and where it would ship.

ARMS (each then gets the SAME production lens field + the shipped JPEG, so the comparison is
against what actually ships):
  base        plain pixel mean, i.e. r27
  aligned     each member warped onto the base mean by its OWN pooled (median-over-views)
              displacement, then averaged. Pooling over views keeps the registration of the
              result identical to base's, so the existing lens field stays valid.
  aligned_pv  per-view alignment. Almost certainly noise-fitting -- the per-view field oracle
              already died that way -- but it bounds what alignment could ever be worth.
"""
import argparse, io, json, os, sys, time
import numpy as np
import cv2
import torch
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)

MEMBERS = {
    "HCM0181": ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
                "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
                "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
                "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"],
}


def gray(x):
    return (cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="HCM0181")
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--clip", type=float, default=6.0)
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    dirs = MEMBERS[args.tag]
    gtd = f"/mnt/d/avv/data/phase1/public_set/{args.tag}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
    print(f"{args.tag}: {len(dirs)} members x {len(stems)} real test poses\n", flush=True)

    def ld(d, s):
        return np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                          dtype=np.float32) / 255.0

    # ---- pass 1: member -> base-mean displacement, per view, at 1/ds ----
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    stacks = [[] for _ in dirs]
    t0 = time.time()
    for c, s in enumerate(stems):
        mem = [ld(d, s) for d in dirs]
        base = np.mean(mem, axis=0)
        bg = gray(base)
        H, W, _ = base.shape
        for k, m in enumerate(mem):
            fl = np.clip(dis.calc(bg, gray(m), None), -args.clip, args.clip)
            stacks[k].append(cv2.resize(fl, (W // args.ds, H // args.ds),
                                        interpolation=cv2.INTER_AREA))
        if c % 20 == 0:
            print(f"  flow {c}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    stacks = [np.stack(x) for x in stacks]

    print(f"\nMEMBER-TO-MEAN DISPLACEMENT (the quantity in question):")
    pooled = []
    for k, st in enumerate(stacks):
        p = np.median(st, axis=0)
        pooled.append(p)
        pv = np.abs(st).mean(axis=(1, 2, 3))
        # how much of each view's displacement is the FIXED part vs per-view noise
        num = np.mean([np.sum(st[i] * p) for i in range(st.shape[0])])
        den = np.sum(p * p)
        print(f"  member {k}: pooled mean|d| {np.abs(p).mean():.4f} px  max {np.abs(p).max():.3f}"
              f"   per-view mean|d| {pv.mean():.4f}+-{pv.std():.4f}"
              f"   fixed-part projection {num/max(den,1e-9):.3f}")
    print(f"  [reference] render-to-PHOTO field mean|d| ~0.10 px -- that one was worth +0.73 LB\n",
          flush=True)

    # ---- pass 2: score the arms against real GT ----
    cache = np.load(f"{HERE}/cache/pub_{args.tag}.npz")
    lens = upsample(LooPool(cache[f"s8"]).pooled("median"),
                    *[int(x) for x in cache["HW"]], "cubic")
    full = [upsample(p, *[int(x) for x in cache["HW"]], "cubic") for p in pooled]

    def enc(img):
        b = io.BytesIO()
        Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            b, "JPEG", **SHIPPED_JPEG)
        return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                          dtype=np.float32) / 255.0

    arms = ("base", "aligned", "aligned_pv")
    acc = {a: {"png": [0.0, 0.0, 0.0], "jpg": [0.0, 0.0, 0.0]} for a in arms}
    t0 = time.time()
    for c, s in enumerate(stems):
        mem = [ld(d, s) for d in dirs]
        H, W, _ = mem[0].shape
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                                        dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        out = {
            "base": np.mean(mem, axis=0),
            "aligned": np.mean([warp(mem[k], full[k], "lanczos") for k in range(len(mem))], axis=0),
            "aligned_pv": np.mean([warp(mem[k], upsample(stacks[k][c], H, W, "cubic"), "lanczos")
                                   for k in range(len(mem))], axis=0),
        }
        for a in arms:
            im = np.clip(warp(np.clip(out[a], 0, 1), lens, "lanczos"), 0, 1)
            for key, x in (("png", im), ("jpg", enc(im))):
                r = torch.from_numpy(np.ascontiguousarray(np.clip(x, 0, 1))).permute(
                    2, 0, 1).unsqueeze(0).to(dev)
                with torch.no_grad():
                    acc[a][key][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                    acc[a][key][1] += float(repo_ssim(r, g))
                    acc[a][key][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
        if c % 20 == 0:
            print(f"  score {c}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    n = len(stems)
    res = {}
    print(f"\n{'arm':>12} {'enc':>5} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8}  {'vs base':>8}")
    for key in ("png", "jpg"):
        for a in arms:
            P, S, L = (x / n for x in acc[a][key])
            sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
            res[f"{a}_{key}"] = dict(score=sc, psnr=P, ssim=S, lpips=L)
            d = sc - res[f"base_{key}"]["score"]
            print(f"{a:>12} {key:>5} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f}  {d:+8.4f}")
    with open(f"{HERE}/res_memalign_{args.tag}.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
