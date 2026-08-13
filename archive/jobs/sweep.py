"""Production-harness sweep of lens-field variants.

Fit every field on the scene's TRAIN renders vs its TRAIN photos (never test GT),
apply to the renders at the REAL test poses, score against the REAL test GT with the
project scorer.  5 public towers, 290 images.
"""
import os, sys, json, time
import numpy as np
import cv2
import torch
import fieldlib as F

cv2.setNumThreads(8)
torch.set_num_threads(8)

RENDER = "/mnt/d/avv/output/{s}_gsplatB9ut/test_poses_renders_png"
GT = "/mnt/d/avv/data/phase1/public_set/{s}/test/images"
RES = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/res"
os.makedirs(RES, exist_ok=True)


def build(setname, scene):
    """returns {variant_name: (field_or_None, interp, upsample_interp)}"""
    V = {}
    if setname == "grid":
        V["none"] = (None, "lanczos4", cv2.INTER_CUBIC)
        for ds in (2, 4, 8, 16, 32):
            st, _ = F.load_stack(scene, ds)
            for est in ("mean", "median"):
                V[f"g{ds}_{est}"] = (F.pool(st, est), "lanczos4", cv2.INTER_CUBIC)
    elif setname == "param":
        st4, _ = F.load_stack(scene, 4)
        st8, _ = F.load_stack(scene, 8)
        med4 = F.pool(st4, "median")
        med8 = F.pool(st8, "median")
        V["g8_median"] = (med8, "lanczos4", cv2.INTER_CUBIC)
        for kind in ("trans", "affine", "poly2", "poly3", "poly4", "brown"):
            f, _ = F.fit_parametric(med4, kind, out_hw=(989, 1320))
            V[f"p_{kind}"] = (f, "lanczos4", cv2.INTER_CUBIC)
    elif setname == "scale":
        st8, _ = F.load_stack(scene, 8)
        med8 = F.pool(st8, "median")
        V["none"] = (None, "lanczos4", cv2.INTER_CUBIC)
        for a in (0.7, 0.85, 1.0, 1.15, 1.3):
            V[f"s{a}"] = (med8 * a, "lanczos4", cv2.INTER_CUBIC)
    elif setname == "extra":
        st8, _ = F.load_stack(scene, 8)
        st4, _ = F.load_stack(scene, 4)
        med8 = F.pool(st8, "median")
        V["g8_median"] = (med8, "lanczos4", cv2.INTER_CUBIC)
        V["g8_median_cubic"] = (med8, "cubic", cv2.INTER_CUBIC)
        V["g8_median_uplanc"] = (med8, "lanczos4", cv2.INTER_LANCZOS4)
        V["g8_trim20"] = (F.pool(st8, "trim20"), "lanczos4", cv2.INTER_CUBIC)
        V["g8_trim33"] = (F.pool(st8, "trim33"), "lanczos4", cv2.INTER_CUBIC)
        # hybrid: median grid at ds8 smoothed by a gaussian (a soft way to add resolution
        # without noise), and ds4 median smoothed to the same effective bandwidth
        med4 = F.pool(st4, "median")
        for sig in (0.5, 1.0, 2.0):
            V[f"g4_med_sm{sig}"] = (cv2.GaussianBlur(med4, (0, 0), sig), "lanczos4",
                                    cv2.INTER_CUBIC)
        V["g8_med_sm1"] = (cv2.GaussianBlur(med8, (0, 0), 1.0), "lanczos4", cv2.INTER_CUBIC)
    elif setname == "ship":
        st8, _ = F.load_stack(scene, 8)
        med8 = F.pool(st8, "median")
        V["none"] = (None, "lanczos4", cv2.INTER_CUBIC)
        for a in (1.0, 1.35, 1.4):
            V[f"med_s{a}"] = (med8 * a, "lanczos4", cv2.INTER_CUBIC)
    elif setname == "interp":
        # lever (c) re-measured AT THE NEW OPERATING POINT (scale 1.35): does the resample
        # kernel still matter once the displacement is 1.35x bigger, and does warping at 2x
        # resolution ("higher precision") buy anything?
        st8, _ = F.load_stack(scene, 8)
        med = F.pool(st8, "median") * 1.35
        V["none"] = (None, "lanczos4", cv2.INTER_CUBIC)
        V["linear"] = (med, "linear", cv2.INTER_CUBIC)
        V["cubic"] = (med, "cubic", cv2.INTER_CUBIC)
        V["lanczos4"] = (med, "lanczos4", cv2.INTER_CUBIC)
        V["lanczos4_ss2"] = (med, "SS2", cv2.INTER_CUBIC)
    elif setname == "scale2":
        st8, _ = F.load_stack(scene, 8)
        med8 = F.pool(st8, "median")
        mean8 = F.pool(st8, "mean")
        V["none"] = (None, "lanczos4", cv2.INTER_CUBIC)
        for a in (1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0):
            V[f"med_s{a}"] = (med8 * a, "lanczos4", cv2.INTER_CUBIC)
        for a in (1.0, 1.3, 1.6, 1.8):
            V[f"mean_s{a}"] = (mean8 * a, "lanczos4", cv2.INTER_CUBIC)
    elif setname == "pc":
        st8, _ = F.load_stack(scene, 8)
        med8 = F.pool(st8, "median")
        V["none"] = (None, "lanczos4", cv2.INTER_CUBIC)
        V["g8_median"] = (med8, "lanczos4", cv2.INTER_CUBIC)
        PC = "/home/bkai/.claude/jobs/1c9cf7e9/tmp/pc"
        for win, stride in ((256, 64), (128, 32), (64, 16)):
            p = f"{PC}/{scene}_w{win}s{stride}.npz"
            if not os.path.exists(p):
                continue
            z = np.load(p)
            m = np.median(z["stack"], 0)
            full = F.upsample_lattice(m, 989, 1320, win, stride)
            V[f"pc{win}"] = (full, "lanczos4", cv2.INTER_CUBIC)
            V[f"pc{win}+dis"] = (0.5 * full + 0.5 * cv2.resize(med8, (1320, 989),
                                                               interpolation=cv2.INTER_CUBIC),
                                 "lanczos4", cv2.INTER_CUBIC)
    elif setname == "pool":
        st8, _ = F.load_stack(scene, 8)
        V["g8_median"] = (F.pool(st8, "median"), "lanczos4", cv2.INTER_CUBIC)
        # field pooled across the OTHER four scenes (leave-this-scene-out)
        others = [F.pool(F.load_stack(s, 8)[0], "median") for s in F.PUB if s != scene]
        V["xscene_loo"] = (np.median(np.stack(others), 0), "lanczos4", cv2.INTER_CUBIC)
        allf = [F.pool(F.load_stack(s, 8)[0], "median") for s in F.PUB]
        V["xscene_all"] = (np.median(np.stack(allf), 0), "lanczos4", cv2.INTER_CUBIC)
        own = V["g8_median"][0]
        loo = V["xscene_loo"][0]
        V["blend50"] = (0.5 * own + 0.5 * loo, "lanczos4", cv2.INTER_CUBIC)
        V["blend75"] = (0.75 * own + 0.25 * loo, "lanczos4", cv2.INTER_CUBIC)
    else:
        raise ValueError(setname)
    return V


def main():
    setname = sys.argv[1]
    scenes = sys.argv[2].split(",") if len(sys.argv) > 2 else F.PUB
    jpeg = "--jpeg" in sys.argv
    sc = F.Scorer("cuda")
    out = {}
    for s in scenes:
        V = build(setname, s)
        rd, gd = RENDER.format(s=s), GT.format(s=s)
        gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gd)}
        files = sorted(f for f in os.listdir(rd) if f.lower().endswith(".png"))
        files = [f for f in files if os.path.splitext(f)[0] in gt_by]
        maps = {}
        acc = {k: np.zeros(3) for k in V}
        t0 = time.time()
        for i, f in enumerate(files):
            r = F.load_u8(os.path.join(rd, f))
            g = F.load_u8(os.path.join(gd, gt_by[os.path.splitext(f)[0]]))
            H, W, _ = r.shape
            gt_t = F.to_t(g, sc.dev)
            for k, (fld, interp, up) in V.items():
                if fld is None:
                    y = r
                elif interp == "SS2":
                    if k not in maps:
                        fu = cv2.resize(fld, (2 * W, 2 * H), interpolation=up)
                        yy, xx = np.mgrid[0:2 * H, 0:2 * W].astype(np.float32)
                        maps[k] = ((xx + 2 * fu[..., 0]).astype(np.float32),
                                   (yy + 2 * fu[..., 1]).astype(np.float32))
                    big = cv2.resize(r.astype(np.float32) / 255.0, (2 * W, 2 * H),
                                     interpolation=cv2.INTER_LANCZOS4)
                    wp = cv2.remap(big, *maps[k], cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REFLECT)
                    y = (np.clip(cv2.resize(wp, (W, H), interpolation=cv2.INTER_AREA), 0, 1)
                         * 255 + 0.5).astype(np.uint8)
                else:
                    if k not in maps:
                        maps[k] = F.make_maps(fld, H, W, up)
                    y = F.warp_u8(r, *maps[k], F.INTERP[interp])
                if jpeg:
                    y = F.jpeg_rt(y)
                acc[k] += np.array(sc(y, gt_t))
            if i % 20 == 0:
                print(f"  {s} {i}/{len(files)} {time.time()-t0:.0f}s", flush=True)
        n = len(files)
        out[s] = {k: dict(zip(("psnr", "ssim", "lpips"), (acc[k] / n).tolist()),
                          n=n, score=F.score(*(acc[k] / n))) for k in V}
        base = out[s].get("none", out[s].get("g8_median"))
        print(f"\n== {s} (n={n}) ==")
        for k in V:
            d = out[s][k]
            print(f"  {k:18s} P {d['psnr']:7.4f}  S {d['ssim']:.5f}  L {d['lpips']:.5f}  "
                  f"score {d['score']:8.4f}  d {d['score']-base['score']:+.4f}", flush=True)
    tag = setname + ("_jpeg" if jpeg else "")
    json.dump(out, open(f"{RES}/{tag}.json", "w"), indent=1)
    print("saved", f"{RES}/{tag}.json")


if __name__ == "__main__":
    main()
