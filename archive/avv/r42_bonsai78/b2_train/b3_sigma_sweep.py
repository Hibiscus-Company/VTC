#!/usr/bin/env python
"""B2 (3) de-risk: score the PREDICTED-sigma blur operator on today's (non-deblurred)
sr01 renders, 28 bonsai eval holes.  CPU ONLY.

This is the exact operator the patched renderer applies, run on the model we already have.
It cannot show the benefit of the TRAINING fix (this model was never deblurred), but it
bounds the DOWNSIDE of the test-time half and calibrates --blur_gamma: if applying the
predicted blur to a render costs more than the training fix can plausibly buy, ship
--blur_mode none / const instead of predict.

A1 already measured the sharpening side (all positive-alpha unsharp LOSES; best global is
a MILD BLUR, us(sigma 1.6, a=-0.15) = +0.0735) and A2 measured the prediction-keyed
UNSHARP side (+0.010 legal, +0.017 oracle).  Nobody has measured the prediction-keyed
BLUR side, which is what the B2 renderer patch actually does.
"""
import os, sys, csv, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/mnt/d/avv/r42_bonsai78/a1_oracle")
sys.path.insert(0, "/mnt/d/avv/r42_bonsai78/b2_train")
import numpy as np

REN = "/mnt/d/avv/r36_shape/sr01/eval_png"
GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
SIDE = "/mnt/d/avv/r42_bonsai78/b2_train/bonsai_sharp_sidecar.csv"
ES = "/mnt/d/avv/evalsplit/bonsai"


def sigma_table():
    """sigma_test per eval hole, from the TRAIN frames only (Rule-10 clean)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rb2", "/mnt/d/avv/r42_bonsai78/b2_train/render_gsplat_b2.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    side = {r["name"]: float(r["sigma_init"]) for r in csv.DictReader(open(SIDE))}
    EV = set(json.load(open(f"{ES}/split.json"))["eval_frames"])
    tr = [n for n in sorted(side) if int(n[6:12]) not in EV]
    blur = {"sigma": np.array([side[n] for n in tr]), "names": tr, "sigma_max": 2.5}
    rows = list(csv.DictReader(open(f"{ES}/eval_poses.csv")))
    lut, _ = R.build_sigma_lookup(blur, rows, "predict", 1.0, 13.0, 2.5, 0.0)
    return {os.path.splitext(k)[0]: v for k, v in lut.items()}, R


def job(a):
    stem, kind, param, sig = a
    import cpu_metrics as M
    g = M.load(f"{GT}/{stem}.jpg")
    r = M.load(f"{REN}/{stem}.png")
    if kind == "base":
        out = r
    elif kind == "const":
        out = M.gaussian_blur(r, param)
    else:                                   # predicted, gamma = param
        s = param * sig
        out = M.gaussian_blur(r, s) if s > 1e-3 else r
    P, S, L = M.metrics_fast(out.clamp(0, 1), g, M.gt_feats(g))
    return stem, kind, param, P, S, L


def main():
    lut, R = sigma_table()
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(REN) if f.endswith(".png"))
    assert len(stems) == 28, len(stems)
    v = np.array([lut[s] for s in stems])
    print(f"predicted sigma over the 28 holes: med {np.median(v):.3f} "
          f"[{v.min():.3f},{v.max():.3f}] px")
    arms = [("base", 0.0)]
    arms += [("const", c) for c in (0.15, 0.3, 0.5, 0.8)]
    arms += [("pred", g) for g in (0.15, 0.3, 0.5)]
    jobs = [(s, k, p, lut[s]) for k, p in arms for s in stems]
    from multiprocessing import Pool
    with Pool(6) as pool:
        res = pool.map(job, jobs, chunksize=2)
    agg = {}
    for stem, k, p, P, S, L in res:
        agg.setdefault((k, p), []).append((P, S, L))
    base = None
    print(f"\n{'arm':>16} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'SCORE':>9} {'dScore':>8}")
    for k, p in arms:
        A = np.array(agg[(k, p)])
        P, S, L = A.mean(0)
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * P / 50)
        if base is None:
            base = sc
        tag = "base" if k == "base" else (f"const s={p}" if k == "const" else f"pred g={p}")
        print(f"{tag:>16} {P:8.4f} {S:8.4f} {L:8.4f} {sc:9.4f} {sc-base:+8.4f}")


if __name__ == "__main__":
    main()
