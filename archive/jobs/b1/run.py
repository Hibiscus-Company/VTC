#!/usr/bin/env python
"""B1 driver: score a list of named operator settings over a pair-set, with a per-frame cache.

Parallelises over FRAMES (GT VGG features are computed once per frame and reused for every
setting), which is the cheap layout.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse, hashlib, json, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/mnt/d/avv"
CACHE = "/mnt/d/avv/r42_bonsai78/b1_operator/cache"

PAIRSETS = {
    # name: (render_dir, render_ext, gt_dir, gt_ext)
    "fit":       (f"{ROOT}/blurbound/bonsai/train_render", "jpg",
                  f"{ROOT}/evalsplit/bonsai/train_sub/images", "jpg"),
    "check":     (f"{ROOT}/blurbound/bonsai/train_render", "jpg",
                  f"{ROOT}/evalsplit/bonsai/train_sub/images", "jpg"),
    "eval_sr01": (f"{ROOT}/r36_shape/sr01/eval_png", "png",
                  f"{ROOT}/evalsplit/bonsai/eval_gt", "jpg"),
    "eval_ema":  (f"{ROOT}/tw_test/bonsai_ema099/eval_png", "png",
                  f"{ROOT}/evalsplit/bonsai/eval_gt", "jpg"),
    "eval_sr01b": (f"{ROOT}/r36_shape/sr01b/eval_png", "png",
                   f"{ROOT}/evalsplit/bonsai/eval_gt", "jpg"),
    "eval_s101": (f"{ROOT}/r38/sr01_s101/eval_png", "png",
                  f"{ROOT}/evalsplit/bonsai/eval_gt", "jpg"),
}

HOLES = [10, 120, 190, 260, 440, 510, 630, 710, 810, 910, 1100, 1180, 1250, 1320,
         1400, 1470, 1540, 1640, 1720, 1820, 1930, 2120, 2190, 2260, 2330, 2440, 2510, 2650]


def train_frames():
    d = PAIRSETS["fit"][0]
    return sorted(int(f[6:12]) for f in os.listdir(d) if f.endswith(".jpg"))


def frames_for(pairset):
    if pairset.startswith("eval"):
        return [f"frame_{h:06d}" for h in HOLES]
    tr = train_frames()
    trs = set(tr)
    if pairset == "fit":
        out = []
        for h in HOLES:
            cand = h + 10 if (h + 10) in trs else h - 10
            out.append(cand)
        return [f"frame_{c:06d}" for c in sorted(set(out))]
    if pairset == "check":
        used = set()
        for h in HOLES:
            used.add(h + 10 if (h + 10) in trs else h - 10)
        rest = [t for t in tr if t not in used]
        idx = [round(i * (len(rest) - 1) / 27) for i in range(28)]
        return [f"frame_{rest[i]:06d}" for i in sorted(set(idx))]
    raise KeyError(pairset)


def skey(key):
    return hashlib.md5(key.encode()).hexdigest()[:10]


def worker(args):
    pairset, stem, settings, deliveries = args
    import core
    rd, rext, gd, gext = PAIRSETS[pairset]
    cdir = os.path.join(CACHE, pairset)
    os.makedirs(cdir, exist_ok=True)

    todo = []
    res = {}
    for name, params in settings:
        for dl in deliveries:
            cf = os.path.join(cdir, f"{stem}__{skey(name)}__{dl}.json")
            if os.path.exists(cf):
                try:
                    res[(name, dl)] = json.load(open(cf))
                    continue
                except Exception:
                    pass
            todo.append((name, params, dl, cf))
    if not todo:
        return stem, res

    g = core.load(os.path.join(gd, f"{stem}.{gext}"))
    feats = core.gt_feats(g)
    r0 = core.load(os.path.join(rd, f"{stem}.{rext}"))

    cache_img = {}
    for name, params, dl, cf in todo:
        if name not in cache_img:
            cache_img[name] = core.apply_op(r0, params)
        x = cache_img[name]
        nbytes = -1
        if dl == "float":
            y = x
        elif dl == "u8":
            y = core.quantize8(x)
        elif dl == "ship":
            y, nbytes = core.ship_encode(x)
        else:
            raise KeyError(dl)
        P, S, L = core.metrics_fast(y, g, feats)
        rec = dict(psnr=P, ssim=S, lpips=L, score=core.score_from(P, S, L), bytes=nbytes)
        json.dump(rec, open(cf, "w"))
        res[(name, dl)] = rec
    return stem, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairset", required=True)
    ap.add_argument("--settings", required=True, help="json file: [[name, params], ...]")
    ap.add_argument("--deliveries", default="float")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--nframes", type=int, default=0, help="deterministic subsample of the pairset")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    settings = json.load(open(a.settings))
    settings = [(n, p) for n, p in settings]
    dls = a.deliveries.split(",")
    frames = frames_for(a.pairset)
    if a.nframes and a.nframes < len(frames):
        idx = sorted({round(i * (len(frames) - 1) / (a.nframes - 1)) for i in range(a.nframes)})
        frames = [frames[i] for i in idx]
    t0 = time.time()
    jobs = [(a.pairset, s, settings, dls) for s in frames]
    with mp.Pool(a.procs, maxtasksperchild=2) as pool:
        out = {}
        for i, (stem, res) in enumerate(pool.imap_unordered(worker, jobs)):
            out[stem] = {f"{n}||{d}": v for (n, d), v in res.items()}
            print(f"[{i+1}/{len(frames)}] {stem} {time.time()-t0:.0f}s", flush=True)

    agg = {}
    for name, _ in settings:
        for d in dls:
            k = f"{name}||{d}"
            rows = [out[s][k] for s in frames]
            P = sum(r["psnr"] for r in rows) / len(rows)
            S = sum(r["ssim"] for r in rows) / len(rows)
            L = sum(r["lpips"] for r in rows) / len(rows)
            import core
            agg[k] = dict(psnr=P, ssim=S, lpips=L, score=core.score_from(P, S, L), n=len(rows))
    base = None
    for d in dls:
        bk = f"base||{d}"
        if bk in agg:
            base = agg[bk]
        print(f"\n=== {a.pairset}  delivery={d}  n={len(frames)}")
        rows = sorted([(k, v) for k, v in agg.items() if k.endswith(f"||{d}")],
                      key=lambda kv: -kv[1]["score"])
        b = agg.get(f"base||{d}")
        for k, v in rows:
            ds = v["score"] - b["score"] if b else float("nan")
            print(f"  {k.split('||')[0]:<34s} P{v['psnr']:8.4f} S{v['ssim']:.5f} "
                  f"L{v['lpips']:.5f}  score {v['score']:9.4f}  d {ds:+8.4f}")
    if a.out:
        json.dump(dict(pairset=a.pairset, frames=frames, per_frame=out, agg=agg),
                  open(a.out, "w"), indent=1)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
