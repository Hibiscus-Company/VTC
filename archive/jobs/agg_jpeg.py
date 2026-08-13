"""JPEG round-trip test: score chosen aggregators BEFORE and AFTER the shipped encode
(quality=100, subsampling=2, optimize, progressive). Standing insight: at production
quality the shipped JPEG BEATS lossless PNG because its structured artifacts substitute
for the per-view texture pixel-mean averaging destroys -- so a *smoother* aggregator may
give back less, and a *sharper* one may gain less. Must be measured post-encode.

usage: python agg_jpeg.py <pool> <agg_name> [<agg_name> ...]
"""
import os, sys, io, json
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_lib as A
import agg_core as C
from agg_run import POOLS, registry

ENC = dict(format="JPEG", quality=100, subsampling=2, optimize=True, progressive=True)


def rt(img):
    b = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, **ENC)
    n = b.tell()
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"), dtype=np.float32) / 255.0, n


def main():
    pool = sys.argv[1]
    want = sys.argv[2:]
    variants = POOLS[pool]
    dev = A.init()
    stems, gt_by = A.stems_for(variants)
    R = registry(len(variants))
    miss = [w for w in want if w not in R]
    if miss:
        sys.exit(f"unknown aggregators: {miss}")
    per = {w: {"png": [], "jpg": []} for w in want}
    bytes_ = {w: 0 for w in want}
    for i, s in enumerate(stems):
        X = np.stack([A.load(A.mdir(v), s) for v in variants], 0)
        g = A.gt_tensor(gt_by[s], dev)
        for w in want:
            img = R[w](X)
            per[w]["png"].append(A.metrics(img, g, dev))
            j, nb = rt(img)
            bytes_[w] += nb
            per[w]["jpg"].append(A.metrics(j, g, dev))
        del X, g
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(stems)}", flush=True)
    p = f"/home/bkai/.claude/jobs/1c9cf7e9/tmp/aggjpeg_{pool}.json"
    json.dump({k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
               for k, v in per.items()}, open(p, "w"))

    def comp(a):
        a = np.asarray(a); P, S, L = a.mean(0)
        return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    bm_png, bm_jpg = comp(per["mean"]["png"]), comp(per["mean"]["jpg"])
    print(f"\npool={pool}  views={len(stems)}   [baseline mean: PNG {bm_png:.4f} / "
          f"JPG {bm_jpg:.4f}, jpeg tax {bm_jpg-bm_png:+.4f}]")
    print(f"{'aggregator':<22}{'PNG':>9}{'dPNG':>8}{'JPG':>9}{'dJPG':>8}"
          f"{'jpegtax':>9}{'MB/60v':>8}")
    for w in want:
        a, b = comp(per[w]["png"]), comp(per[w]["jpg"])
        print(f"{w:<22}{a:9.4f}{a-bm_png:+8.4f}{b:9.4f}{b-bm_jpg:+8.4f}"
              f"{b-a:+9.4f}{bytes_[w]/1e6:8.1f}")


if __name__ == "__main__":
    main()
