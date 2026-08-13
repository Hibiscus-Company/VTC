#!/usr/bin/env python
"""EXACT production byte cost of every candidate encode setting, on the real r29 PNGs.

Score is measured on the public harness; BYTES must be measured on the private artefact, because
the 350 MiB cap is the only thing that ever bound the encode axis. Reads the same PNG dirs the
r29 assembler read, so the 'base' column reproduces the shipped zip byte-for-byte.
"""
import io, os, json, sys
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.MAXBLOCK = 1 << 26

SH = dict(quality=100, subsampling=2, optimize=True, progressive=True)
SETTINGS = {
    "base_q100ss2": dict(SH),
    "q99ss2": dict(SH, quality=99),
    "q98ss2": dict(SH, quality=98),
    "q97ss2": dict(SH, quality=97),
    "q95ss2": dict(SH, quality=95),
    "q100ss1": dict(SH, subsampling=1),
    "q100ss0": dict(SH, subsampling=0),
    "rgb_q100": dict(SH, subsampling=0, keep_rgb=True),
    "rgb_q97": dict(SH, subsampling=0, keep_rgb=True, quality=97),
    "rgb_q95": dict(SH, subsampling=0, keep_rgb=True, quality=95),
    "rgb_q92": dict(SH, subsampling=0, keep_rgb=True, quality=92),
    "rgb_q90": dict(SH, subsampling=0, keep_rgb=True, quality=90),
    "rgb_q87": dict(SH, subsampling=0, keep_rgb=True, quality=87),
    "noprog": dict(SH, progressive=False),
}
SRC = {**{t: f"/mnt/d/avv/r29/tower_ens/{t}/png" for t in
          ("HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674")},
       "chair": "/mnt/d/avv/r29/video_ens/chair/png",
       "bonsai": "/mnt/d/avv/r29/video_ens/bonsai/png"}


def one(job):
    scene, path = job
    im = Image.open(path).convert("RGB")
    out = {}
    for k, kw in SETTINGS.items():
        b = io.BytesIO()
        im.save(b, "JPEG", **kw)
        out[k] = b.tell()
    return scene, out


def main():
    jobs = []
    for scene, d in SRC.items():
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".png"):
                jobs.append((scene, os.path.join(d, f)))
    print(f"{len(jobs)} images x {len(SETTINGS)} settings", flush=True)
    tot = {s: {k: 0 for k in SETTINGS} for s in SRC}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, (scene, out) in enumerate(ex.map(one, jobs, chunksize=4)):
            for k, v in out.items():
                tot[scene][k] += v
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    # zip container overhead: ZIP_STORED, 30 + 2*len(name) + 46 per entry + EOCD
    OVER = sum(30 + 46 + 2 * len(os.path.basename(p)) + 2 * (len(s) + 1) for s, p in jobs) + 22
    print(f"\n{'setting':>14} " + " ".join(f"{s:>9}" for s in sorted(SRC)) + f" {'TOTAL_MB':>10} {'MiB':>8}")
    for k in SETTINGS:
        row = [tot[s][k] for s in sorted(SRC)]
        T = sum(row) + OVER
        print(f"{k:>14} " + " ".join(f"{v/1e6:9.2f}" for v in row) + f" {T/1e6:10.2f} {T/1048576:8.2f}")
    json.dump(dict(tot=tot, over=OVER), open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
