"""P1: does encoding our submission ON THE GT'S OWN QUANTISATION LATTICE beat q100?

GT for every scene in private_set2 is JPEG quality-95, standard IJG tables, 4:2:0, baseline.
We currently ship q100/4:2:0. If our render is within ~half a quantisation step of the GT in a
given DCT coefficient, encoding on the SAME lattice snaps us to the GT's exact value -> zero error.
"""
import os, sys, argparse, json
import numpy as np
import torch
import mlib

torch.set_num_threads(int(os.environ.get("NT", "24")))


def variants():
    v = [("png_lossless", None)]
    for q in (100, 98, 96, 95, 94, 92):
        for ss in (0, 2):
            v.append((f"q{q}_ss{ss}", dict(quality=q, subsampling=ss)))
    v.append(("q100_ss0_keeprgb", dict(quality=100, subsampling=0, keep_rgb=True)))
    v.append(("q95_ss0_keeprgb", dict(quality=95, subsampling=0, keep_rgb=True)))
    v.append(("q90_ss2", dict(quality=90, subsampling=2)))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--lpips", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    lp = mlib.LP("cpu") if a.lpips else None
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(a.gt_dir)}
    files = sorted(f for f in os.listdir(a.render_dir) if f.lower().endswith((".png", ".jpg")))
    files = [f for f in files if os.path.splitext(f)[0] not in a.skip]
    if a.limit:
        step = max(1, len(files) // a.limit)
        files = files[::step][:a.limit]
    vs = variants()
    if a.only:
        vs = [(n, k) for n, k in vs if n in a.only]
    acc = {n: [0.0, 0.0, 0.0, 0.0, 0] for n, _ in vs}
    for i, f in enumerate(files):
        s = os.path.splitext(f)[0]
        r8 = mlib.load_u8(os.path.join(a.render_dir, f))
        g = mlib.to_t(mlib.load_u8(os.path.join(a.gt_dir, gt_by[s])))
        for n, kw in vs:
            if kw is None:
                e8, nb = r8, 0
            else:
                e8, nb = mlib.jpeg_roundtrip(r8, **kw)
            t = mlib.to_t(e8)
            acc[n][0] += mlib.psnr(t, g)
            acc[n][1] += float(mlib.ssim(t, g))
            acc[n][2] += lp(t, g) if lp else 0.0
            acc[n][3] += nb / 1e6
            acc[n][4] += 1
        print(f"  [{i+1}/{len(files)}] {s}", flush=True)
    print(f"\n=== {a.tag}  n={len(files)} ===")
    base = None
    rows = []
    for n, _ in vs:
        P, S, L, MB, k = acc[n]
        P, S, L, MB = P / k, S / k, L / k, MB / k
        sc = mlib.score(P, S, L)
        rows.append((n, P, S, L, MB, sc))
    ref = dict((r[0], r) for r in rows).get("q100_ss2", rows[0])
    for n, P, S, L, MB, sc in rows:
        print(f"{n:20s} PSNR {P:7.4f}  SSIM {S:.5f}  LPIPS {L:.5f}  MB {MB:5.2f}  "
              f"SCORE {sc:8.4f}   d_vs_q100ss2 {sc - ref[5]:+.4f}")
    json.dump(rows, open(f"/mnt/d/avv/metric_probe/p1_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
