"""P1b: lean JPEG-lattice probe. Fewer variants, fewer images, both scene families."""
import os, sys, json
import numpy as np
import torch
import mlib

torch.set_num_threads(int(os.environ.get("NT", "10")))

VARIANTS = [
    ("png_lossless", None),
    ("q100_ss2_SHIPPED", dict(quality=100, subsampling=2)),
    ("q100_ss0", dict(quality=100, subsampling=0)),
    ("q98_ss2", dict(quality=98, subsampling=2)),
    ("q96_ss2", dict(quality=96, subsampling=2)),
    ("q95_ss2_GTLATTICE", dict(quality=95, subsampling=2)),
    ("q95_ss0", dict(quality=95, subsampling=0)),
    ("q92_ss2", dict(quality=92, subsampling=2)),
]

CASES = {
    "chair": ("/mnt/d/avv/chair_eval/base60k/eval_png", "/mnt/d/avv/evalsplit/chair/eval_gt", []),
    "HCM0421": ("/mnt/d/avv/evalgen/HCM0421/eval_png", "/mnt/d/avv/evalsplit/HCM0421/eval_gt",
                ["DJI_20241230093301_0003_V"]),
    "bonsai": ("/mnt/d/avv/bonsai_eval/K4_pC_seed7/eval_png", "/mnt/d/avv/evalsplit/bonsai2/eval_gt", []),
}

N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
key = sys.argv[1]
rd, gd, skip = CASES[key]
lp = mlib.LP("cpu")
gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gd)}
files = sorted(f for f in os.listdir(rd) if f.lower().endswith((".png", ".jpg")))
files = [f for f in files if os.path.splitext(f)[0] not in skip]
files = files[:: max(1, len(files) // N)][:N]

acc = {n: [0.0, 0.0, 0.0, 0.0, 0] for n, _ in VARIANTS}
for i, f in enumerate(files):
    s = os.path.splitext(f)[0]
    r8 = mlib.load_u8(os.path.join(rd, f))
    g = mlib.to_t(mlib.load_u8(os.path.join(gd, gt_by[s])))
    for n, kw in VARIANTS:
        e8, nb = (r8, 0) if kw is None else mlib.jpeg_roundtrip(r8, **kw)
        t = mlib.to_t(e8)
        acc[n][0] += mlib.psnr(t, g)
        acc[n][1] += float(mlib.ssim(t, g))
        acc[n][2] += lp(t, g)
        acc[n][3] += nb / 1e6
        acc[n][4] += 1
    print(f"  [{i+1}/{len(files)}] {s}", flush=True)

rows = []
for n, _ in VARIANTS:
    P, S, L, MB, k = acc[n]
    rows.append((n, P / k, S / k, L / k, MB / k, mlib.score(P / k, S / k, L / k)))
ref = [r for r in rows if r[0] == "q100_ss2_SHIPPED"][0]
print(f"\n=== P1b {key} n={len(files)} ===")
for n, P, S, L, MB, sc in rows:
    print(f"{n:20s} PSNR {P:7.4f}({P-ref[1]:+.4f})  SSIM {S:.5f}({S-ref[2]:+.5f})  "
          f"LPIPS {L:.5f}({L-ref[3]:+.5f})  MB {MB:5.2f}  SCORE {sc:8.4f}  d {sc-ref[5]:+.4f}")
json.dump(rows, open(f"/mnt/d/avv/metric_probe/p1b_{key}.json", "w"), indent=1)
