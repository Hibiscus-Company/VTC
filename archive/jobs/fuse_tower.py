#!/usr/bin/env python
"""FUSED tower stage: members -> weighted mean -> energy restore -> lens field, all in float32.

Drop-in replacement for the THREE staged calls in build_r28_energy.sh / build_r29.sh

    ensemble_renders.py  ->  png_ens   (uint8 PNG)
    energy_restore.py    ->  png_er    (uint8 PNG)
    apply_field.py       ->  png       (uint8 PNG)

Each of those stages writes a lossless-PNG *uint8* file that the next stage reads back, so the
tower chain quantizes to 8 bits three times before the encoder ever sees a pixel (four, if the
r22 mean PNG that feeds it is counted -- it enters at weight 0.8). Only the LAST one is free:
rounding right before the JPEG encoder is idempotent with the uint8 cast the encoder needs
anyway. This script keeps float32 all the way to the final write, so exactly one rounding
survives -- the irreducible one.

Semantics are byte-for-byte the same operators: the pyramid primitives come from lapfuse.py and
the warp from gsplat_track/fit_field.apply_field (INTER_LANCZOS4), same as production. The only
change is that no intermediate is round-tripped through uint8.

Output is the final warped PNG, so build_*.sh's assembler is untouched -- swap three python
calls for one and re-point nothing else.

    python fuse_tower.py \
      --member_dirs A B C --weights 4 1 1 \
      --field /mnt/d/avv/fields_median/HCM0421.npy \
      --lam 1.0 --k 8 --map_dirs M1 M2 ... --out_dir OUT/png
      [--verify_against /mnt/d/avv/r28e_v2/tower_ens/HCM0421/png]

--member_dirs / --weights define the ENSEMBLE (weights are normalized).
--map_dirs are the members that drive the disagreement map (production passes the individual
  models, not the pre-averaged r22 PNG); --k is the TRUE member count behind the ensemble.
--verify_against prints mean|diff| vs the staged build; it should be ~0.3/255 (rounding only).
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track")
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fit_field import apply_field

Image.MAX_IMAGE_PIXELS = None


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def restore(ens, mem_L0, lam, k, win=3, nlev=5, clamp=4.0):
    K = _K
    laps, res, sizes = lap_pyr(ens, nlev, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    V = 0.0
    for ml in mem_L0:
        V = V + boxf(((ml - L0) ** 2).sum(1, keepdim=True), win)
    V = V / len(mem_L0) * (k / (k - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)
    return lap_recon([L0 * (1.0 + lam * (r - 1.0))] + laps[1:], res, sizes, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member_dirs", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--map_dirs", nargs="+", required=True)
    ap.add_argument("--field", default=None, help="omit for the video scenes (no field)")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--verify_against", default=None)
    args = ap.parse_args()

    w = np.ones(len(args.member_dirs)) if args.weights is None else np.asarray(args.weights, float)
    assert len(w) == len(args.member_dirs), "--weights count != --member_dirs count"
    w = w / w.sum()

    field = np.load(args.field) if args.field else None
    if field is not None:
        amax = float(np.abs(field).max())
        assert 1e-6 < amax < 8.0, f"implausible field, max|d|={amax}"
        meta = args.field + ".meta.json"
        assert os.path.exists(meta), "field has no provenance -- refusing (Rule 10)"
        import json
        assert os.sep + "test" + os.sep not in json.load(open(meta)).get("gt_dir", "") + os.sep, \
            "field was fit against TEST GT -- refusing (Rule 10)"

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.member_dirs[0])
                   if f.lower().endswith(".png"))
    for d in list(args.member_dirs) + list(args.map_dirs):
        have = {os.path.splitext(f)[0] for f in os.listdir(d)}
        miss = [s for s in stems if s not in have]
        assert not miss, f"{d} missing {len(miss)} stems e.g. {miss[:2]}"

    os.makedirs(args.out_dir, exist_ok=True)
    diffs = []
    with torch.no_grad():
        for s in stems:
            ens = None
            for wi, d in zip(w, args.member_dirs):
                a = load(os.path.join(d, s + ".png")) * float(wi)
                ens = a if ens is None else ens + a
            mem_L0 = [lap_pyr(load(os.path.join(d, s + ".png")), 5, _K)[0][0]
                      for d in args.map_dirs]
            out = restore(ens, mem_L0, args.lam, args.k).clamp(0, 1) if args.lam else ens.clamp(0, 1)
            x = np.ascontiguousarray(out[0].permute(1, 2, 0).numpy())
            if field is not None:
                x = np.clip(apply_field(x, field), 0.0, 1.0)
            arr = (x * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(args.out_dir, s + ".png"))
            if args.verify_against:
                ref = np.asarray(Image.open(os.path.join(args.verify_against, s + ".png")
                                            ).convert("RGB"), dtype=np.float32)
                diffs.append(np.abs(arr.astype(np.float32) - ref).mean())
    print(f"fused {len(stems)} images -> {args.out_dir}  (lam={args.lam} k={args.k} "
          f"{len(args.member_dirs)} weighted members, {len(args.map_dirs)} map members)")
    if diffs:
        print(f"vs staged build: mean|diff| {np.mean(diffs):.4f}/255  max-image {np.max(diffs):.4f}"
              f"   (expect ~0.2-0.5 = the removed roundings; a big number means a real mismatch)")


if __name__ == "__main__":
    main()
