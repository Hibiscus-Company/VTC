#!/usr/bin/env python
"""jpegalloc STAGE 3: allocate bytes ACROSS DCT FREQUENCIES, not just across scenes.

libjpeg's quality knob moves one scalar. Inspecting the tables shows q100 = all-ones and
q98 quantizes ONLY the high-frequency entries -- so the quality ladder is really a coarse,
coupled HF-quantization + chroma-quantization knob. Custom qtables decouple them:

    LUMA  : how much HF texture the encoder snaps to a lattice   (cheap bytes, adds texture)
    CHROMA: amplitude precision of the colour planes             (independent of ss)
    ss    : chroma RESOLUTION

The hypothesis this tests is that 4:2:0 spends the chroma budget the wrong way: it throws away
chroma RESOLUTION (edges) to keep chroma AMPLITUDE precision that nothing needs. Full-resolution
chroma with a coarse chroma table should dominate 4:2:0 with an exact table at equal bytes.
Same hoisted chain as ja_score.py; CONTROL arm is the shipped q100/ss2 and a lossless png.
"""
import argparse, io, os, sys, json, time
import numpy as np
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from energy_restore import restore
from fieldlib import LooPool, upsample, warp
from ja_score import MEMS

Image.MAX_IMAGE_PIXELS = None
SHIP = dict(optimize=True, progressive=True)


def flat(v):
    return [int(v)] * 64


def hf(lo, v, i0):
    """lo for the first i0 zigzag coefficients, v for the rest."""
    return [int(lo)] * i0 + [int(v)] * (64 - i0)


def grad(vmax, p=2.0):
    return [int(max(1, round(1 + ((i / 63.0) ** p) * (vmax - 1)))) for i in range(64)]


# (name, quality-or-None, qtables-or-None, subsampling)
ARMS = [
    ("q100ss2",      100, None, 2),                       # SHIPPED baseline
    ("q98ss0",        98, None, 0),                       # best scalar arm from stage 2
    ("L1_C4_ss0",   None, [flat(1), flat(4)], 0),         # chroma-amp term ALONE
    ("L1_C8_ss0",   None, [flat(1), flat(8)], 0),
    ("LHF2_C1_ss0", None, [hf(1, 2, 6), flat(1)], 0),     # luma-HF term ALONE
    ("LHF2_C4_ss0", None, [hf(1, 2, 6), flat(4)], 0),     # both
    ("LHF2_C8_ss0", None, [hf(1, 2, 6), flat(8)], 0),
    ("LHF3_C8_ss0", None, [hf(1, 3, 6), flat(8)], 0),     # stronger HF
    ("LHF2_ss2",    None, [hf(1, 2, 6), flat(1)], 2),     # luma HF only, shipped chroma path
]


def enc_dec(x, q, qt, ss):
    im = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
    b = io.BytesIO()
    if ss is None:
        im.save(b, "PNG", optimize=True)
    elif qt is not None:
        im.save(b, "JPEG", qtables=qt, subsampling=ss, **SHIP)
    else:
        im.save(b, "JPEG", quality=q, subsampling=ss, **SHIP)
    raw = b.getvalue()
    d = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(d), len(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="HCM0181")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=1.30)
    ap.add_argument("--field", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    tag = args.scene
    MEM = [d for d in MEMS[tag] if os.path.isdir(d)]
    gtd = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    if args.limit:
        stems = stems[::max(1, len(stems) // args.limit)][:args.limit]
    print(f"{tag}: {len(MEM)} members, {len(stems)} poses, lam={args.lam}", flush=True)

    lens = None
    if args.field:
        c = np.load(f"{HERE}/lens/cache/pub_{tag}.npz")
        lens = upsample(LooPool(c["s8"]).pooled("median"),
                        *[int(x) for x in c["HW"]], "cubic") * args.gain

    per = {a[0]: [] for a in ARMS}
    nb = {a[0]: 0 for a in ARMS}
    t0 = time.time()
    for ci, s in enumerate(stems):
        mem = []
        for d in MEM:
            a = np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                           dtype=np.float32) / 255.0
            mem.append(torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev))
        ens = torch.stack(mem).mean(0)
        if len(mem) > 1 and args.lam > 0:
            ens = restore(ens, mem, args.lam, len(mem), 3)
        x = ens.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        if lens is not None:
            x = np.clip(warp(x, lens, "lanczos"), 0, 1)
        g = torch.from_numpy(
            np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                       dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        for name, q, qt, ss in ARMS:
            d, n = enc_dec(x, q, qt, ss)
            nb[name] += n
            r = torch.from_numpy(d).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                p = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                sv = float(repo_ssim(r, g))
                lv = float(vgg(r * 2 - 1, g * 2 - 1).item())
            per[name].append((p, sv, lv))
        if ci % 10 == 0:
            print(f"  {ci}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    n = len(stems)
    A = {a[0]: np.asarray(per[a[0]]) for a in ARMS}

    def isc(v):
        return 100 * (0.4 * (1 - v[:, 2]) + 0.3 * v[:, 1] + 0.3 * np.minimum(v[:, 0] / 50, 1))

    b = isc(A["q100ss2"])
    bb = nb["q100ss2"] / 1e6
    print(f"\n=== {tag} QTABLE ALLOCATION, full shipped chain, n={n} ===")
    print(f"{'arm':>12} {'SCORE':>9} {'d':>9} {'se':>7} {'t':>6} {'wins':>7} {'MB':>7} "
          f"{'dMB':>7} {'dPSNR':>8} {'dSSIM':>9} {'dLPIPS':>10}")
    res = {}
    for name, _, _, _ in ARMS:
        d = isc(A[name]) - b
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        dd = A[name] - A["q100ss2"]
        mb = nb[name] / 1e6
        res[name] = dict(d=float(d.mean()), se=float(se), mb=mb, dmb=mb - bb,
                         score=float(isc(A[name]).mean()))
        print(f"{name:>12} {isc(A[name]).mean():9.4f} {d.mean():+9.4f} {se:7.4f} "
              f"{d.mean()/se if se>0 else 0:6.2f} {int((d>0).sum()):>4}/{n:<3} {mb:7.2f} "
              f"{mb-bb:+7.2f} {dd[:,0].mean():+8.4f} {dd[:,1].mean():+9.5f} {dd[:,2].mean():+10.5f}")
    json.dump(res, open(f"{HERE}/ja_qt_{tag}{args.tag}.json", "w"), indent=1)
    np.savez(f"{HERE}/ja_qtper_{tag}{args.tag}.npz", **A)
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
