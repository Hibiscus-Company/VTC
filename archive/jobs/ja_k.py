#!/usr/bin/env python
"""jpegalloc STAGE 4 -- THE DECONFOUNDING CONTROL.

q98/ss0 beat the shipped q100/ss2 by +0.0249 on HCM0181 (4-member ensemble) and LOST by -0.0077
on HCM0193. Those two runs differ in TWO things at once: the scene, and the ensemble depth
(HCM0193 has a single member render, HCM0181 has four). Only one of those explanations survives
contact with production, which ships k=7-8 on every scene:

  H1 "it is the scene"           -> the gain is an HCM0181 quirk and MUST NOT be shipped.
  H2 "it is the ensemble depth"  -> JPEG's HF quantization noise substitutes for the texture that
                                    pixel-averaging destroys, so the gain GROWS with k and the
                                    production number is LARGER than +0.0249.

This script holds the SCENE FIXED at HCM0181 and sweeps k in {1,2,4}. Same GT, same field, same
poses -- only the ensemble depth moves. If delta(q98ss0 - q100ss2) is negative at k=1 and climbs
with k, H2 is confirmed and H1 is dead.

The k=4 no-restore arm asks a second question production needs answered: energy restoration already
puts finest-band energy back, so is the JPEG-noise gain a SUBSTITUTE for it (would vanish once
restore is on) or a COMPLEMENT (survives)?
"""
import io, os, sys, time
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
# (config name, k, use_restore)
CFG = [("k1", 1, False), ("k2", 2, True), ("k4", 4, True), ("k4_norestore", 4, False)]
ENC = [("q100ss2", 100, 2), ("q98ss0", 98, 0)]


def enc_dec(x, q, ss):
    im = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q, subsampling=ss, **SHIP)
    raw = b.getvalue()
    return np.ascontiguousarray(
        np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255.0), len(raw)


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    tag = "HCM0181"
    MEM = MEMS[tag]
    gtd = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    stems = stems[::max(1, len(stems) // N)][:N]
    c = np.load(f"{HERE}/lens/cache/pub_{tag}.npz")
    lens = upsample(LooPool(c["s8"]).pooled("median"),
                    *[int(x) for x in c["HW"]], "cubic") * 1.30
    print(f"{tag}: k-sweep on {len(stems)} poses, dev={dev}", flush=True)

    per = {(cn, en): [] for cn, _, _ in CFG for en, _, _ in ENC}
    nb = {(cn, en): 0 for cn, _, _ in CFG for en, _, _ in ENC}
    t0 = time.time()
    for ci, s in enumerate(stems):
        mem = []
        for d in MEM:
            a = np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                           dtype=np.float32) / 255.0
            mem.append(torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev))
        g = torch.from_numpy(
            np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                       dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        for cn, k, ur in CFG:
            sub = mem[:k]
            ens = torch.stack(sub).mean(0)
            if ur and k > 1:
                ens = restore(ens, sub, 1.0, k, 3)
            x = np.clip(warp(ens.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy(),
                             lens, "lanczos"), 0, 1)
            for en, q, ss in ENC:
                d, nn = enc_dec(x, q, ss)
                nb[(cn, en)] += nn
                r = torch.from_numpy(d).permute(2, 0, 1).unsqueeze(0).to(dev)
                with torch.no_grad():
                    p = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                    sv = float(repo_ssim(r, g))
                    lv = float(vgg(r * 2 - 1, g * 2 - 1).item())
                per[(cn, en)].append((p, sv, lv))
        if ci % 10 == 0:
            print(f"  {ci}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    n = len(stems)

    def isc(v):
        v = np.asarray(v)
        return 100 * (0.4 * (1 - v[:, 2]) + 0.3 * v[:, 1] + 0.3 * np.minimum(v[:, 0] / 50, 1))

    print(f"\n=== HCM0181 ENSEMBLE-DEPTH SWEEP, n={n} ===")
    print("delta = q98ss0 - q100ss2, paired per image")
    print(f"{'config':>14} {'base(q100ss2)':>14} {'delta':>9} {'se':>7} {'t':>7} {'wins':>8} "
          f"{'dMB/scene':>10}")
    for cn, k, ur in CFG:
        b = isc(per[(cn, "q100ss2")])
        a = isc(per[(cn, "q98ss0")])
        d = a - b
        se = d.std(ddof=1) / np.sqrt(n)
        dmb = (nb[(cn, "q98ss0")] - nb[(cn, "q100ss2")]) / 1e6 / n * 60
        print(f"{cn:>14} {b.mean():14.4f} {d.mean():+9.4f} {se:7.4f} {d.mean()/se:7.2f} "
              f"{int((d>0).sum()):>5}/{n:<3} {dmb:+10.2f}")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
