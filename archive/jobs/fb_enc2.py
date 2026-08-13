#!/usr/bin/env python
"""Round 2 freebies: can the JPEG round-trip loss be pre-compensated for ZERO extra bytes?

Round 1 found the only positive encode arm is keep_rgb (Adobe APP14 transform=0, no RGB<->YCbCr
conversion) -- but it costs +145% bytes and does not fit the 350 MiB cap. That localises a real
loss channel: at fixed 4:4:4 chroma, removing the YCbCr integer round-trip alone moved the score.

PRE-COMPENSATION recovers part of it for free. The encoder is a deterministic map J: u8 -> u8.
One Newton step on J(X) = X0:
        X1 = round(X0 + mu * (X0 - J(X0)))
and we ship X1 instead of X0, so the DECODED image lands closer to the pixels we intended. Byte
cost is whatever the changed coefficients cost -- measured, not assumed. No retrain, no re-ensemble,
purely a build-time change to the encode step.

CONTROLS: ctrl_dup re-encodes the baseline a second time (encoder determinism / harness noise floor);
ctrl_noise adds the same-magnitude random perturbation as precomp1 instead of the correction (proves
the CORRECTION is load-bearing, not just extra dither).
"""
import argparse, io, os, sys, json, time
import numpy as np
import torch
from PIL import Image, ImageFile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from fieldlib import LooPool, upsample, warp              # noqa: E402
from energy_restore import restore                        # noqa: E402
from fb_encode import POOLS, u8                           # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ImageFile.MAXBLOCK = 1 << 26
SH = dict(quality=100, subsampling=2, optimize=True, progressive=True)
CACHE = "/mnt/d/avv/fb_cache"


def enc(arr_u8, kw):
    b = io.BytesIO()
    Image.fromarray(arr_u8).save(b, "JPEG", **kw)
    raw = b.getvalue()
    return raw, np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def build_cache(tag, lam, gain):
    d = os.path.join(CACHE, tag)
    os.makedirs(d, exist_ok=True)
    MEM = POOLS[tag]
    gtd = f"/mnt/d/avv/data/phase1/public_set/{tag}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(m, s + ".png")) for m in MEM))
    if all(os.path.exists(os.path.join(d, s + ".png")) for s in stems):
        return stems, gtd, gt_by
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = np.load(f"{HERE}/lens/cache/pub_{tag}.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"),
                    *[int(x) for x in cache["HW"]], "cubic") * gain
    for i, s in enumerate(stems):
        mem = [torch.from_numpy(np.asarray(Image.open(os.path.join(m, s + ".png")).convert("RGB"),
                                           dtype=np.float32) / 255.0
                                ).permute(2, 0, 1).unsqueeze(0).to(dev) for m in MEM]
        ens = torch.stack(mem).mean(0)
        ens_q = torch.from_numpy(u8(ens[0].permute(1, 2, 0).cpu().numpy()).astype(np.float32) / 255.
                                 ).permute(2, 0, 1).unsqueeze(0).to(dev)
        if len(MEM) > 1:
            e = restore(ens_q, mem, lam, len(MEM)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            e = u8(e).astype(np.float32) / 255.0
        else:
            e = ens_q[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(u8(np.clip(warp(e, lens, "lanczos"), 0, 1))).save(
            os.path.join(d, s + ".png"))
        if i % 20 == 0:
            print(f"  cache {tag} {i}/{len(stems)}", flush=True)
    return stems, gtd, gt_by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["HCM0181"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=1.30)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    rng = np.random.default_rng(0)

    ARMS = ["prod_q100ss2", "ctrl_dup", "ctrl_noise", "precomp_mu0.5", "precomp_mu1.0",
            "precomp_mu1.0x2", "rgb_q100", "rgb_q87"]
    allres = {}
    for tag in args.scenes:
        stems, gtd, gt_by = build_cache(tag, args.lam, args.gain)
        if args.n:
            stems = stems[:args.n]
        d = os.path.join(CACHE, tag)
        per = {a: [] for a in ARMS}
        nb = {a: 0 for a in ARMS}
        acc = {a: np.zeros(3) for a in ARMS}
        t0 = time.time()
        for c, s in enumerate(stems):
            X0 = np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"))
            g = torch.from_numpy(np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                                            dtype=np.float32) / 255.
                                 ).permute(2, 0, 1).unsqueeze(0).to(dev)
            raw0, J0 = enc(X0, SH)
            resid = X0.astype(np.float32) - J0.astype(np.float32)
            out = {"prod_q100ss2": (raw0, J0)}
            out["ctrl_dup"] = enc(X0, SH)
            npert = rng.permutation(resid.reshape(-1)).reshape(resid.shape)  # same histogram
            out["ctrl_noise"] = enc(np.clip(X0 + npert + 0.5, 0, 255).astype(np.uint8), SH)
            for mu in (0.5, 1.0):
                X1 = np.clip(X0.astype(np.float32) + mu * resid + 0.5, 0, 255).astype(np.uint8)
                out[f"precomp_mu{mu}"] = enc(X1, SH)
            X1 = np.clip(X0.astype(np.float32) + resid + 0.5, 0, 255).astype(np.uint8)
            r2 = X0.astype(np.float32) - out["precomp_mu1.0"][1].astype(np.float32)
            X2 = np.clip(X1.astype(np.float32) + r2 + 0.5, 0, 255).astype(np.uint8)
            out["precomp_mu1.0x2"] = enc(X2, SH)
            out["rgb_q100"] = enc(X0, dict(SH, subsampling=0, keep_rgb=True))
            out["rgb_q87"] = enc(X0, dict(SH, subsampling=0, keep_rgb=True, quality=87))
            for a in ARMS:
                raw, dec = out[a]
                nb[a] += len(raw)
                j = dec.astype(np.float32) / 255.
                r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
                with torch.no_grad():
                    P = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                    S = float(repo_ssim(r, g)); L = float(vgg(r * 2 - 1, g * 2 - 1).item())
                acc[a] += (P, S, L)
                per[a].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
            if c % 15 == 0:
                print(f"  {tag} {c}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
        n = len(stems)
        base = np.array(per["prod_q100ss2"])
        print(f"\n=== {tag} n={n} ===")
        print(f"{'arm':>16} {'SCORE':>9} {'dSCORE':>9} {'t':>6} {'wins':>7} {'PSNR':>8} "
              f"{'SSIM':>7} {'LPIPS':>7} {'MB':>7} {'dMB%':>7}")
        allres[tag] = {}
        for a in ARMS:
            P, S, L = acc[a] / n
            sc = float(np.mean(per[a])); dd = np.array(per[a]) - base
            t = dd.mean() / (dd.std(ddof=1) / np.sqrt(n)) if dd.std(ddof=1) > 1e-12 else 0.0
            allres[tag][a] = dict(score=sc, d=sc - base.mean(), t=float(t),
                                  wins=int((dd > 0).sum()), psnr=P, ssim=S, lpips=L, bytes=nb[a],
                                  dbytes_pct=100. * (nb[a] - nb["prod_q100ss2"]) / nb["prod_q100ss2"])
            print(f"{a:>16} {sc:9.4f} {sc-base.mean():+9.4f} {t:6.2f} "
                  f"{allres[tag][a]['wins']:3d}/{n:<3d} {P:8.4f} {S:7.4f} {L:7.4f} "
                  f"{nb[a]/1e6:7.2f} {allres[tag][a]['dbytes_pct']:+7.2f}")
        json.dump(allres, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
