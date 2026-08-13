#!/usr/bin/env python
"""ACHROMATIC-ONLY ENERGY RESTORE.

The shipped operator adds  lam*(r-1)*L0  to all three RGB channels of the finest band. The
shipped encoder is JPEG subsampling=2 (4:2:0), which throws away half the horizontal chroma
bandwidth immediately afterwards. So the chromatic part of that boost cannot survive to be
scored -- but it IS paid for, twice: in PSNR/SSIM as colour fringing on the sub-sampled grid,
and in bytes.

Split the boost b = lam*(r-1)*L0 in RGB space into the achromatic direction u and its
orthogonal complement:
    b_achro  = <b,u> u,     u = (1,1,1)/sqrt(3)        (survives 4:2:0)
    b_chroma = b - b_achro                             (mostly destroyed by 4:2:0)
and ship only b_achro. 'luma601' uses the BT.601 direction instead of (1,1,1).

CONTROL  ctrl_chroma_only: ship only b_chroma. If achro ~ full and chroma_only ~ 0, the
chromatic half of the shipped boost is confirmed dead weight. If chroma_only is positive the
split is wrong and this is just a lambda rescale -- which the lam ladder below also separates,
since achro at matched lam has strictly LESS total boost energy than full RGB.
"""
import argparse, io, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, upsample, warp
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg

Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
MEM = [f"/mnt/d/avv/output/HCM0181_{v}/test_poses_renders_png"
       for v in ("gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7")]
GTD = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"


def load(p, dev):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)


def restore(ens, mem_l0, lam, k, mode="rgb", win=3, nlev=5, clamp=4.0):
    K = _K.to(ens.device)
    laps, res, sizes = lap_pyr(ens, nlev, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    V = 0.0
    for ml in mem_l0:
        V = V + boxf(((ml - L0) ** 2).sum(1, keepdim=True), win)
    V = V / len(mem_l0) * (k / (k - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)
    b = lam * (r - 1.0) * L0                                   # the shipped boost, RGB
    if mode != "rgb":
        if mode == "luma601":
            u = torch.tensor([0.299, 0.587, 0.114], device=ens.device)
        else:
            u = torch.tensor([1.0, 1.0, 1.0], device=ens.device)
        u = (u / u.norm()).view(1, 3, 1, 1)
        b_a = (b * u).sum(1, keepdim=True) * u                 # projection onto u
        b = b_a if mode in ("achro", "luma601") else b - b_a   # else: chroma-only control
    return lap_recon([L0 + b] + laps[1:], res, sizes, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    dev = args.device
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:args.n]
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"),
                    *[int(x) for x in cache["HW"]], "cubic")

    SPEC = ([("base", None, 0.0)]
            + [(f"ship_rgb_lam{l}", "rgb", l) for l in (1.0, 1.25)]
            + [(f"achro_lam{l}", "achro", l) for l in (1.0, 1.25, 1.5, 1.75)]
            + [(f"luma601_lam{l}", "luma601", l) for l in (1.25,)]
            + [("CTRL_chromaonly_lam1.0", "chroma", 1.0)])
    arms = [s[0] for s in SPEC]
    acc = {a: [0.0, 0.0, 0.0] for a in arms}
    nb = {a: 0 for a in arms}
    t0 = time.time()

    for c, s in enumerate(stems):
        mem = [load(os.path.join(d, s + ".png"), dev) for d in MEM]
        ens = torch.stack(mem).mean(0)
        g = load(os.path.join(GTD, gt_by[s]), dev)
        K = _K.to(dev)
        mem_l0 = [lap_pyr(m, 5, K)[0][0] for m in mem]          # hoisted

        for name, mode, lam in SPEC:
            o = ens if mode is None else restore(ens, mem_l0, lam, len(mem), mode)
            x = o.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            x = np.clip(warp(x, lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
            nb[name] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                acc[name][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[name][1] += float(repo_ssim(r, g))
                acc[name][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
        if c % 5 == 0:
            el = time.time() - t0
            print(f"  {c+1}/{len(stems)} {el:.0f}s eta {el/(c+1)*(len(stems)-c-1):.0f}s",
                  flush=True)

    n = len(stems)
    print(f"\nFULL SHIPPED CHAIN, {TAG} real test GT, n={n}, k=4 UT pool")
    print(f"{'arm':>24} {'SCORE':>9} {'vs base':>9} {'vs ship':>9} {'PSNR':>8} {'SSIM':>7} "
          f"{'LPIPS':>8} {'MB/60':>7}")
    met = {a: tuple(x / n for x in acc[a]) for a in arms}
    sc = {a: 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
          for a, (P, S, L) in met.items()}
    for a in arms:
        P, S, L = met[a]
        print(f"{a:>24} {sc[a]:9.4f} {sc[a]-sc['base']:+9.4f} {sc[a]-sc['ship_rgb_lam1.0']:+9.4f} "
              f"{P:8.4f} {S:7.4f} {L:8.4f} {nb[a]/1e6:7.2f}")


if __name__ == "__main__":
    main()
