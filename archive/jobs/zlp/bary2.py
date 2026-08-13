#!/usr/bin/env python
"""LPIPS-BARYCENTER, steepest-descent line search (Adam was injecting uniform noise).

x(eta) = ens - eta * G,   G = grad_x mean_i LPIPS(x, m_i) |_{x=ens},  G normalised to unit RMS.
eta is then literally "RMS pixel change" in [0,1] units. One VGG fwd/bwd per member per image;
every eta on the line is free. Also tested on top of the shipped energy-restore output.

CONTROLS
  ctrl_rand   ens - eta * N, N = spatially shuffled G (same magnitude histogram, no location
              information). Separates "LPIPS gradient geometry" from "any perturbation of this
              size". If ctrl_rand ~ real, the gradient carries nothing.
  ctrl_1mem   gradient of LPIPS(x, m_0) only -> consensus vs single-member texture pull.
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


def energy_restore(ens, members, lam, k, win=3, nlev=5, clamp=4.0, mem_l0=None):
    K = _K.to(ens.device)
    laps, res, sizes = lap_pyr(ens, nlev, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    V = 0.0
    for ml in mem_l0:
        V = V + boxf(((ml - L0) ** 2).sum(1, keepdim=True), win)
    V = V / len(mem_l0) * (k / (k - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)
    return lap_recon([L0 * (1.0 + lam * (r - 1.0))] + laps[1:], res, sizes, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--etas", type=float, nargs="*", default=[0.002, 0.004, 0.008, 0.016])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--layers", action="store_true")
    args = ap.parse_args()
    dev = args.device

    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    stems = stems[:args.n]
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"),
                    *[int(x) for x in cache["HW"]], "cubic")

    @torch.no_grad()
    def layer_lpips(a, b):
        i0, i1 = vgg.scaling_layer(a * 2 - 1), vgg.scaling_layer(b * 2 - 1)
        f0, f1 = vgg.net.forward(i0), vgg.net.forward(i1)
        return [float(vgg.lins[k]((lpips_pkg.normalize_tensor(f0[k]) -
                                  lpips_pkg.normalize_tensor(f1[k])) ** 2).mean())
                for k in range(vgg.L)]

    def grad_of(x0, targets):
        x = x0.clone().requires_grad_(True)
        for t in targets:
            (vgg(x * 2 - 1, t * 2 - 1).mean() / len(targets)).backward()
        G = x.grad.detach()
        return G / (G.pow(2).mean().sqrt() + 1e-12)     # unit RMS

    arms = ["base", "ship_lam1.0"]
    arms += [f"gd{e}" for e in args.etas]
    arms += [f"ship+gd{e}" for e in args.etas]
    if args.controls:
        e = args.etas[len(args.etas) // 2]
        arms += [f"CTRL_rand{e}", f"CTRL_1mem{e}"]

    acc = {a: [0.0, 0.0, 0.0] for a in arms}
    lay = {a: np.zeros(5) for a in arms}
    t0 = time.time()

    for c, s in enumerate(stems):
        mem = [load(os.path.join(d, s + ".png"), dev) for d in MEM]
        ens = torch.stack(mem).mean(0)
        g = load(os.path.join(GTD, gt_by[s]), dev)
        K = _K.to(dev)
        mem_l0 = [lap_pyr(m, 5, K)[0][0] for m in mem]          # hoisted, shared by all arms
        ship = energy_restore(ens, mem, 1.0, len(mem), mem_l0=mem_l0)

        G = grad_of(ens, mem)
        Gs = grad_of(ship, mem)
        outs = {"base": ens, "ship_lam1.0": ship}
        for e in args.etas:
            outs[f"gd{e}"] = ens - e * G
            outs[f"ship+gd{e}"] = ship - e * Gs
        if args.controls:
            e = args.etas[len(args.etas) // 2]
            flat = G.reshape(-1)
            outs[f"CTRL_rand{e}"] = ens - e * flat[torch.randperm(
                flat.numel(), device=dev)].reshape(G.shape)
            outs[f"CTRL_1mem{e}"] = ens - e * grad_of(ens, [mem[0]])

        for a in arms:
            x = outs[a].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            x = np.clip(warp(x, lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                acc[a][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[a][1] += float(repo_ssim(r, g))
                acc[a][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
                if args.layers:
                    lay[a] += np.array(layer_lpips(r, g))
        if c % 5 == 0:
            el = time.time() - t0
            print(f"  {c+1}/{len(stems)} {el:.0f}s eta {el/(c+1)*(len(stems)-c-1):.0f}s",
                  flush=True)

    n = len(stems)
    print(f"\nFULL SHIPPED CHAIN, {TAG}, n={n}, k=4 UT pool")
    print(f"{'arm':>16} {'SCORE':>9} {'dScore':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8}")
    base = None
    for a in arms:
        P, S, L = (x / n for x in acc[a])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        if a == "base":
            base = sc
        print(f"{a:>16} {sc:9.4f} {sc-base:+9.4f} {P:8.4f} {S:7.4f} {L:8.4f}")
    if args.layers:
        print(f"\n{'arm':>16} " + " ".join(f"{x:>10}" for x in
              ("relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3")))
        for a in arms:
            print(f"{a:>16} " + " ".join(f"{v/n:10.5f}" for v in lay[a]))


if __name__ == "__main__":
    main()
