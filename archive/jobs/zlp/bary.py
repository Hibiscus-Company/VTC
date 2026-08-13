#!/usr/bin/env python
"""LPIPS-BARYCENTER FUSION.

Diagnosis (from I_lpips.py, reconfirmed here per-layer): only 13.4% of our LPIPS lives in
relu1_2 -- the ONLY band the shipped energy-restore operator touches. 25.4% is in relu2_2,
45% in relu4_3+relu5_3. Laplacian band gain is provably exhausted (n2 -0.012, n3 -0.019,
n5 worse than n1; global gain +0.06 max; attenuation negative).

The pixel mean is the minimiser of E||x - m||^2 over the member posterior. It is NOT the
minimiser of E LPIPS(x, m). If GT behaves like another draw from the same posterior the members
are drawn from, the LPIPS-optimal point estimate is the LPIPS BARYCENTER of the members:

    x* = argmin_x  mean_i LPIPS(x, m_i)        (members only -- no GT, no per-image tuning)

This descends ALL FIVE VGG layers, not just the finest band. Started at x = pixel mean, run
with Adam, snapshotted at several step counts (early stopping = trust region on the PSNR loss).

CONTROLS
  bary1_*   same optimisation against a SINGLE member -> isolates "member consensus" from
            "move toward any natural image's texture statistics" (= texture injection).
  baryC_*   same optimisation against the MEAN ITSELF as the only target -> the mean is a
            stationary point, so any movement is pure metric-hacking drift. Detects whether
            the gain is adversarial LPIPS texture rather than information.

Everything is scored through the FULL SHIPPED CHAIN: operator -> median lens field warped with
INTER_LANCZOS4 -> JPEG quality=100 subsampling=2 optimize progressive -> decode -> metrics
against the REAL test GT of public_set/HCM0181.
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


def energy_restore(ens, members, lam, k, win=3, nlev=5, clamp=4.0):
    """The shipped r28 operator (level-0 only)."""
    K = _K.to(ens.device)
    laps, res, sizes = lap_pyr(ens, nlev, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    V = 0.0
    for m in members:
        ml = lap_pyr(m, nlev, K)[0][0]
        V = V + boxf(((ml - L0) ** 2).sum(1, keepdim=True), win)
    V = V / len(members) * (k / (k - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)
    return lap_recon([L0 * (1.0 + lam * (r - 1.0))] + laps[1:], res, sizes, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--snaps", type=int, nargs="*", default=[6, 12, 20, 40])
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--layers", action="store_true", help="also report per-VGG-layer LPIPS")
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

    # ---------------- per-layer LPIPS (shares the same net) ----------------
    @torch.no_grad()
    def layer_lpips(a, b):
        i0, i1 = vgg.scaling_layer(a * 2 - 1), vgg.scaling_layer(b * 2 - 1)
        f0, f1 = vgg.net.forward(i0), vgg.net.forward(i1)
        out = []
        for k in range(vgg.L):
            d = (lpips_pkg.normalize_tensor(f0[k]) - lpips_pkg.normalize_tensor(f1[k])) ** 2
            out.append(float(vgg.lins[k](d).mean()))
        return out

    def bary(x0, targets, steps, snaps, lr):
        """argmin_x mean_j LPIPS(x, targets[j]); returns {step: image}."""
        x = x0.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        out = {}
        for s in range(1, max(snaps) + 1):
            opt.zero_grad(set_to_none=True)
            for t in targets:                       # backward per target: 1x VGG graph in mem
                (vgg(x * 2 - 1, t * 2 - 1).mean() / len(targets)).backward()
            opt.step()
            if s in snaps:
                out[s] = x.detach().clamp(0, 1).clone()
        return out

    arms = ["base", "ship_lam1.0"]
    arms += [f"bary{s}" for s in args.snaps]
    arms += [f"bary{s}+ship" for s in args.snaps]
    if args.controls:
        arms += [f"CTRL_bary1_{args.snaps[-1]}", f"CTRL_self_{args.snaps[-1]}"]

    acc = {a: [0.0, 0.0, 0.0] for a in arms}
    lay = {a: np.zeros(5) for a in arms}
    nbytes = {a: 0 for a in arms}
    t0 = time.time()

    for c, s in enumerate(stems):
        mem = [load(os.path.join(d, s + ".png"), dev) for d in MEM]
        ens = torch.stack(mem).mean(0)
        g = load(os.path.join(GTD, gt_by[s]), dev)

        outs = {"base": ens,
                "ship_lam1.0": energy_restore(ens, mem, 1.0, len(mem))}
        snapd = bary(ens, mem, args.steps, args.snaps, args.lr)
        for k, v in snapd.items():
            outs[f"bary{k}"] = v
            outs[f"bary{k}+ship"] = energy_restore(v, mem, 1.0, len(mem))
        if args.controls:
            L = args.snaps[-1]
            outs[f"CTRL_bary1_{L}"] = bary(ens, [mem[0]], L, [L], args.lr)[L]
            outs[f"CTRL_self_{L}"] = bary(ens, [ens], L, [L], args.lr)[L]

        for a in arms:
            x = outs[a].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            x = np.clip(warp(x, lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
            nbytes[a] += len(b.getvalue())
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
            print(f"  {c+1}/{len(stems)}  {el:.0f}s  eta {el/(c+1)*(len(stems)-c-1):.0f}s",
                  flush=True)

    n = len(stems)
    print(f"\nFULL SHIPPED CHAIN (op -> median field lanczos4 -> JPEG q100/ss2), {TAG}, "
          f"n={n}, k=4 UT pool, lr={args.lr}")
    print(f"{'arm':>16} {'SCORE':>9} {'dScore':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'MB/60':>7}")
    base = None
    for a in arms:
        P, S, L = (x / n for x in acc[a])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        if a == "base":
            base = sc
        print(f"{a:>16} {sc:9.4f} {sc-base:+9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {nbytes[a]/1e6:7.2f}")
    if args.layers:
        print(f"\n{'arm':>16} " + " ".join(f"{x:>10}" for x in
              ("relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3")))
        for a in arms:
            print(f"{a:>16} " + " ".join(f"{v/n:10.5f}" for v in lay[a]))


if __name__ == "__main__":
    main()
