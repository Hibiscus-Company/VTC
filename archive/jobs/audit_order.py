#!/usr/bin/env python
"""AUDIT part 2: ORDER OF OPERATIONS.

The shipped chain is  mean -> ENERGY RESTORE -> lens warp (lanczos4) -> JPEG.
The warp is a resample and measurably destroys 2.4% of Laplacian energy, so it attenuates
exactly the high frequency the operator just put back.  Test the alternative order
  mean -> lens warp -> ENERGY RESTORE -> JPEG
with the disagreement map rebuilt on the warped members (the field is the same for every
member and every view, so warping members is legal and cheap).

Shared computation (loads, warps, pyramids) is hoisted out of the arm loop.
"""
import argparse, io, os, sys, time, json
import numpy as np
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, boxf, _K   # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
C = 4.0 / 3.0


def u8(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_t(a, dev):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev).float().div_(255.0) \
        .permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=f"{HERE}/audit_order.json")
    args = ap.parse_args()
    dev = "cuda"
    from utils.loss_utils import ssim as repo_ssim
    from fieldlib import LooPool, upsample, warp
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:args.n]
    cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
    K = _K.to(dev)

    def W(t):
        a = t[0].permute(1, 2, 0).cpu().numpy()
        return np.clip(warp(np.ascontiguousarray(a), lens, "lanczos"), 0, 1)

    def np2t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev).permute(2, 0, 1).unsqueeze(0)

    def rmap(L0, mL0s):
        Eb = boxf((L0 ** 2).sum(1, keepdim=True), 3)
        V = sum(boxf(((m - L0) ** 2).sum(1, keepdim=True), 3) for m in mL0s) / len(mL0s)
        return torch.sqrt(1.0 + C * V / (Eb + 1e-10)).clamp(max=4.0)

    ARMS = ["base", "SHIP(before,1.0)", "AFTER,0.75", "AFTER,1.0", "AFTER,1.25",
            "BEFORE,1.25"]
    acc = {a: [0.0, 0.0, 0.0] for a in ARMS}
    t0 = time.time()
    for ci, s in enumerate(stems):
        mem = [to_t(u8(os.path.join(d, s + ".png")), dev) for d in MEM]
        ens = torch.stack(mem).mean(0)
        g = to_t(u8(os.path.join(GTD, gt_by[s])), dev)
        # --- unwarped analysis (shipped order) ---
        L0 = lap_pyr(ens, 5, K)[0][0]
        r_b = rmap(L0, [lap_pyr(m, 5, K)[0][0] for m in mem])
        # --- warped analysis (alternative order) ---
        ensw = np2t(W(ens))
        memw = [np2t(W(m)) for m in mem]
        L0w = lap_pyr(ensw, 5, K)[0][0]
        r_a = rmap(L0w, [lap_pyr(m, 5, K)[0][0] for m in memw])

        for a in ARMS:
            if a == "base":
                x = W(ens)
            elif a.startswith("SHIP") or a.startswith("BEFORE"):
                lam = float(a.split(",")[1].rstrip(")"))
                x = W((ens + lam * (r_b - 1.0) * L0).clamp(0, 1))
            else:
                lam = float(a.split(",")[1])
                x = (ensw + lam * (r_a - 1.0) * L0w).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            b = io.BytesIO()
            Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
                b, "JPEG", **SHIPPED_JPEG)
            y = to_t(np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                                dtype=np.uint8), dev)
            with torch.no_grad():
                acc[a][0] += 10 * np.log10(1.0 / max(float(((y - g) ** 2).mean()), 1e-12))
                acc[a][1] += float(repo_ssim(y, g))
                acc[a][2] += float(vgg(y * 2 - 1, g * 2 - 1).item())
        if (ci + 1) % 5 == 0:
            n = ci + 1
            bs = 100 * (0.4 * (1 - acc["base"][2] / n) + 0.3 * acc["base"][1] / n
                        + 0.3 * acc["base"][0] / n / 50)
            print(f"[{n}/{len(stems)} {time.time()-t0:.0f}s] " + " ".join(
                f"{a}{100*(0.4*(1-acc[a][2]/n)+0.3*acc[a][1]/n+0.3*acc[a][0]/n/50)-bs:+.4f}"
                for a in ARMS[1:]), flush=True)

    n = len(stems)
    print(f"\n=== ORDER OF OPERATIONS, HCM0181 real test GT, n={n} ===")
    print(f"{'arm':>18} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs base':>9} {'vs SHIP':>9}")
    sc = {}
    for a in ARMS:
        P, S, L = (v / n for v in acc[a])
        sc[a] = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    for a in ARMS:
        P, S, L = (v / n for v in acc[a])
        print(f"{a:>18} {sc[a]:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} "
              f"{sc[a]-sc['base']:+9.4f} {sc[a]-sc['SHIP(before,1.0)']:+9.4f}")
    json.dump({"n": n, "score": sc, "raw": {a: [v / n for v in acc[a]] for a in ARMS}},
              open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
