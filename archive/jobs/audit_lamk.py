#!/usr/bin/env python
"""AUDIT part 2: does the OPTIMAL lambda drift with ensemble depth k?

lam=1.0 was chosen on a k=4 harness pool.  Production ships k=7 (r28) and k=8-9 (r29).
Algebra says the operator's V is ~1.000*s^2 at EVERY k (the k/(k-1) factor and the unweighted
deviation average cancel), while the physical deficit it should restore is s^2*(1-1/k_eff):
0.750*s^2 at k=4 but 0.870*s^2 at k=8.  So the operator's boost, measured as a multiple of the
deficit, FALLS from 1.333x at k=4 to 1.150x at k=8 -- production may be under-boosting relative
to the point where lam was calibrated.

This measures the lambda-vs-k drift directly: full lambda curve at k=2 and at k=4, on the
production harness through the full shipped chain.  If the vertex moves right as k grows, r29's
tower lambda should be above 1.0.
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
LAMS = (0.5, 1.0, 1.5)


def u8(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_t(a, dev):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev).float().div_(255.0) \
        .permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=f"{HERE}/audit_lamk.json")
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

    ARMS = [f"k{k}_base" for k in (2, 4)] + \
           [f"k{k}_lam{l}" for k in (2, 4) for l in LAMS]
    acc = {a: [0.0, 0.0, 0.0] for a in ARMS}
    t0 = time.time()
    for ci, s in enumerate(stems):
        mem = [to_t(u8(os.path.join(d, s + ".png")), dev) for d in MEM]
        mL0 = [lap_pyr(m, 5, K)[0][0] for m in mem]
        g = to_t(u8(os.path.join(GTD, gt_by[s])), dev)
        pre = {}
        for k in (2, 4):
            ens = torch.stack(mem[:k]).mean(0)
            L0 = lap_pyr(ens, 5, K)[0][0]
            Eb = boxf((L0 ** 2).sum(1, keepdim=True), 3)
            V = sum(boxf(((m - L0) ** 2).sum(1, keepdim=True), 3) for m in mL0[:k]) / k
            V = V * (k / (k - 1.0))
            r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=4.0)
            pre[k] = (ens, L0, r)
        for a in ARMS:
            k = int(a[1])
            ens, L0, r = pre[k]
            x = ens if a.endswith("base") else ens + float(a.split("lam")[1]) * (r - 1.0) * L0
            xn = np.clip(warp(np.ascontiguousarray(
                x.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()), lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((xn * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED_JPEG)
            y = to_t(np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                                dtype=np.uint8), dev)
            with torch.no_grad():
                acc[a][0] += 10 * np.log10(1.0 / max(float(((y - g) ** 2).mean()), 1e-12))
                acc[a][1] += float(repo_ssim(y, g))
                acc[a][2] += float(vgg(y * 2 - 1, g * 2 - 1).item())
        if (ci + 1) % 5 == 0:
            n = ci + 1

            def sc(a):
                return 100 * (0.4 * (1 - acc[a][2] / n) + 0.3 * acc[a][1] / n
                              + 0.3 * acc[a][0] / n / 50)
            print(f"[{n}/{len(stems)} {time.time()-t0:.0f}s] " + "  ".join(
                f"k{k}:" + "/".join(f"{sc(f'k{k}_lam{l}')-sc(f'k{k}_base'):+.4f}" for l in LAMS)
                for k in (2, 4)), flush=True)

    n = len(stems)

    def sc(a):
        P, S, L = (v / n for v in acc[a])
        return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    print(f"\n=== LAMBDA vs DEPTH, HCM0181 real test GT, full shipped chain, n={n} ===")
    res = {}
    for k in (2, 4):
        ys = [sc(f"k{k}_lam{l}") - sc(f"k{k}_base") for l in LAMS]
        c2, c1, c0 = np.polyfit(LAMS, ys, 2)
        vtx = -c1 / (2 * c2)
        res[k] = dict(gains=ys, vertex=float(vtx), peak=float(c0 + c1 * vtx + c2 * vtx ** 2))
        print(f"  k={k}  base={sc(f'k{k}_base'):.4f}  " +
              "  ".join(f"lam{l}:{y:+.4f}" for l, y in zip(LAMS, ys)) +
              f"   -> vertex lam*={vtx:.3f}  peak={res[k]['peak']:+.4f}")
    d = res[4]["vertex"] - res[2]["vertex"]
    print(f"\n  vertex drift per doubling of k: {d:+.3f}   "
          f"extrapolated lam* at k=8: {res[4]['vertex']+d:.3f}")
    print(f"  cost of shipping lam=1.0 at the k=8 extrapolated vertex: "
          f"see r29 build arg tower_lam")
    json.dump({"n": n, "arms": {a: [v / n for v in acc[a]] for a in ARMS},
               "score": {a: sc(a) for a in ARMS}, "fit": res}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
