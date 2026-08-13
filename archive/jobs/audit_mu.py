#!/usr/bin/env python
"""AUDIT part 2: the operator's boost MULTIPLE mu is set by --k, and production sets it too low.

For a uniform k-member mean the code's V is
    V_code = mean_i E(L0_i - L0_mean) * k/(k-1) = (1 - 1/k) s^2 * k/(k-1) = s^2,
while the energy a single member has above the mean's is exactly V_target = (1 - 1/k) s^2.
So the operator always restores
    mu = V_code / V_target = k/(k-1)
times the single-member deficit -- 1.333 at the k=4 harness where lam was calibrated, but only
1.167 at r28's k=7 and 1.143 at r29's k=8.  (The weighted tower mean lands in the same place:
V_code/V_target = 1.173 / 1.150, measured algebraically.)  mu is NOT a free choice in the code --
it falls out of --k -- and the k=4 c-sweep says the score is still climbing at mu=1.6.

If a member were an unbiased sample around GT, mu*=1 would be right.  Members are all
systematically blurrier than GT, so with E_GT = E_mu + beta s^2 the energy-matching optimum is
    mu*(k) = (beta - 1/k) / (1 - 1/k),
which falls with k but far more slowly than k/(k-1).  This measures mu*(2) and mu*(4) directly,
fits beta, and predicts mu*(8) for the r29 towers.
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
GRID = {2: (1.15, 1.60, 2.00, 2.50), 4: (1.60, 2.00, 2.50)}   # k=4 low end comes from audit_er


def u8(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_t(a, dev):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev).float().div_(255.0) \
        .permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default=f"{HERE}/audit_mu.json")
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
           [f"k{k}_mu{m}" for k in (2, 4) for m in GRID[k]] + ["k4_mu1.333(SHIP)"]
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
            dev_u = sum(boxf(((m - L0) ** 2).sum(1, keepdim=True), 3) for m in mL0[:k]) / k
            pre[k] = (ens, L0, Eb, dev_u)          # V_code(mu) = mu * dev_u   (dev_u == V_target)
        for a in ARMS:
            k = int(a[1])
            ens, L0, Eb, dev_u = pre[k]
            if a.endswith("base"):
                x = ens
            else:
                mu = float(a.split("mu")[1].split("(")[0])
                r = torch.sqrt(1.0 + mu * dev_u / (Eb + 1e-10)).clamp(max=4.0)
                x = ens + (r - 1.0) * L0            # lam = 1.0 throughout
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
                f"k{k}[" + " ".join(f"{m}:{sc(f'k{k}_mu{m}')-sc(f'k{k}_base'):+.4f}"
                                    for m in GRID[k]) + "]" for k in (2, 4)) +
                f"  k4_SHIP1.333:{sc('k4_mu1.333(SHIP)')-sc('k4_base'):+.4f}", flush=True)

    n = len(stems)

    def sc(a):
        P, S, L = (v / n for v in acc[a])
        return 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
    print(f"\n=== BOOST MULTIPLE mu, HCM0181 real test GT, full shipped chain, n={n} ===")
    out = {}
    for k in (2, 4):
        ms = list(GRID[k]) + ([1.333] if k == 4 else [])
        ys = [sc(f"k{k}_mu{m}") - sc(f"k{k}_base") for m in GRID[k]]
        if k == 4:
            ys = ys + [sc("k4_mu1.333(SHIP)") - sc("k4_base")]
        order = np.argsort(ms)
        ms = list(np.array(ms)[order]); ys = list(np.array(ys)[order])
        c2, c1, c0 = np.polyfit(ms, ys, 2)
        v = -c1 / (2 * c2)
        out[k] = dict(mu=ms, gain=ys, vertex=float(v), shipped_mu=k / (k - 1.0))
        print(f"  k={k} (shipped mu={k/(k-1.0):.3f}, base={sc(f'k{k}_base'):.4f})  " +
              "  ".join(f"mu{m}:{y:+.4f}" for m, y in zip(ms, ys)) + f"   vertex mu*={v:.3f}")
    print("\n  measured mu*(2)=%.3f  mu*(4)=%.3f" % (out[2]["vertex"], out[4]["vertex"]))
    # mu*(k) = (beta - 1/k)/(1 - 1/k)  ->  beta from each k
    for k in (2, 4):
        b = out[k]["vertex"] * (1 - 1 / k) + 1 / k
        out[k]["beta"] = float(b)
        print(f"    implied beta from k={k}: {b:.3f}  -> predicted mu*(7)={(b-1/7)/(1-1/7):.3f}"
              f"  mu*(8)={(b-1/8)/(1-1/8):.3f}")
    json.dump({"n": n, "score": {a: sc(a) for a in ARMS},
               "raw": {a: [v / n for v in acc[a]] for a in ARMS}, "fit": out},
              open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
