#!/usr/bin/env python
"""ADVERSARIAL RE-MEASUREMENT of "r29 --weights 4 1 1 costs -0.0139 vs flat at equal quality".

Three things the original never did:
  1. n=60 (it used 50) on the SAME PoolA, to see if -0.0139 survives more samples.
  2. The LITERAL production edit: 2-stage combine at 4:1:1 vs 2-stage at 6:1:1.  The original
     compared the 2-stage 4:1:1 against the IDEAL one-stage flat mean, which is NOT reachable
     by editing that token, and then subtracted an assumed rounding tax.  Both prod arms here
     carry the identical double-rounding, so their difference IS the token.
  3. A genuinely EQUAL-QUALITY family split.  PoolA's small family (e15/e17) is 0.214 dB WORSE
     than its big six (oracle per-member error PSNR vs real test GT), so PoolA measures the
     "small family is worse" case, not the equal case the claim generalises to.  Splits with
     |quality gap| ~ 0.01-0.11 dB are run alongside it.

One member load per view; all arms share the stack, the member Laplacians, the field and the
encoder.  Full shipped chain: combine -> uint8 -> energy restore(lam=1,k=8) -> median lens
field x1.30 INTER_LANCZOS4 -> JPEG q100/ss2 -> decode -> score.
"""
import os, sys, io, time, json, argparse
import numpy as np, torch, cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, upsample
from cb_run import mdir, GTD, r8, restore

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(2); torch.set_num_threads(2)
JPG = dict(quality=100, subsampling=2, optimize=True, progressive=True)

# canonical order == cb_cov_A.json order
VAR = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
       "m31b_nolpips", "m31b_taillpips", "e15ceil95", "e17visnorm"]
K8 = 8


def famw_idx(X, small, ws):
    """total weight ws on the members listed in `small`, 1-ws spread over the rest."""
    k = X.shape[0]
    big = [i for i in range(k) if i not in small]
    w = torch.zeros(k, 1, 1, 1, device=X.device)
    w[big] = (1.0 - ws) / len(big)
    w[small] = ws / len(small)
    return (X * w).sum(0)


def prod2s(X, small, wbig):
    """literal ensemble_renders.py --dirs <already-uint8 6-member mean> <m1> <m2>
       --weights wbig 1 1 .  The big-family mean is rounded to uint8 FIRST."""
    big = [i for i in range(X.shape[0]) if i not in small]
    tot = wbig + len(small)
    return r8(X[big].mean(0)) * (wbig / tot) + X[small].sum(0) / tot


SMALL_E = [6, 7]        # e15ceil95, e17visnorm   -> 0.214 dB WORSE than big six  (PoolA)
SMALL_M = [4, 5]        # m31b pair               -> 0.113 dB better
SMALL_U = [0, 1]        # gsplatB9ut, gsplatB10ut8M -> 0.011 dB better  == EQUAL QUALITY

ARMS = {
    "mean8_CONTROL":  lambda X: X.mean(0),
    "null_ws0.25":    lambda X: famw_idx(X, SMALL_E, 0.25),          # identical to control
    "famwE_ws0.333":  lambda X: famw_idx(X, SMALL_E, 1.0 / 3),       # the claim's arm
    "famwM_ws0.333":  lambda X: famw_idx(X, SMALL_M, 1.0 / 3),
    "famwU_ws0.333":  lambda X: famw_idx(X, SMALL_U, 1.0 / 3),       # equal-quality split
    "prod411_E":      lambda X: prod2s(X, SMALL_E, 4),               # literal r29
    "prod611_E":      lambda X: prod2s(X, SMALL_E, 6),               # the one-token "fix"
    "prod411_U":      lambda X: prod2s(X, SMALL_U, 4),
    "prod611_U":      lambda X: prod2s(X, SMALL_U, 6),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=1.30)
    ap.add_argument("--dev", default="cuda:1")
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default=f"{HERE}/refw_411.json")
    args = ap.parse_args()
    DEV = args.dev
    K = _K.to(DEV)
    want = [a for a in (args.arms.split(",") if args.arms else list(ARMS)) if a]
    assert "mean8_CONTROL" in want

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(mdir(v), s + ".png")) for v in VAR))[:args.n]

    z = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
    H, W = [int(x) for x in z["HW"]]
    lens = upsample(LooPool(z["s8"]).pooled("median"), H, W, "cubic") * args.gain
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    MX = (xx + lens[..., 0]).astype(np.float32); MY = (yy + lens[..., 1]).astype(np.float32)

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(DEV).eval()

    per = {a: [] for a in want}
    t0 = time.time()
    for c, s in enumerate(stems):
        X = torch.stack([torch.from_numpy(
            np.asarray(Image.open(os.path.join(mdir(v), s + ".png")).convert("RGB"),
                       np.float32) / 255.).permute(2, 0, 1) for v in VAR], 0).to(DEV)
        memL0 = [lap_pyr(X[i:i + 1], 5, K)[0][0] for i in range(K8)]      # HOISTED
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"),
                                        np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(DEV)
        for a in want:
            e = r8(ARMS[a](X)).unsqueeze(0)
            o = restore(e, memL0, args.lam, K8, K).clamp(0, 1)
            x = o[0].permute(1, 2, 0).cpu().numpy()
            x = np.clip(cv2.remap(np.ascontiguousarray(x), MX, MY, cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REFLECT), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(b, "JPEG", **JPG)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float32) / 255.
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(DEV)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(float(((r - g) ** 2).mean()), 1e-12))
                S = float(repo_ssim(r, g)); L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            per[a].append((P, S, L, 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.))))
        del X, memL0, g
        if c % 10 == 0:
            print(f"  {c+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    n = len(stems)
    A = {a: np.array(per[a]) for a in want}
    base = A["mean8_CONTROL"][:, 3]
    print(f"\nn={n} lam={args.lam} gain={args.gain}  FULL SHIPPED CHAIN, control = flat mean8")
    print(f"{'arm':>16} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'d':>9} {'se':>7} "
          f"{'t':>6} {'wins':>7}")
    for a in want:
        m = A[a].mean(0); d = A[a][:, 3] - base
        se = d.std(ddof=1) / np.sqrt(n) if a != "mean8_CONTROL" else 0.0
        t = d.mean() / se if se > 0 else 0.0
        print(f"{a:>16} {m[3]:9.4f} {m[0]:8.4f} {m[1]:8.5f} {m[2]:8.5f} {d.mean():+9.4f} "
              f"{se:7.4f} {t:6.2f} {int((d>0).sum()):3d}/{n}")

    def pair(x, y):
        d = A[x][:, 3] - A[y][:, 3]
        se = d.std(ddof=1) / np.sqrt(n)
        return d.mean(), se, d.mean() / se if se > 0 else 0.0, int((d > 0).sum())
    print("\nDECISION-RELEVANT PAIRED CONTRASTS (the literal one-token edit):")
    for x, y in [("prod611_E", "prod411_E"), ("prod611_U", "prod411_U"),
                 ("famwE_ws0.333", "mean8_CONTROL"), ("famwU_ws0.333", "mean8_CONTROL"),
                 ("famwM_ws0.333", "mean8_CONTROL"), ("null_ws0.25", "mean8_CONTROL")]:
        if x in A and y in A:
            d, se, t, w = pair(x, y)
            print(f"  {x:>14} - {y:<14} {d:+9.4f} +/- {se:.4f}  t={t:6.2f}  wins {w}/{n}")

    json.dump(dict(n=n, vars=VAR, lam=args.lam, gain=args.gain,
                   per_view={a: A[a].tolist() for a in want}), open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
