#!/usr/bin/env python
"""COMBINER LENS: does anything beat the weighted pixel mean at PRODUCTION k with MIXED
families, measured through the FULL SHIPPED CHAIN?

chain per arm:  combine(8 members) -> uint8 round (png_ens) -> energy restore(lam, k=8)
                -> median lens field * gain, INTER_LANCZOS4 -> JPEG q100/ss2 -> score

CONTROL = flat pixel mean over all 8 members (arm "mean8").  Every arm sees the SAME
member stack, the SAME restore code, the SAME field and the SAME encoder; only the
combine step differs.  Shared work (member Laplacians, the field maps, the GT tensor)
is hoisted out of the arm loop.
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

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(2); torch.set_num_threads(2)
JPG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
OUT = "/mnt/d/avv/output"
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"

# ---- production-shaped MIXED-FAMILY pools (families from cb_struct.py RMS clustering) ----
# PoolA: big family of 6 (UT + m31b cluster, intra-RMS 5.8-9.3) + small family of 2
#        (e-cluster, intra 7.6) at inter-RMS 10.7-10.9.  Mirrors r29 towers: 6 UT-ish
#        members + 2 mip3d members.
POOLS = {
    "A": (["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
           "m31b_nolpips", "m31b_taillpips"], ["e15ceil95", "e17visnorm"]),
    "A6": (["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"],
           ["e15ceil95", "e17visnorm"]),
    "B": (["gsplatB1", "gsplatB2", "gsplatB3", "gsplatB5affine", "gsplatB7ppisp2",
           "gsplatB4warm"], ["m31b_nolpips", "m31b_taillpips"]),
}


def mdir(v):
    return os.path.join(OUT, "HCM0181_" + v, "test_poses_renders_png")


def r8(t):
    return (t.clamp(0, 1) * 255.0 + 0.5).floor().clamp(0, 255) / 255.0


# ------------------------------------------------------------------ combiners
def c_famw(X, nbig, ws):
    """weighted mean: total weight ws on the SMALL family, 1-ws spread over the big one."""
    k = X.shape[0]; nsm = k - nbig
    w = torch.empty(k, 1, 1, 1, device=X.device)
    w[:nbig] = (1.0 - ws) / nbig
    w[nbig:] = ws / nsm
    return (X * w).sum(0)


def c_winsor(X, t=1):
    S, _ = torch.sort(X, dim=0)
    k = X.shape[0]
    S = torch.cat([S[t:t + 1].expand(t, -1, -1, -1), S[t:k - t],
                   S[k - t - 1:k - t].expand(t, -1, -1, -1)], 0)
    return S.mean(0)


def c_trim(X, t=1):
    S, _ = torch.sort(X, dim=0)
    return S[t:X.shape[0] - t].mean(0)


def _sqdev(X, ref, r):
    """per-member LOCAL squared deviation from a consensus, RGB-summed, box-smoothed."""
    d = ((X - ref.unsqueeze(0)) ** 2).sum(1, keepdim=True)       # (k,1,H,W)
    return boxf(d, 2 * r + 1)


def c_invvar(X, r=8, eps=1e-4):
    med = X.median(0).values
    W = 1.0 / (_sqdev(X, med, r) + eps)
    return (X * W).sum(0) / W.sum(0)


def c_agree(X, r=8, b=1.0):
    """soft per-pixel confidence: w_i = exp(-0.5 (d_i / (b*s))^2), s = local consensus scale."""
    med = X.median(0).values
    D = torch.sqrt(_sqdev(X, med, r) + 1e-12)
    s = D.mean(0, keepdim=True).clamp_min(1.0 / 255.0)
    W = torch.exp(-0.5 * (D / (b * s)) ** 2)
    return (X * W).sum(0) / W.sum(0)


def c_hedge(X, r=8, a=1.0):
    """the OPPOSITE of invvar: UP-weight members that deviate locally, i.e. keep more of
    the disagreement energy in the mean itself instead of leaving it for the restore."""
    med = X.median(0).values
    D = _sqdev(X, med, r)
    W = 1.0 + a * D / D.mean(0, keepdim=True).clamp_min(1e-8)
    return (X * W).sum(0) / W.sum(0)


def c_hybinv(X, r=8):
    """ROBUST LOW frequencies + MEAN high frequencies = mean + LP_r(median - mean).
    This is the mirror image of agg_hybrid_freq (LP=mean, HP=median), which was measured
    and lost.  Motivated by the error geography: flat regions hold 39% of the LPIPS, and a
    robust low-frequency estimate is the only combiner-side handle on them, while the mean
    keeps maximal variance reduction in the high band where the restore operator lives."""
    m = X.mean(0); d = (X.median(0).values - m).unsqueeze(0)
    return m + boxf(d, 2 * r + 1)[0]


def c_moment(X):
    """per-member per-channel gain+bias matched to the pool mean before averaging."""
    m = X.mean(0)
    mu_r = m.mean((1, 2), keepdim=True); sd_r = m.std((1, 2), keepdim=True)
    mu = X.mean((2, 3), keepdim=True); sd = X.std((2, 3), keepdim=True).clamp_min(1e-6)
    return ((X - mu) * (sd_r / sd) + mu_r).mean(0)


def c_famhier_robust(X, nbig):
    """within the big family use the plain mean (variance reduction), ACROSS families use
    the equal-family mean.  identical to famw(0.5) -- kept as an explicit alias."""
    return 0.5 * X[:nbig].mean(0) + 0.5 * X[nbig:].mean(0)


def restore(ens, memL0, lam, k, K, nlev=5, win=3):
    laps, res, sizes = lap_pyr(ens, nlev, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    V = 0.0
    for ml in memL0:
        V = V + boxf(((ml - L0) ** 2).sum(1, keepdim=True), win)
    V = V / len(memL0) * (k / (k - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=4.0)
    return lap_recon([L0 * (1.0 + lam * (r - 1.0))] + laps[1:], res, sizes, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="A")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=1.30)
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dev", default="cuda:0")
    args = ap.parse_args()
    DEV = args.dev
    big, small = POOLS[args.pool]
    VAR = big + small
    NB = len(big); K8 = len(VAR)
    K = _K.to(DEV)

    ARMS = {
        "mean8":        lambda X: X.mean(0),                       # CONTROL (= famw 0.25)
        "famw_s0.20":   lambda X: c_famw(X, NB, 0.20),
        "famw_s0.333":  lambda X: c_famw(X, NB, 1.0 / 3),          # r29 production shape
        "famw_s0.50":   lambda X: c_famw(X, NB, 0.50),             # equal-family hierarchy
        "winsor1":      lambda X: c_winsor(X, 1),
        "invvar_r8":    lambda X: c_invvar(X, 8),
        "agree_b0.5":   lambda X: c_agree(X, 8, 0.5),
        "agree_b1":     lambda X: c_agree(X, 8, 1.0),
        "hedge_a1":     lambda X: c_hedge(X, 8, 1.0),
        "moment":       c_moment,
        "trim1":        lambda X: c_trim(X, 1),
        "winsor2":      lambda X: c_winsor(X, 2),
        "trim2":        lambda X: c_trim(X, 2),
        # --- the literal r29 tower combine and the SHIPPABLE standalone winsor patch ---
        # prod2s: ensemble_renders.py --dirs <r22 6-member png_ens> <mip3d> <new> --weights 4 1 1
        #         i.e. the 6-member mean is rounded to uint8 FIRST, then re-averaged.
        # *_wx:   the patch, applied as a separate pass over the written png_ens -> one
        #         EXTRA uint8 round, which is what a standalone script actually costs.
        "prod2s":       lambda X: r8(X[:6].mean(0)) * (4. / 6) + (X[6] + X[7]) / 6,
        "prod2s_wx":    lambda X: r8(r8(X[:6].mean(0)) * (4. / 6) + (X[6] + X[7]) / 6)
                                  + 0.5 * (c_winsor(X, 1) - X.mean(0)),
        "wx05_patch":   lambda X: r8(X.mean(0)) + 0.5 * (c_winsor(X, 1) - X.mean(0)),
        "winsorx0.25":  lambda X: X.mean(0) + 0.25 * (c_winsor(X, 1) - X.mean(0)),
        "winsorx0.7":   lambda X: X.mean(0) + 0.70 * (c_winsor(X, 1) - X.mean(0)),
        "winsorx0.85":  lambda X: X.mean(0) + 0.85 * (c_winsor(X, 1) - X.mean(0)),
        "winsorx1.5":   lambda X: X.mean(0) + 1.5 * (c_winsor(X, 1) - X.mean(0)),
        "winsorx0.5":   lambda X: X.mean(0) + 0.5 * (c_winsor(X, 1) - X.mean(0)),
        "invvar_r32":   lambda X: c_invvar(X, 32),
        "agree_b2":     lambda X: c_agree(X, 8, 2.0),
        "famw_s0.40":   lambda X: c_famw(X, NB, 0.40),
        "famw_s0.15":   lambda X: c_famw(X, NB, 0.15),
        "famw_s0.00":   lambda X: c_famw(X, NB, 0.00),
        "hybinv_r8":    lambda X: c_hybinv(X, 8),
        "hybinv_r32":   lambda X: c_hybinv(X, 32),
    }
    want = [a for a in (args.arms.split(",") if args.arms else list(ARMS)) if a]
    assert "mean8" in want, "control must be included"
    bad = [a for a in want if a not in ARMS]
    assert not bad, bad

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(mdir(v), s + ".png")) for v in VAR))
    stems = stems[:args.n]

    z = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
    H, W = [int(x) for x in z["HW"]]
    lens = upsample(LooPool(z["s8"]).pooled("median"), H, W, "cubic") * args.gain
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    MX = (xx + lens[..., 0]).astype(np.float32); MY = (yy + lens[..., 1]).astype(np.float32)

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(DEV).eval()

    per = {a: [] for a in want}; nby = {a: 0 for a in want}
    t0 = time.time()
    for c, s in enumerate(stems):
        X = torch.stack([torch.from_numpy(
            np.asarray(Image.open(os.path.join(mdir(v), s + ".png")).convert("RGB"),
                       np.float32) / 255.).permute(2, 0, 1) for v in VAR], 0).to(DEV)
        memL0 = [lap_pyr(X[i:i + 1], 5, K)[0][0] for i in range(K8)]          # HOISTED
        gnp = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"),
                         np.float32) / 255.
        g = torch.from_numpy(gnp).permute(2, 0, 1).unsqueeze(0).to(DEV)
        for a in want:
            e = r8(ARMS[a](X)).unsqueeze(0)
            o = restore(e, memL0, args.lam, K8, K).clamp(0, 1)
            x = o[0].permute(1, 2, 0).cpu().numpy()
            x = np.clip(cv2.remap(np.ascontiguousarray(x), MX, MY, cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REFLECT), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(b, "JPEG", **JPG)
            nby[a] += b.tell()
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           np.float32) / 255.
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(DEV)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(float(((r - g) ** 2).mean()), 1e-12))
                S = float(repo_ssim(r, g)); L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            per[a].append((P, S, L, 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50., 1.))))
        del X, memL0, g
        if c % 5 == 0:
            print(f"  {c+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    n = len(stems)
    A = {a: np.array(per[a]) for a in want}
    base = A["mean8"][:, 3]
    print(f"\nPOOL {args.pool}  big={NB} small={K8-NB}  k={K8}  n={n}  lam={args.lam} "
          f"gain={args.gain}\nFULL SHIPPED CHAIN: combine -> uint8 -> restore(k={K8}) -> "
          f"field(lanczos4) -> JPEG q100/ss2")
    print(f"{'arm':>14} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'d(mean8)':>10} "
          f"{'se':>7} {'t':>6} {'wins':>7} {'MB':>7}")
    rows = []
    for a in want:
        m = A[a].mean(0); d = A[a][:, 3] - base
        se = d.std(ddof=1) / np.sqrt(n) if a != "mean8" else 0.0
        t = d.mean() / se if se > 0 else 0.0
        rows.append((a, m[3], m[0], m[1], m[2], d.mean(), se, t, int((d > 0).sum()), nby[a] / 1e6))
    rows.sort(key=lambda r: -r[1])
    for a, sc, P, S, L, d, se, t, w, mb in rows:
        print(f"{a:>14} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {d:+10.4f} {se:7.4f} "
              f"{t:6.2f} {w:3d}/{n} {mb:7.2f}")
    if args.out:
        json.dump(dict(pool=args.pool, vars=VAR, nbig=NB, n=n, lam=args.lam, gain=args.gain,
                       per_view={a: A[a].tolist() for a in want},
                       mb={a: nby[a] / 1e6 for a in want}), open(args.out, "w"))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
