#!/usr/bin/env python
"""PER-IMAGE PSF / DEBLUR AXIS.

Fits the MSE-optimal 7x7 linear correction kernel K (plus bias) mapping the PRODUCTION-CHAIN
render -> photo.  Fitting K directly is strictly stronger than "estimate PSF, then invert it":
whatever PSF you estimate, the Wiener/RL inverse restricted to a 7x7 support is one particular
7x7 filter, and K is the best one.  So K's score is an UPPER BOUND on the whole axis at that
support.  Sign is reported: K sharper-than-delta = deblur, blurrier = the render is too sharp.

LEAVE-ONE-OUT: A and b are additive over images, so the kernel applied to image i is fitted on
the other n-1 images only.  That removes self-fit bias and makes the number an honest estimate
of what a TRAIN-fitted kernel could do (a train fit is if anything worse: different poses).

SPATIAL VARIATION: same fit repeated on a 3x3 grid of tiles, applied with bilinear blending of
the 9 kernels -> tests the "defocus varies with field position" hypothesis.

INTERACTION WITH ENERGY RESTORATION: arms with lam=0 measure whether the kernel is just undoing
/ duplicating the shipped finest-band boost.
"""
import io, os, sys, time, json, argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as Fn
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K            # noqa: E402
from fieldlib import upsample, warp                          # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(24)
Image.MAX_IMAGE_PIXELS = None
Q100 = dict(quality=100, subsampling=2, optimize=True, progressive=True)
KS = 7
R = KS // 2
NT = 3  # tiles per axis


def restore_mu(ens, members, lam, mu, win=3, nlev=5, clamp=4.0):
    K = _K.to(ens.device)
    laps, res, sizes = lap_pyr(ens, nlev, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    V = 0.0
    for m in members:
        ml = lap_pyr(m, nlev, K)[0][0]
        V = V + boxf(((ml - L0) ** 2).sum(1, keepdim=True), win)
    V = V / len(members) * mu
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)
    out = [L0 * (1.0 + lam * (r - 1.0))] + laps[1:]
    return lap_recon(out, res, sizes, K)


def ld(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def accum(x, g):
    """x,g: HxWx3 float. Returns per-tile (A[50,50], b[50]) summed over the 3 channels."""
    H, W, _ = x.shape
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
    U = Fn.unfold(Fn.pad(t, (R, R, R, R), mode="reflect"), KS)      # [1, 3*49, H*W]
    U = U.view(3, KS * KS, H, W)
    G = torch.from_numpy(np.ascontiguousarray(g.transpose(2, 0, 1)))
    ys = np.linspace(0, H, NT + 1).astype(int)
    xs = np.linspace(0, W, NT + 1).astype(int)
    A = np.zeros((NT * NT, KS * KS + 1, KS * KS + 1), np.float64)
    B = np.zeros((NT * NT, KS * KS + 1), np.float64)
    for ti in range(NT):
        for tj in range(NT):
            k = ti * NT + tj
            Ut = U[:, :, ys[ti]:ys[ti + 1], xs[tj]:xs[tj + 1]].reshape(3, KS * KS, -1)
            Gt = G[:, ys[ti]:ys[ti + 1], xs[tj]:xs[tj + 1]].reshape(3, 1, -1)
            one = torch.ones_like(Gt)
            Uf = torch.cat([Ut, one], 1).permute(1, 0, 2).reshape(KS * KS + 1, -1)  # [50,3M]
            Gf = Gt.permute(1, 0, 2).reshape(1, -1)                                  # [1,3M]
            A[k] += (Uf @ Uf.T).double().numpy()
            B[k] += (Uf @ Gf.T).double().numpy()[:, 0]
    return A, B


def solve(A, B, ridge=1e-3):
    n = A.shape[-1]
    d = np.trace(A) / n
    return np.linalg.solve(A + ridge * d * np.eye(n), B)


def apply_k(x, W_):
    """W_: [NT*NT, 50] -> bilinear-blended spatially varying filter. x HxWx3."""
    H, Wd, _ = x.shape
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
    U = Fn.unfold(Fn.pad(t, (R, R, R, R), mode="reflect"), KS).view(3, KS * KS, H, Wd)
    cy = (np.linspace(0, H, NT + 1)[:-1] + np.linspace(0, H, NT + 1)[1:]) / 2
    cx = (np.linspace(0, Wd, NT + 1)[:-1] + np.linspace(0, Wd, NT + 1)[1:]) / 2
    yy = np.arange(H, dtype=np.float32)
    xx = np.arange(Wd, dtype=np.float32)
    wy = np.stack([np.interp(yy, cy, np.eye(NT)[i]) for i in range(NT)])   # [NT,H]
    wx = np.stack([np.interp(xx, cx, np.eye(NT)[j]) for j in range(NT)])   # [NT,W]
    Wfull = torch.zeros(KS * KS + 1, H, Wd)
    for i in range(NT):
        for j in range(NT):
            k = i * NT + j
            w = torch.from_numpy(np.outer(wy[i], wx[j]).astype(np.float32))
            Wfull += torch.from_numpy(W_[k].astype(np.float32))[:, None, None] * w
    out = (U * Wfull[None, :KS * KS]).sum(1) + Wfull[KS * KS][None]
    return out.permute(1, 2, 0).numpy()


def sig_of(k):
    """equivalent gaussian sigma of a 7x7 kernel (positive-part 2nd moment); <0 = sharpening."""
    K = k.reshape(KS, KS)
    y, x = np.mgrid[-R:R + 1, -R:R + 1]
    s = K.sum()
    m2 = (K * (x ** 2 + y ** 2)).sum() / max(s, 1e-9) / 2.0
    return np.sign(m2) * np.sqrt(abs(m2)), s, K[R, R]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="HCM0181")
    ap.add_argument("--pool", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--field", default="")
    ap.add_argument("--score", type=int, default=1)
    args = ap.parse_args()
    TAG = args.tag
    POOL = args.pool.split(",") if args.pool else [
        "gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
        "m31b_nolpips", "m31b_taillpips", "e17visnorm", "e15ceil95"]
    dirs = [f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png" for m in POOL]
    dirs = [d for d in dirs if os.path.isdir(d)]
    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
    if args.limit:
        stems = stems[:args.limit]
    print(f"{TAG}: pool k={len(dirs)}  n={len(stems)}", flush=True)

    H, W = ld(os.path.join(dirs[0], stems[0] + ".png")).shape[:2]
    fld = args.field or os.path.join(HERE, "fieldrefit", f"flow_{TAG}_B11.npz")
    Fm = None
    if os.path.exists(fld):
        z = np.load(fld)
        Fm = upsample(np.median(z["s8"].astype(np.float32), 0) * 1.30, H, W, "cubic")
        print(f"field mean|f| = {np.linalg.norm(Fm,axis=2).mean():.4f} px", flush=True)
    else:
        print("NO FIELD FILE -- chain runs without the lens field", flush=True)

    VARS = ["e10", "e00"]     # lam 1.0 (shipped) and lam 0 (no energy restoration)
    cache = {v: [] for v in VARS}
    Aa = {v: np.zeros((NT * NT, 50, 50)) for v in VARS}
    Bb = {v: np.zeros((NT * NT, 50)) for v in VARS}
    Ai = {v: [] for v in VARS}
    Bi = {v: [] for v in VARS}
    t0 = time.time()
    for n, s_ in enumerate(stems):
        mem = [torch.from_numpy(ld(os.path.join(d, s_ + ".png")).transpose(2, 0, 1))[None]
               for d in dirs]
        ens = torch.stack(mem).mean(0)
        g = ld(os.path.join(gtd, gt_by[s_]))
        for v in VARS:
            lam = 1.0 if v == "e10" else 0.0
            with torch.no_grad():
                y = restore_mu(ens, mem, lam, (len(dirs) / max(len(dirs) - 1.0, 1.0))).clamp(0, 1)
            y = y[0].permute(1, 2, 0).numpy()
            if Fm is not None:
                y = np.clip(warp(y, Fm, "lanczos"), 0, 1)
            cache[v].append(y.astype(np.float16))
            a, b = accum(y, g)
            Ai[v].append(a); Bi[v].append(b)
            Aa[v] += a; Bb[v] += b
        del mem, ens
        if n % 10 == 9:
            print(f"  fit {n+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    rep = {}
    for v in VARS:
        Wg = solve(Aa[v].sum(0), Bb[v].sum(0))
        s, tot, ctr = sig_of(Wg[:49])
        gs = [sig_of(solve(Aa[v][k], Bb[v][k])[:49]) for k in range(NT * NT)]
        rep[v] = dict(sigma_eq=float(s), ksum=float(tot), center=float(ctr), bias=float(Wg[49]),
                      tile_sigma=[float(a) for a, _, _ in gs],
                      tile_center=[float(c) for _, _, c in gs])
        print(f"\n[{v}] GLOBAL kernel: sigma_eq={s:+.4f}px  sum={tot:.5f}  center={ctr:.4f} "
              f"bias={Wg[49]:+.5f}", flush=True)
        print("     kernel(7x7) x1000:\n" +
              np.array2string((Wg[:49].reshape(7, 7) * 1000), precision=1, suppress_small=True))
        print(f"     TILE sigma_eq (3x3 field grid): "
              f"{np.array2string(np.array([a for a,_,_ in gs]).reshape(3,3), precision=4)}")
        # LOO in-sample MSE reduction (pixel domain, pre-JPEG), global kernel
        num = den = 0.0
        for i in range(N):
            w = solve(Aa[v].sum(0) - Ai[v][i].sum(0), Bb[v].sum(0) - Bi[v][i].sum(0))
            A_i, b_i = Ai[v][i].sum(0), Bi[v][i].sum(0)
            sse_k = w @ A_i @ w - 2 * w @ b_i
            d = np.zeros(50); d[24] = 1.0
            sse_0 = d @ A_i @ d - 2 * d @ b_i
            num += sse_0 - sse_k; den += sse_0
        rep[v]["loo_sse_red_frac_rel"] = float(num / abs(den)) if den else 0.0
        print(f"     LOO SSE change (rel to identity SSE-with-offset) = {num:.4f}", flush=True)

    if not args.score:
        json.dump(rep, open(os.path.join(HERE, f"psf_{TAG}.json"), "w"), indent=1)
        return

    # ---------------- scoring ----------------
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cpu"
    try:
        free, _ = torch.cuda.mem_get_info(0)
        if free > 4e9:
            dev = "cuda:0"
        else:
            free1, _ = torch.cuda.mem_get_info(1)
            if free1 > 4e9:
                dev = "cuda:1"
    except Exception:
        pass
    print(f"\nscoring on {dev}", flush=True)
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    ARMS = ["REF", "K7", "K7G", "NOREST", "NOREST_K7"]
    acc = {a: np.zeros(3) for a in ARMS}
    per = {a: [] for a in ARMS}
    for i, s_ in enumerate(stems):
        g = ld(os.path.join(gtd, gt_by[s_]))
        gt = torch.from_numpy(g.transpose(2, 0, 1))[None].to(dev)
        outs = {}
        for v, base, karm in (("e10", "REF", "K7"), ("e00", "NOREST", "NOREST_K7")):
            x = cache[v][i].astype(np.float32)
            outs[base] = x
            Ag = Aa[v].sum(0) - Ai[v][i].sum(0)
            Bg = Bb[v].sum(0) - Bi[v][i].sum(0)
            wg = solve(Ag, Bg)
            outs[karm] = np.clip(apply_k(x, np.tile(wg[None], (NT * NT, 1))), 0, 1)
            if v == "e10":
                Wt = np.stack([solve(Aa[v][k] - Ai[v][i][k], Bb[v][k] - Bi[v][i][k])
                               for k in range(NT * NT)])
                outs["K7G"] = np.clip(apply_k(x, Wt), 0, 1)
        for a in ARMS:
            b = io.BytesIO()
            Image.fromarray((np.clip(outs[a], 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **Q100)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j.transpose(2, 0, 1)))[None].to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(((r - gt) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, gt))
                L = float(vgg(r * 2 - 1, gt * 2 - 1).item())
            acc[a] += (P, S, L)
            per[a].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
        if i % 5 == 4 or i == N - 1:
            n_ = i + 1
            print(f"\n[{TAG} n={n_}] {time.time()-t0:.0f}s", flush=True)
            r0 = np.array(per["REF"]); r00 = np.array(per["NOREST"])
            for a in ARMS:
                P, S, L = acc[a] / n_
                sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
                base = r00 if a.startswith("NOREST") else r0
                d = np.array(per[a]) - base
                t = d.mean() / (d.std(ddof=1) / np.sqrt(n_)) if n_ > 2 and d.std() > 0 else float("nan")
                print(f"  {a:>10} {sc:9.4f} P{P:8.4f} S{S:7.4f} L{L:8.5f} "
                      f"d{d.mean():+8.4f} t{t:7.2f} w{int((d>0).sum()):3d}/{n_}", flush=True)
            rep["score"] = {a: dict(score=100 * (0.4 * (1 - acc[a][2] / n_) + 0.3 * acc[a][1] / n_
                                                 + 0.3 * acc[a][0] / n_ / 50),
                                    psnr=acc[a][0] / n_, ssim=acc[a][1] / n_, lpips=acc[a][2] / n_)
                            for a in ARMS}
            rep["n"] = n_
            rep["per"] = {a: per[a] for a in ARMS}
            json.dump(rep, open(os.path.join(HERE, f"psf_{TAG}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
