#!/usr/bin/env python
"""THE r30 JOINT ARM TEST -- are the three stack-ons additive at PRODUCTION depth?

Three independently-measured candidates all touch the same chain and all move HF energy the
same direction, so they may substitute rather than add:
    (1) JPEG q98/ss0 instead of q100/ss2      (jpegalloc, +0.0249 measured at k=4)
    (2) field amplitude gain 1.30 -> 1.45     (residual, +0.0487 measured at k=4)
    (3) restore multiple mu 1.143 -> 1.37     (fault audit, +0.0092 measured at k=4)
Each was measured ALONE, on a k=4 pool, against a k=4 baseline whose mu is 1.333 -- but
production ships k=8 towers whose mu is 1.143. This script emulates production exactly:
8-member HCM0181 pool, restore(lam=1.0, mu=8/7), B11-class field x1.30, q100/ss2 = REFERENCE,
then each arm alone and all three together.
"""
import io, os, sys, time, json, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K                      # noqa: E402
from fieldlib import upsample, warp                                   # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(4)
Image.MAX_IMAGE_PIXELS = None
TAG = "HCM0181"
# production-depth pool: the 4 UT members + 4 different-family members (same 8 the adversarial
# verifier used).  Production towers are k=8, chair k=8, bonsai k=7.
POOL = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
        "m31b_nolpips", "m31b_taillpips", "e17visnorm", "e15ceil95"]
D = lambda m: f"/mnt/d/avv/output/{TAG}_{m}/test_poses_renders_png"

Q100 = dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q98 = dict(quality=98, subsampling=0, optimize=True, progressive=True)

#      name          mu       field gain   encode
SPEC = [("R29_ref",   8/7.0,   1.30, Q100),   # exactly what r29 ships on towers
        ("q98",       8/7.0,   1.30, Q98),
        ("f145",      8/7.0,   1.45, Q100),
        ("mu137",     1.37,    1.30, Q100),
        ("ALL_145",   1.37,    1.45, Q98),
        ("ALL_155",   1.37,    1.55, Q98),
        ("ALL_mu130", 1.30,    1.45, Q98),
        ("f145_q98",  8/7.0,   1.45, Q98)]
ARMS = [a for a, *_ in SPEC]
REF = "R29_ref"


def restore_mu(ens, members, lam, mu, win=3, nlev=5, clamp=4.0):
    """energy_restore.restore() with the k/(k-1) multiple replaced by an explicit mu."""
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


def ldt(p, dev):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)


def report(per, acc, nb, N, out):
    res = {}
    for a in ARMS:
        P, S, L = acc[a] / N
        res[a] = dict(score=100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)),
                      psnr=P, ssim=S, lpips=L, mb=nb[a] / 1e6)
    json.dump({"n": N, "res": res, "per": per}, open(out, "w"), indent=1)
    r0 = np.array(per[REF])
    print(f"\n[n={N}] r30 JOINT ARMS, {TAG}, 8-member production-depth pool, full shipped chain")
    print(f"{'arm':>10} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} "
          f"{'vs r29':>9} {'t':>7} {'win/N':>7} {'MB/60':>8}")
    for a in ARMS:
        v = np.array(per[a]) - r0
        t = (v.mean() / (v.std(ddof=1) / np.sqrt(N))) if a != REF and v.std() > 0 else float("nan")
        ts = f"{t:7.2f}" if t == t else f"{'--':>7}"
        r_ = res[a]
        print(f"{a:>10} {r_['score']:9.4f} {r_['psnr']:8.4f} {r_['ssim']:7.4f} {r_['lpips']:8.4f} "
              f"{r_['score']-res[REF]['score']:+9.4f} {ts} {int((v>0).sum()):3d}/{N} "
              f"{r_['mb']:8.2f}", flush=True)
    # additivity check
    s = {a: res[a]["score"] - res[REF]["score"] for a in ARMS}
    print(f"  ADDITIVITY: q98 {s['q98']:+.4f} + f145 {s['f145']:+.4f} + mu137 {s['mu137']:+.4f} "
          f"= {s['q98']+s['f145']+s['mu137']:+.4f}  vs JOINT ALL_145 {s['ALL_145']:+.4f}"
          f"  (ratio {(s['ALL_145']/(s['q98']+s['f145']+s['mu137'])) if abs(s['q98']+s['f145']+s['mu137'])>1e-9 else float('nan'):.2f})",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "final_joint.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()

    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    dirs = [D(m) for m in POOL]
    for d in dirs:
        assert os.path.isdir(d), d
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in dirs))
    if args.limit:
        stems = stems[:args.limit]
    print(f"pool k={len(dirs)}  n={len(stems)}", flush=True)

    H, W = 989, 1320
    z = np.load(os.path.join(HERE, "fieldrefit", f"flow_{TAG}_B11.npz"))
    b11 = np.median(z["s8"].astype(np.float32), 0)
    F = {}
    for a, mu, g, enc in SPEC:
        if g not in F:
            F[g] = upsample(b11 * g, H, W, "cubic")
    print("field mean|f| px: " + "  ".join(
        f"x{g} {np.linalg.norm(F[g],axis=2).mean():.4f}" for g in sorted(F)), flush=True)

    acc = {a: np.zeros(3) for a in ARMS}
    nb = {a: 0 for a in ARMS}
    per = {a: [] for a in ARMS}
    t0 = time.time()
    for n, s_ in enumerate(stems):
        mem = [ldt(os.path.join(d, s_ + ".png"), dev) for d in dirs]
        ens = torch.stack(mem).mean(0)
        # hoisted: one restore per distinct mu, shared by every arm using it
        ers = {}
        for mu in sorted({m for _, m, _, _ in SPEC}):
            with torch.no_grad():
                ers[mu] = restore_mu(ens, mem, 1.0, mu).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        del mem, ens
        g_ = torch.from_numpy(
            np.asarray(Image.open(os.path.join(gtd, gt_by[s_])).convert("RGB"),
                       dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)

        def chain(i):
            a, mu, gain, enc = SPEC[i]
            x = np.clip(warp(ers[mu], F[gain], "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **enc)
            by = b.getvalue()
            return len(by), np.asarray(Image.open(io.BytesIO(by)).convert("RGB"),
                                       dtype=np.float32) / 255.0

        with ThreadPoolExecutor(max_workers=8) as ex:
            jj = dict(zip(ARMS, ex.map(chain, range(len(SPEC)))))
        for a in ARMS:
            L_, j = jj[a]
            nb[a] += L_
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(((r - g_) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, g_))
                L = float(vgg(r * 2 - 1, g_ * 2 - 1).item())
            acc[a] += (P, S, L)
            per[a].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
            del r
        del g_, ers
        if n % 10 == 9 or n == len(stems) - 1:
            print(f"  {n+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
            report(per, acc, nb, n + 1, args.out)
    print(f"\nwrote {args.out}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
