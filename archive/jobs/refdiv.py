#!/usr/bin/env python
"""REFUTATION LENS for the "+0.2746 = value of the 2 different-family members" claim.

The claimant's control arm is cb_run.py's `famw_s0.00` on PoolA: it sets the weight of the
2 e-family members to zero IN THE MEAN, but leaves the energy-restore mis-specified --
restore still gets k=8 and ALL EIGHT member Laplacian bands, including the two members that
are no longer in the mean.  That is not "drop 2 members"; it is "6-member mean scored with
an 8-member restore".

More importantly it confounds DIVERSITY with DEPTH: going 8 -> 6 members loses variance
reduction no matter which 2 you drop.  Here every drop is done HONESTLY (restore k and the
member-band set both follow the surviving pool) and for THREE different pairs:

  drop_e2     : the 2 cross-family (e-cluster) members   <- the claimant's pair
  drop_ut2a/b : 2 SAME-family members (UT cluster, 4 reps)  <- the depth control
  drop_m31b2  : the m31b pair

If the three honest drops cost about the same, the -0.2746 is the price of DEPTH, not of
diversity, and the claim is refuted.

Everything else is byte-identical to cb_run.py: same members, same stems, same uint8 round,
same restore code, same field, same JPEG, same scorer.
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

# PoolA exactly as cb_run.py defines it: big 6 then small 2.
VAR = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
       "m31b_nolpips", "m31b_taillpips", "e15ceil95", "e17visnorm"]
IDX = {v: i for i, v in enumerate(VAR)}
K8 = len(VAR)


def mdir(v):
    return os.path.join(OUT, "HCM0181_" + v, "test_poses_renders_png")


def r8(t):
    return (t.clamp(0, 1) * 255.0 + 0.5).floor().clamp(0, 255) / 255.0


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


# ---- arm spec: (members-in-the-mean, members-whose-bands-feed-the-restore, k) --------------
def sub(names):
    return [IDX[n] for n in names]


UT4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
M2 = ["m31b_nolpips", "m31b_taillpips"]
E2 = ["e15ceil95", "e17visnorm"]
ALL = UT4 + M2 + E2

ARMS = {
    # CONTROL: the 8-member flat mean, exactly the claimant's baseline
    "mean8":            (ALL, ALL),
    # the claimant's own control arm, reproduced bit-for-bit (mis-specified restore: k=8,
    # all 8 bands, but only 6 members in the mean)
    "famw_s0.00_repro": (UT4 + M2, ALL),
    # --- HONEST drops: restore follows the surviving pool ---------------------------------
    "drop_e2":          (UT4 + M2, UT4 + M2),                       # the cross-family pair
    "drop_ut2a":        (["gsplatB9ut", "gsplatB10ut8M"] + M2 + E2,
                         ["gsplatB9ut", "gsplatB10ut8M"] + M2 + E2),  # SAME-family pair
    "drop_ut2b":        (["gsplatB11ut60k", "gsplatB12ut8Ms7"] + M2 + E2,
                         ["gsplatB11ut60k", "gsplatB12ut8Ms7"] + M2 + E2),
    "drop_m31b2":       (UT4 + E2, UT4 + E2),
    # a 6-member pool that keeps ONE member of each of the 3 families' partners removed --
    # mixed drop, one UT + one e: tests whether it is the e-family specifically
    "drop_ut1e1":       (["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k"] + M2
                         + ["e15ceil95"],
                        ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k"] + M2
                         + ["e15ceil95"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=1.30)
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default=f"{HERE}/refdiv.json")
    ap.add_argument("--dev", default="cuda:1")
    args = ap.parse_args()
    DEV = args.dev
    K = _K.to(DEV)
    want = [a for a in (args.arms.split(",") if args.arms else list(ARMS)) if a]
    assert "mean8" in want

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

    per = {a: [] for a in want}
    t0 = time.time()
    for c, s in enumerate(stems):
        X = torch.stack([torch.from_numpy(
            np.asarray(Image.open(os.path.join(mdir(v), s + ".png")).convert("RGB"),
                       np.float32) / 255.).permute(2, 0, 1) for v in VAR], 0).to(DEV)
        memL0 = [lap_pyr(X[i:i + 1], 5, K)[0][0] for i in range(K8)]      # HOISTED, shared
        gnp = np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"),
                         np.float32) / 255.
        g = torch.from_numpy(gnp).permute(2, 0, 1).unsqueeze(0).to(DEV)
        for a in want:
            mm, rm = ARMS[a]
            e = r8(X[sub(mm)].mean(0)).unsqueeze(0)
            bands = [memL0[i] for i in sub(rm)]
            o = restore(e, bands, args.lam, len(bands), K).clamp(0, 1)
            x = o[0].permute(1, 2, 0).cpu().numpy()
            x = np.clip(cv2.remap(np.ascontiguousarray(x), MX, MY, cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REFLECT), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(b, "JPEG", **JPG)
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
    print(f"\nDIVERSITY-vs-DEPTH, HCM0181, n={n}, FULL SHIPPED CHAIN "
          f"(lam={args.lam} gain={args.gain})")
    print(f"{'arm':>18} {'k':>2} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} "
          f"{'d(mean8)':>10} {'se':>7} {'t':>7} {'wins':>7}")
    rows = []
    for a in want:
        m = A[a].mean(0); d = A[a][:, 3] - base
        se = d.std(ddof=1) / np.sqrt(n) if a != "mean8" else 0.0
        t = d.mean() / se if se > 0 else 0.0
        rows.append((a, len(ARMS[a][1]), m[3], m[0], m[1], m[2], d.mean(), se, t,
                     int((d > 0).sum())))
    rows.sort(key=lambda r: -r[2])
    for a, k, sc, P, S, L, d, se, t, w in rows:
        print(f"{a:>18} {k:2d} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {d:+10.4f} {se:7.4f} "
              f"{t:7.2f} {w:3d}/{n}")
    json.dump(dict(n=n, per_view={a: A[a].tolist() for a in want}), open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
