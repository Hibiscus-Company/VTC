#!/usr/bin/env python
"""FAMILY-WEIGHT SWEEP THROUGH THE FULL SHIPPED CHAIN.

r29 ships the towers as  0.667*[r22 6-member mean] + 0.167*mip555 + 0.167*mip777, i.e. the
2-member minority family carries 0.333 while its uniform-per-member share is 0.250. That 0.333
comes from ONE 2-member blend sweep, run on the data-starved eval-split proxy, on RAW pixels
(no energy restore, no field, no JPEG), then extrapolated by an "effective member count" argument.

Here it is re-derived at production structure -- 6 majority + 2 minority members, restore k=8 --
on the production harness, through the FULL shipped chain:
    weighted mean -> energy restore(lam=1.0, k) -> median lens field x1.30, INTER_LANCZOS4
    -> JPEG q100 subsampling=2 optimize progressive -> decode -> score
CONTROLS: wB=0 (drop the minority family entirely) and the honest 2-fold-CV LS-optimal per-member
weights, which is the "least-squares beats uniform once families differ in quality" hypothesis.

PERF: the energy-restore disagreement map needs each MEMBER's finest Laplacian band, which does
not depend on the arm. The stock restore() recomputes all 8 member pyramids inside every arm
(81 pyramids/stem, 76 s/stem). Here they are computed ONCE per stem on the GPU and reused; the
identity against the production operator is asserted at startup.
"""
import io, os, sys, time, json
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from energy_restore import restore
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
FIELD_GAIN = 1.30

A6 = os.environ.get("WT_A", "gsplatB9ut,gsplatB10ut8M,gsplatB11ut60k,gsplatB12ut8Ms7,gsplatB1,gsplatB2").split(",")
B2 = os.environ.get("WT_B", "m31b_nolpips,m31b_taillpips").split(",")
MEM = A6 + B2
NA, NK = len(A6), len(A6) + len(B2)
WGRID = [float(x) for x in os.environ.get(
    "WT_GRID", "0.125,0.20,0.25,0.291,0.3333,0.40,0.50").split(",")]
UNI = len(B2) / len(MEM)
UKEY = f"wB{UNI:.4f}"

DIRS = {}
for m in MEM:
    for sub in ("test_poses_renders_png", "tp_png"):
        p = f"/mnt/d/avv/output/{TAG}_{m}/{sub}"
        if os.path.isdir(p):
            DIRS[m] = p
            break


def ld(p, dev):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)


def restore_fast(ens, mem_L0, lam, k, win=3, nlev=5, clamp=4.0):
    """Identical to energy_restore.restore but takes PRE-COMPUTED member finest bands."""
    K = _K.to(ens.device)
    laps, res, sizes = lap_pyr(ens, nlev, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    V = 0.0
    for ml in mem_L0:
        V = V + boxf(((ml - L0) ** 2).sum(1, keepdim=True), win)
    V = V / len(mem_L0) * (k / (k - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)
    return lap_recon([L0 * (1.0 + lam * (r - 1.0))] + laps[1:], res, sizes, K)


def ls_weights(Cm):
    v = np.linalg.solve(Cm, np.ones(Cm.shape[0]))
    return v / v.sum()


def main():
    nlim = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(DIRS[m], s + ".png")) for m in MEM))

    # ---- assert the fast operator is the production operator ------------------------------
    tst = [ld(os.path.join(DIRS[m], stems[0] + ".png"), dev) for m in MEM]
    e = torch.stack(tst).mean(0)
    L0s = [lap_pyr(m, 5, _K.to(dev))[0][0] for m in tst]
    a, b = restore(e, tst, 1.0, NK), restore_fast(e, L0s, 1.0, NK)
    d = float((a - b).abs().max())
    print(f"restore_fast vs production restore: max|diff| = {d:.3e}")
    assert d < 1e-5, "fast restore diverges from the production operator"
    del tst, e, L0s, a, b

    # ---- honest 2-fold-CV LS weights, fitted on the OTHER fold's residual Gram -------------
    Z = np.load(f"{HERE}/wt_gram.npz")
    gnames = list(Z["names"]); gstems = list(Z["stems"])
    ii = [gnames.index(m) for m in MEM]
    si = {s: k for k, s in enumerate(gstems)}
    fold = {s: (k % 2) for k, s in enumerate(stems)}          # deterministic alternating split
    Gs = Z["G"][:, ii][:, :, ii]
    lsw = {}
    for f in (0, 1):
        tr = [si[s] for s in stems if fold[s] != f]           # fit on the complementary fold
        lsw[f] = ls_weights(Gs[tr].mean(0))
    print("2-fold-CV LS weights (each applied only to the held-out fold):")
    for f in (0, 1):
        print(f"  fold{f}: " + " ".join(f"{m}={w:.3f}" for m, w in zip(MEM, lsw[f])))
        print(f"          minority family share = {lsw[f][NA:].sum():.4f} (uniform 0.250)")

    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in cache["HW"]]
    lens = upsample(LooPool(cache["s8"]).pooled("median"), H, W, "cubic") * FIELD_GAIN

    arms = ["drop_B_k6"] + [f"wB{w:.4f}" for w in WGRID] + ["LS_2foldCV"]
    acc = {a: [0.0, 0.0, 0.0] for a in arms}
    per = {a: [] for a in arms}
    nb = {a: 0 for a in arms}
    stems = stems[:nlim]
    t0 = time.time()
    for n, s in enumerate(stems):
        x = [ld(os.path.join(DIRS[m], s + ".png"), dev) for m in MEM]
        L0 = [lap_pyr(m, 5, _K.to(dev))[0][0] for m in x]      # HOISTED: 8 pyramids, not 81
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                                        dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        mA = torch.stack(x[:NA]).mean(0)
        mB = torch.stack(x[NA:]).mean(0)
        plan = {"drop_B_k6": (mA, L0[:NA], NA)}
        for w in WGRID:
            plan[f"wB{w:.4f}"] = ((1 - w) * mA + w * mB, L0, NK)
        wv = lsw[fold[s]]
        plan["LS_2foldCV"] = (sum(float(wv[i]) * x[i] for i in range(NK)), L0, NK)
        for a in arms:
            ens, ml, k = plan[a]
            o = restore_fast(ens, ml, 1.0, k).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            y = np.clip(warp(o, lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((y * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            nb[a] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, g))
                L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            acc[a][0] += P; acc[a][1] += S; acc[a][2] += L
            per[a].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
        if n % 5 == 0:
            print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    print(f"\nFAMILY WEIGHT, {TAG}, n={N}, FULL SHIPPED CHAIN "
          f"({NA}+{len(B2)} members, restore lam=1.0 k={NK}, field x{FIELD_GAIN} lanczos4, JPEG q100/ss2)")
    print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'vs uniform':>11} "
          f"{'t(paired)':>10} {'MB/60':>7}")
    out = {}
    for a in arms:
        P, S, L = (v / N for v in acc[a])
        out[a] = (100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)), P, S, L)
    ref = np.array(per[UKEY])
    base = out[UKEY][0]
    for a in arms:
        sc, P, S, L = out[a]
        d = np.array(per[a]) - ref
        t = d.mean() / (d.std(ddof=1) / np.sqrt(N)) if d.std() > 0 else 0.0
        print(f"{a:>12} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {sc-base:+11.4f} {t:10.2f} "
              f"{nb[a]/1e6:7.2f}")
    json.dump({"B": B2, "score": {a: out[a] for a in arms}, "per": {a: per[a] for a in arms}},
              open(f"{HERE}/wt_chain_{os.environ.get('WT_ID','1')}.json", "w"), indent=1)
    grid = np.array(WGRID)
    sc = np.array([out[f"wB{w:.4f}"][0] for w in WGRID])
    # parabola through the 3 best grid points -> continuous argmax
    o = sc.argmax()
    if 0 < o < len(grid) - 1:
        x0, x1, x2 = grid[o-1:o+2]; y0, y1, y2 = sc[o-1:o+2]
        den = (y0 - 2*y1 + y2)
        xm = x1 - 0.5 * (grid[o+1]-grid[o]) * (y2 - y0) / den if den != 0 else grid[o]
    else:
        xm = grid[o]
    print(f"\nargmax on the sweep grid: wB = {grid[o]:.4f}  (parabolic {xm:.4f}); "
          f"uniform-per-member = {UNI:.4f}")
    print(f"LS 2-fold CV -> {out['LS_2foldCV'][0]-base:+.4f} vs uniform {UNI:.3f}")
    print(f"drop minority family -> {out['drop_B_k6'][0]-base:+.4f} vs uniform {UNI:.3f}")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
