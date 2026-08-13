#!/usr/bin/env python
"""INDEPENDENT RE-MEASUREMENT of the "0.250 is a -0.0413 regression" claim.

Same harness (HCM0181, real test GT, n=60), same full shipped chain:
    weighted family mean -> energy restore(lam, k) -> median lens field x1.30 lanczos4
    -> JPEG q100/ss2/optimize/progressive -> decode -> score.

THREE configurations of the SAME 6+2 structure, plus the family-mean quality of each family
measured through the same chain (lam=0 so no k-dependence), plus a lam=0 replicate of the
weight contrast in cfg4 and cfgP to test whether the weight preference is really a weight
effect or an artifact of the restore amplitude (the restore's disagreement map is computed
from UNWEIGHTED member deviations no matter what the blend weight is, so raising wB raises r).

Shared computation (member loads, member finest Laplacian bands, GT load, lens field) is
hoisted out of the arm loop.
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

MEM = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
       "gsplatB1", "gsplatB2", "fid_ctrl", "m31b_taillpips", "m31b_nolpips", "fid_l2reg"]
IDX = {m: i for i, m in enumerate(MEM)}

# the three 6+2 configurations
A_CLAIM = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
           "gsplatB1", "gsplatB2"]                     # cfg1/cfg4 majority (2 of 6 are AA-family)
CFG = {
    "c4": (A_CLAIM, ["fid_ctrl", "m31b_taillpips"]),   # the claim's "production analogue"
    "c1": (A_CLAIM, ["m31b_nolpips", "m31b_taillpips"]),
    "cP": (["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7",
            "m31b_nolpips", "m31b_taillpips"],          # majority all UT-lineage (prod-shaped)
           ["fid_ctrl", "fid_l2reg"]),                  # minority a single coherent family
}

DIRS = {}
for m in MEM:
    for sub in ("test_poses_renders_png", "tp_png"):
        p = f"/mnt/d/avv/output/{TAG}_{m}/{sub}"
        if os.path.isdir(p):
            DIRS[m] = p
            break
    assert m in DIRS, m


def ld(p, dev):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)


def restore_fast(ens, mem_L0, lam, k, win=3, nlev=5, clamp=4.0):
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

    tst = [ld(os.path.join(DIRS[m], stems[0] + ".png"), dev) for m in MEM[:8]]
    e = torch.stack(tst).mean(0)
    L0s = [lap_pyr(m, 5, _K.to(dev))[0][0] for m in tst]
    d = float((restore(e, tst, 1.0, 8) - restore_fast(e, L0s, 1.0, 8)).abs().max())
    print(f"restore_fast vs production restore: max|diff| = {d:.3e}", flush=True)
    assert d < 1e-5
    del tst, e, L0s

    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in cache["HW"]]
    lens = upsample(LooPool(cache["s8"]).pooled("median"), H, W, "cubic") * FIELD_GAIN

    # arm spec: (name, cfg, wB or None, lam)   wB=None -> family-mean-only arm ('A'/'B' in cfg slot)
    arms = []
    for c in ("c4", "c1", "cP"):
        for w in (0.25, 1.0 / 3.0):
            arms.append((f"{c}_w{w:.4f}_lam1", c, w, 1.0))
    for c in ("c4", "cP"):
        for w in (0.25, 1.0 / 3.0):
            arms.append((f"{c}_w{w:.4f}_lam0", c, w, 0.0))
    arms.append(("cP_w0.4000_lam1", "cP", 0.40, 1.0))
    # family-mean quality, no restore (so no k confound)
    arms.append(("QUAL_A_claim", "c4", "A", 0.0))
    arms.append(("QUAL_B_c4", "c4", "B", 0.0))
    arms.append(("QUAL_B_c1", "c1", "B", 0.0))
    arms.append(("QUAL_A_cP", "cP", "A", 0.0))
    arms.append(("QUAL_B_cP", "cP", "B", 0.0))
    names = [a[0] for a in arms]

    acc = {a: [0.0, 0.0, 0.0] for a in names}
    per = {a: [] for a in names}
    nb = {a: 0 for a in names}
    stems = stems[:nlim]
    t0 = time.time()
    for n, s in enumerate(stems):
        x = [ld(os.path.join(DIRS[m], s + ".png"), dev) for m in MEM]
        L0 = [lap_pyr(m, 5, _K.to(dev))[0][0] for m in x]          # HOISTED
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(gtd, gt_by[s])).convert("RGB"),
                                        dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        fam = {}
        for c, (A, B) in CFG.items():
            ia = [IDX[m] for m in A]; ib = [IDX[m] for m in B]
            fam[c] = (torch.stack([x[i] for i in ia]).mean(0),
                      torch.stack([x[i] for i in ib]).mean(0),
                      [L0[i] for i in ia + ib])
        for (nm, c, w, lam) in arms:
            mA, mB, mL0 = fam[c]
            if w == "A":
                ens, ml, k = mA, mL0[:6], 6
            elif w == "B":
                ens, ml, k = mB, mL0[6:], 2
            else:
                ens, ml, k = (1 - w) * mA + w * mB, mL0, 8
            o = (restore_fast(ens, ml, lam, k) if lam > 0 else ens).clamp(0, 1)
            o = o[0].permute(1, 2, 0).cpu().numpy()
            y = np.clip(warp(o, lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((y * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            nb[nm] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, g))
                L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            acc[nm][0] += P; acc[nm][1] += S; acc[nm][2] += L
            per[nm].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
        if n % 5 == 0:
            print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    out = {}
    for a in names:
        P, S, L = (v / N for v in acc[a])
        out[a] = (100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)), P, S, L)
    print(f"\nINDEPENDENT REFUTATION RUN  {TAG}  n={N}  FULL SHIPPED CHAIN "
          f"(field x{FIELD_GAIN} lanczos4, JPEG q100/ss2)")
    print(f"{'arm':>20} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'MB/60':>7}")
    for a in names:
        sc, P, S, L = out[a]
        print(f"{a:>20} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {nb[a]/1e6:7.2f}")

    def paired(a, b):
        d = np.array(per[a]) - np.array(per[b])
        se = d.std(ddof=1) / np.sqrt(N)
        return d.mean(), se, d.mean() / se if se > 0 else 0.0, int((d > 0).sum())

    print("\nPAIRED CONTRASTS  (0.3333 minus 0.2500 : positive => the claim's direction)")
    for c in ("c4", "c1", "cP"):
        for lam in ("lam1", "lam0"):
            a, b = f"{c}_w0.3333_{lam}", f"{c}_w0.2500_{lam}"
            if a in per and b in per:
                m, se, t, w = paired(a, b)
                print(f"  {c} {lam}:  {m:+.4f}  95%CI [{m-1.96*se:+.4f},{m+1.96*se:+.4f}]  "
                      f"t={t:+6.2f}  wins {w}/{N}")
    print("\nFAMILY QUALITY THROUGH THE CHAIN (lam=0, no restore)")
    for a in ("QUAL_A_claim", "QUAL_B_c4", "QUAL_B_c1", "QUAL_A_cP", "QUAL_B_cP"):
        print(f"  {a:>14} {out[a][0]:9.4f}")
    print(f"  minority-minus-majority: c4 {out['QUAL_B_c4'][0]-out['QUAL_A_claim'][0]:+.4f} | "
          f"c1 {out['QUAL_B_c1'][0]-out['QUAL_A_claim'][0]:+.4f} | "
          f"cP {out['QUAL_B_cP'][0]-out['QUAL_A_cP'][0]:+.4f}")
    m, se, t, w = paired("cP_w0.4000_lam1", "cP_w0.3333_lam1")
    print(f"\ncP 0.40 vs 0.3333: {m:+.4f} t={t:+.2f}")
    json.dump({"score": out, "per": per}, open(f"{HERE}/refute_wt.json", "w"), indent=1)
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
