#!/usr/bin/env python
"""ADVERSARIAL RE-MEASUREMENT of "raise the mip3d family weight 0.3333 -> 0.40".

The claim rests on ONE choice of minority-family ANALOGUE (fid_ctrl + m31b_taillpips), because
the production harness contains no mip3d variant. The analogue is a free parameter. So sweep it:
hold A6, the chain, the grid and n fixed, and vary ONLY which 2 members play the minority family.
If the sign of (wB=0.40) - (wB=0.3333) flips across plausible analogues, the claimed +0.0188 is a
draw from an analogue-selection distribution, not a property of the weight.

All arms share the SAME per-stem member loads and Laplacian bands (hoisted out of the arm loop).
"""
import io, os, sys, time, json
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from energy_restore import restore
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
FIELD_GAIN = 1.30
DEV = os.environ.get("REF_DEV", "cuda:0")

A6 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7", "gsplatB1", "gsplatB2"]

# minority-family analogues.  #1 and #2 are the claim's; #3/#4 are the SAME AGENT's own earlier
# runs of the same script (wt_chain.log / wt_chain3.log) that were not reported; #5/#6 are new
# pairs built only from non-perceptual-loss recipes -- the regime mip3d actually sits in.
FAMS = {
    "CLAIM  fid_ctrl+taillpips": ["fid_ctrl", "m31b_taillpips"],
    "unrep1 nolpips+taillpips ": ["m31b_nolpips", "m31b_taillpips"],
    "unrep3 e17+nolpips       ": ["e17visnorm", "m31b_nolpips"],
    "new    e15+e17 (geom)    ": ["e15ceil95", "e17visnorm"],
    "new    B8pure+B4warm     ": ["gsplatB8pure", "gsplatB4warm"],
}
WGRID = [0.3333, 0.4000]
MEM = list(dict.fromkeys(A6 + [m for v in FAMS.values() for m in v]))

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
    dev = DEV
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(DIRS[m], s + ".png")) for m in MEM))
    stems = stems[:nlim]

    # identity of the hoisted operator against the production operator, on real data
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

    arms = [(f, w) for f in FAMS for w in WGRID]
    acc = {a: np.zeros(3) for a in arms}
    per = {a: [] for a in arms}
    t0 = time.time()
    for n, s in enumerate(stems):
        x = {m: ld(os.path.join(DIRS[m], s + ".png"), dev) for m in MEM}
        L0 = {m: lap_pyr(x[m], 5, _K.to(dev))[0][0] for m in MEM}
        g = ld(os.path.join(gtd, gt_by[s]), dev)
        mA = torch.stack([x[m] for m in A6]).mean(0)
        for (f, w) in arms:
            B = FAMS[f]
            mB = torch.stack([x[m] for m in B]).mean(0)
            ens = (1 - w) * mA + w * mB
            ml = [L0[m] for m in A6 + B]                 # k=8, production structure
            o = restore_fast(ens, ml, 1.0, 8).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            y = np.clip(warp(o, lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((y * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, g))
                L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            acc[(f, w)] += (P, S, L)
            per[(f, w)].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
        if n % 5 == 0:
            print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    out = {}
    for a in arms:
        P, S, L = acc[a] / N
        out[a] = (100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)), P, S, L)
    print(f"\nANALOGUE SWEEP, {TAG}, n={N}, FULL SHIPPED CHAIN, A6 fixed, restore lam=1.0 k=8")
    print(f"{'minority family B':>28} {'S@0.3333':>9} {'S@0.4000':>9} {'delta':>9} "
          f"{'t':>7} {'wins':>6} {'dPSNR':>8} {'dSSIM':>9} {'dLPIPS':>9}")
    rows = {}
    for f in FAMS:
        a0, a1 = (f, WGRID[0]), (f, WGRID[1])
        d = np.array(per[a1]) - np.array(per[a0])
        t = d.mean() / (d.std(ddof=1) / np.sqrt(N))
        rows[f] = float(out[a1][0] - out[a0][0])
        print(f"{f:>28} {out[a0][0]:9.4f} {out[a1][0]:9.4f} {out[a1][0]-out[a0][0]:+9.4f} "
              f"{t:7.2f} {int((d>0).sum()):3d}/{N:<3d} {out[a1][1]-out[a0][1]:+8.4f} "
              f"{out[a1][2]-out[a0][2]:+9.5f} {out[a1][3]-out[a0][3]:+9.5f}")
    v = np.array(list(rows.values()))
    print(f"\nacross {len(v)} analogues: mean {v.mean():+.4f}  sd {v.std(ddof=1):.4f}  "
          f"min {v.min():+.4f}  max {v.max():+.4f}  positive {int((v>0).sum())}/{len(v)}")
    json.dump({str(k): v for k, v in rows.items()}, open(f"{HERE}/ref_wt.json", "w"), indent=1)
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
