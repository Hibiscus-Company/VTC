#!/usr/bin/env python
"""ADVERSARIAL RE-MEASUREMENT, PART 2: is the claim's +0.0188 an artifact of the MAJORITY pool?

Part 1 varied the minority family and found the sign flips (1/5 analogues positive).
Part 2 holds the claim's own minority family FIXED (fid_ctrl + m31b_taillpips) and varies only
the 6-member MAJORITY pool. The claim's A6 contains gsplatB1 and gsplatB2, whose solo LPIPS is
0.1307/0.1308 against 0.1091-0.1168 for the other four -- two clear outliers. Production's
majority is the r22 6-member mean, a homogeneous ut7/ut13/ut42/ut77/seed101(+1) pool with no
such outliers. If shrinking the majority weight only pays because it dilutes two weak members,
the effect must vanish when the outliers are replaced by ordinary members.

Structure, chain, restore k, field, JPEG and n identical to the claim's cfg4 throughout.
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
from fieldlib import LooPool, upsample, warp

Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG, FIELD_GAIN = "HCM0181", 1.30
DEV = os.environ.get("REF_DEV", "cuda:0")

UT4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
B2 = ["fid_ctrl", "m31b_taillpips"]                      # the CLAIM's minority family, fixed
APOOLS = {
    "A_claim  (UT4+B1+B2 weak)": UT4 + ["gsplatB1", "gsplatB2"],
    "A_homo   (UT4+e17+B8pure)": UT4 + ["e17visnorm", "gsplatB8pure"],
    "A_homo2  (UT4+e15+B4warm)": UT4 + ["e15ceil95", "gsplatB4warm"],
}
WGRID = [0.2500, 0.3333, 0.4000, 0.5000]
MEM = list(dict.fromkeys([m for v in APOOLS.values() for m in v] + B2))

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
                   if all(os.path.exists(os.path.join(DIRS[m], s + ".png")) for m in MEM))[:nlim]
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in cache["HW"]]
    lens = upsample(LooPool(cache["s8"]).pooled("median"), H, W, "cubic") * FIELD_GAIN

    arms = [(a, w) for a in APOOLS for w in WGRID]
    acc = {a: np.zeros(3) for a in arms}
    per = {a: [] for a in arms}
    t0 = time.time()
    for n, s in enumerate(stems):
        x = {m: ld(os.path.join(DIRS[m], s + ".png"), dev) for m in MEM}
        L0 = {m: lap_pyr(x[m], 5, _K.to(dev))[0][0] for m in MEM}
        g = ld(os.path.join(gtd, gt_by[s]), dev)
        mB = torch.stack([x[m] for m in B2]).mean(0)
        for (ap, w) in arms:
            A = APOOLS[ap]
            mA = torch.stack([x[m] for m in A]).mean(0)
            ens = (1 - w) * mA + w * mB
            o = restore_fast(ens, [L0[m] for m in A + B2], 1.0, 8)
            o = o.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
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
            acc[(ap, w)] += (P, S, L)
            per[(ap, w)].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
        if n % 10 == 0:
            print(f"  {n}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    out = {a: (lambda P, S, L: (100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)),
                                P, S, L))(*(acc[a] / N)) for a in arms}
    print(f"\nMAJORITY-POOL CONTROL, {TAG}, n={N}, minority family FIXED at {B2}, k=8, full chain")
    hdr = " ".join(f"{w:>9.4f}" for w in WGRID)
    print(f"{'majority pool A6':>27} {hdr}   {'argmax':>7} {'d(.40-.3333)':>13} {'t':>7} {'wins':>7}")
    res = {}
    for ap in APOOLS:
        sc = [out[(ap, w)][0] for w in WGRID]
        d = np.array(per[(ap, 0.4000)]) - np.array(per[(ap, 0.3333)])
        t = d.mean() / (d.std(ddof=1) / np.sqrt(N))
        res[ap] = float(d.mean())
        print(f"{ap:>27} " + " ".join(f"{v:9.4f}" for v in sc) +
              f"   {WGRID[int(np.argmax(sc))]:7.4f} {d.mean():+13.4f} {t:7.2f} "
              f"{int((d>0).sum()):3d}/{N:<3d}")
    json.dump(res, open(f"{HERE}/ref_wt2.json", "w"), indent=1)
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
