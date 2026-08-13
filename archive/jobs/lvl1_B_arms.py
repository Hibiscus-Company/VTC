#!/usr/bin/env python
"""PHASE B -- a LEVEL-1 energy-restore term with its own lambda, scored through the FULL shipped
chain (mean -> restore -> median lens field lanczos4 -> JPEG q100/ss2) against real test GT.

Phase A showed level 1 is NOT at GT energy at production k: E(L1_mean)/E(L1_gt) falls from 0.964
(k=1) to 0.907 (k=8), and the operator's own disagreement statistic still UNDER-shoots there
(0.937 < 1), so lam1=1.0 cannot overshoot in the global-energy sense.

Arms, all with the shipped lam0=1.0 unless stated:
  base            lam1=0            <- exactly what r28/r29 ship
  l1_0.25/0.5/1.0 the level-1 term
  l1only_0.5      lam0=0, lam1=0.5  <- is level 1 a COMPLEMENT or a SUBSTITUTE for level 0?
  ctrl_shuf_0.5   r1 map spatially shuffled: same histogram, wrong geography
  ctrl_const_0.5  r1 replaced by its per-image mean: pure global level-1 band gain
The two controls isolate the effect: if either matches the real map, the level-1 "disagreement"
information does nothing and this is just band sharpening (which is already dead by proof).

HOISTED: every member pyramid, both r maps and the band products are computed ONCE per image and
reused by all arms.  Only warp/JPEG/LPIPS run per arm.  GPU touched only for LPIPS+SSIM.
"""
import io, json, os, sys, time
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
TAG = "HCM0181"
D = lambda m: f"/mnt/d/avv/output/HCM0181_{m}/test_poses_renders_png"
UT4 = ["gsplatB9ut", "gsplatB10ut8M", "gsplatB11ut60k", "gsplatB12ut8Ms7"]
DIV4 = ["gsplatB8pure", "e17visnorm", "m31b_taillpips", "gsplatB6bilagrid"]
POOL8 = UT4 + DIV4
NLEV, WIN, CLAMP = 5, 3, 4.0

# (name, pool, lam0, lam1, map1)
ARMS = [
    ("k8 base(shipped)",   "p8", 1.0, 0.0,  "raw"),
    ("k8 l1_0.25",         "p8", 1.0, 0.25, "raw"),
    ("k8 l1_0.50",         "p8", 1.0, 0.50, "raw"),
    ("k8 l1_1.00",         "p8", 1.0, 1.00, "raw"),
    ("k8 l1only_0.50",     "p8", 0.0, 0.50, "raw"),
    ("k8 CTRL_shuf_0.50",  "p8", 1.0, 0.50, "shuffle"),
    ("k8 CTRL_const_0.50", "p8", 1.0, 0.50, "const"),
    ("k4ut base(shipped)", "p4", 1.0, 0.0,  "raw"),
    ("k4ut l1_0.50",       "p4", 1.0, 0.50, "raw"),
    ("k4ut l1_1.00",       "p4", 1.0, 1.00, "raw"),
]


def ld(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def rmap(laps_i, Lm, k):
    """the shipped statistic, at whatever level the caller passes in."""
    Eb = boxf((Lm ** 2).sum(1, keepdim=True), WIN)
    V = 0.0
    for Li in laps_i:
        V = V + boxf(((Li - Lm) ** 2).sum(1, keepdim=True), WIN)
    V = V / len(laps_i) * (k / (k - 1.0))
    return torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=CLAMP)


def main():
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
    gtd = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(D(m), s + ".png")) for m in POOL8))
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    H, W = [int(x) for x in cache["HW"]]
    lens = upsample(LooPool(cache["s8"]).pooled("median"), H, W, "cubic")

    names = [a[0] for a in ARMS]
    acc = {n: [0.0, 0.0, 0.0] for n in names}
    nbytes = {n: 0 for n in names}
    t0 = time.time()
    for c, s in enumerate(stems):
        g = ld(os.path.join(gtd, gt_by[s])).to(dev)
        mem = [ld(os.path.join(D(m), s + ".png")) for m in POOL8]
        # ---- HOISTED: one pyramid per member, shared by every arm and both pools ----
        pyr = [lap_pyr(m, NLEV, _K) for m in mem]
        prep = {}
        for tag, idx in (("p8", list(range(8))), ("p4", list(range(4)))):
            k = len(idx)
            laps = [torch.stack([pyr[i][0][l] for i in idx]).mean(0) for l in range(NLEV)]
            res = torch.stack([pyr[i][1] for i in idx]).mean(0)
            sizes = pyr[0][2]
            r0 = rmap([pyr[i][0][0] for i in idx], laps[0], k)
            r1 = rmap([pyr[i][0][1] for i in idx], laps[1], k)
            f1 = r1.reshape(-1)
            gen = torch.Generator().manual_seed(1234 + c)
            r1s = f1[torch.randperm(f1.numel(), generator=gen)].reshape(r1.shape)
            prep[tag] = dict(laps=laps, res=res, sizes=sizes, r0=r0,
                             r1={"raw": r1, "shuffle": r1s, "const": r1.mean().expand_as(r1)})
        for name, tag, lam0, lam1, m1 in ARMS:
            p = prep[tag]
            out = list(p["laps"])
            out[0] = out[0] * (1.0 + lam0 * (p["r0"] - 1.0))
            if lam1 != 0.0:
                out[1] = out[1] * (1.0 + lam1 * (p["r1"][m1] - 1.0))
            x = lap_recon(out, p["res"], p["sizes"], _K).clamp(0, 1)[0].permute(1, 2, 0).numpy()
            x = np.clip(warp(x, lens, "lanczos"), 0, 1)
            b = io.BytesIO()
            Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
            nbytes[name] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                acc[name][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[name][1] += float(repo_ssim(r, g))
                acc[name][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
        if c % 10 == 0:
            print(f"  {c}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    N = len(stems)
    print(f"\nLEVEL-1 ENERGY TERM, {TAG}, n={N}, FULL shipped chain "
          f"(restore -> median field lanczos4 -> JPEG q100/ss2)")
    print(f"{'arm':>20} {'SCORE':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>8} {'vs base':>9} {'MB/60':>7}")
    res = {}
    b8 = b4 = None
    for name, tag, *_ in ARMS:
        P, S, L = (x / N for x in acc[name])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        if name == "k8 base(shipped)":
            b8 = sc
        if name == "k4ut base(shipped)":
            b4 = sc
        base = b4 if tag == "p4" else b8
        res[name] = dict(score=sc, psnr=P, ssim=S, lpips=L, delta=sc - base,
                         mb=nbytes[name] / 1e6)
        print(f"{name:>20} {sc:9.4f} {P:8.4f} {S:7.4f} {L:8.4f} {sc-base:+9.4f} "
              f"{nbytes[name]/1e6:7.2f}")
    json.dump(res, open(f"{HERE}/lvl1_B_arms.json", "w"), indent=1)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
