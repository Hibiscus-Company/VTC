#!/usr/bin/env python
"""LPIPS-DIRECT diagnostic B: the finest band L0 is the ONLY deficient band (diag A).  Its GT
coherence is rho~0.83 and its amplitude after the shipped restore is still only 0.86 of GT.

The shipped operator's gain r = sqrt(1 + c*V/E) is an INCREASING function of the ensemble
disagreement V.  Two rival explanations:
  (H1) coherence-blind "replace lost amplitude": the operator just restores the amplitude that
       averaging destroyed; the residual deficit (0.86 vs 1.00) is UNIFORM over the disagreement
       map, so a FLAT extra L0 gain would be an equally good -- possibly better -- shape.
  (H2) coherence-aware: our L0 content is much more GT-coherent where members AGREE, so boosting
       high-V regions is boosting garbage and a coherence-DAMPED shape would beat it.
Decide by measuring amp and rho per decile of the shipped r-map.
CPU only.
"""
import io, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, boxf, _K            # noqa: E402
from fieldlib import LooPool, upsample, warp     # noqa: E402

Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(6)
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
C_SHIP = 4.0 / 3.0
ND = 10


def u8(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).float().div_(255.0) \
        .permute(2, 0, 1).unsqueeze(0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:n]
    cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
    K = _K
    # accumulators, per decile of r
    A = {k: np.zeros(ND) for k in ("mxx", "mgg", "mxg", "pxx", "pgg", "pxg", "npx", "rsum")}
    t0 = time.time()
    for ci, s in enumerate(stems):
        mem = [to_t(u8(os.path.join(d, s + ".png"))) for d in MEM]
        ens = torch.stack(mem).mean(0)
        g = to_t(u8(os.path.join(GTD, gt_by[s])))
        laps, res, sizes = lap_pyr(ens, 5, K)
        L0 = laps[0]
        Eb = boxf((L0 ** 2).sum(1, keepdim=True), 3)
        V = sum(boxf(((lap_pyr(m, 1, K)[0][0] - L0) ** 2).sum(1, keepdim=True), 3)
                for m in mem) / len(mem)
        r = torch.sqrt(1.0 + C_SHIP * V / (Eb + 1e-10)).clamp(max=4.0)
        x = (ens + (r - 1.0) * L0).clamp(0, 1)
        xn = np.clip(warp(np.ascontiguousarray(x[0].permute(1, 2, 0).numpy()), lens, "lanczos"),
                     0, 1)
        b = io.BytesIO()
        Image.fromarray((xn * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", **SHIPPED)
        xj = to_t(np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.uint8))
        Lp = lap_pyr(xj, 1, K)[0][0]
        Lg = lap_pyr(g, 1, K)[0][0]
        rr = r[0, 0]
        qs = torch.quantile(rr.flatten()[::13].float(),
                            torch.linspace(0, 1, ND + 1)[1:-1])
        idx = torch.bucketize(rr, qs)
        for d in range(ND):
            m = (idx == d)
            mm = m.unsqueeze(0).unsqueeze(0).expand_as(L0)
            a = L0[mm]; c = Lg[mm]; p = Lp[mm]
            A["mxx"][d] += float((a * a).sum()); A["mxg"][d] += float((a * c).sum())
            A["mgg"][d] += float((c * c).sum())
            A["pxx"][d] += float((p * p).sum()); A["pxg"][d] += float((p * c).sum())
            A["pgg"][d] += float((c * c).sum())
            A["npx"][d] += float(m.sum()); A["rsum"][d] += float(rr[m].sum())
        if (ci + 1) % 4 == 0:
            print(f"  {ci+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    print(f"\n=== L0 by decile of the shipped r-map, HCM0181 n={len(stems)} ===")
    print(f"{'dec':>4} {'<r>':>7} {'%L0energy':>10} | {'RAW MEAN':^24} | {'AFTER SHIPPED CHAIN':^24}")
    print(f"{'':>4} {'':>7} {'':>10} | {'amp':>7} {'rho':>7} {'gstar':>7} | "
          f"{'amp':>7} {'rho':>7} {'gstar':>7}")
    for d in range(ND):
        rm = A["rsum"][d] / A["npx"][d]
        fr = 100 * A["mxx"][d] / A["mxx"].sum()
        amp_m = np.sqrt(A["mxx"][d] / A["mgg"][d]); rho_m = A["mxg"][d] / np.sqrt(A["mxx"][d] * A["mgg"][d])
        amp_p = np.sqrt(A["pxx"][d] / A["pgg"][d]); rho_p = A["pxg"][d] / np.sqrt(A["pxx"][d] * A["pgg"][d])
        print(f"{d:>4} {rm:7.3f} {fr:10.2f} | {amp_m:7.4f} {rho_m:7.4f} {A['mxg'][d]/A['mxx'][d]:7.4f} | "
              f"{amp_p:7.4f} {rho_p:7.4f} {A['pxg'][d]/A['pxx'][d]:7.4f}")
    print(f"\nTOTAL raw  amp {np.sqrt(A['mxx'].sum()/A['mgg'].sum()):.4f} "
          f"rho {A['mxg'].sum()/np.sqrt(A['mxx'].sum()*A['mgg'].sum()):.4f}")
    print(f"TOTAL ship amp {np.sqrt(A['pxx'].sum()/A['pgg'].sum()):.4f} "
          f"rho {A['pxg'].sum()/np.sqrt(A['pxx'].sum()*A['pgg'].sum()):.4f}")


if __name__ == "__main__":
    main()
