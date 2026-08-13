#!/usr/bin/env python
"""LPIPS-DIRECT diagnostic A: WHERE, in (Laplacian band x edge-region), is our production output
in EXCESS vs DEFICIT vs GT, and how COHERENT is it there?

Motivation: the per-VGG-layer table says relu2_2 (1/2 resolution -> Laplacian LEVEL 1) carries the
single largest share of LPIPS (25.4%) and its flat-region density is 1.259 (vs 0.607 at strong
edges).  Everything we ship acts on LEVEL 0 only.  The dead flat-attenuation test also acted on
LEVEL 0 only.  So band x region is an unmeasured plane.

For each band l and region m we report
    amp   = sqrt(<Lx^2>/<Lg^2>)     >1 = we have MORE energy than GT there (excess -> attenuate)
    rho   = <Lx,Lg>/sqrt(...)       coherence of our content with GT's
    gstar = <Lx,Lg>/<Lx^2>          MSE-optimal gain on that band+region
CPU only.  No GPU touched.
"""
import io, os, sys, time
import numpy as np
import torch
import cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, boxf, _K            # noqa: E402
from fieldlib import LooPool, upsample, warp     # noqa: E402

Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(6)
cv2.setNumThreads(4)
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
NLEV = 4
C_SHIP = 4.0 / 3.0


def u8(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).float().div_(255.0) \
        .permute(2, 0, 1).unsqueeze(0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:n]
    cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
    K = _K

    MASKS = ["flat<p50", "low p50-80", "edge p80-95", "strong>p95"]
    NM = len(MASKS)
    S_xx = np.zeros((NLEV, NM)); S_gg = np.zeros((NLEV, NM)); S_xg = np.zeros((NLEV, NM))
    S_n = np.zeros((NLEV, NM))
    # also: same stats for the RAW ensemble mean (pre-restore, pre-warp, pre-jpeg) as reference
    R_xx = np.zeros((NLEV, NM)); R_gg = np.zeros((NLEV, NM)); R_xg = np.zeros((NLEV, NM))
    t0 = time.time()
    for ci, s in enumerate(stems):
        mem = [to_t(u8(os.path.join(d, s + ".png"))) for d in MEM]
        ens = torch.stack(mem).mean(0)
        g = to_t(u8(os.path.join(GTD, gt_by[s])))
        # SHIP: L0 energy restore lam=1
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

        # masks from GT gradient
        gray = cv2.cvtColor((g[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
        gmag = np.sqrt(gx * gx + gy * gy)
        q = np.quantile(gmag, [0.50, 0.80, 0.95])
        mfull = [gmag < q[0], (gmag >= q[0]) & (gmag < q[1]),
                 (gmag >= q[1]) & (gmag < q[2]), gmag >= q[2]]

        lx = lap_pyr(xj, NLEV, K)[0]
        lg = lap_pyr(g, NLEV, K)[0]
        lr = lap_pyr(ens, NLEV, K)[0]
        for l in range(NLEV):
            step = 2 ** l
            H, W = lx[l].shape[-2:]
            for j, mk in enumerate(mfull):
                mm = torch.from_numpy(mk[::step, ::step][:H, :W].copy())
                a = lx[l][0].permute(1, 2, 0)[mm]
                c = lg[l][0].permute(1, 2, 0)[mm]
                e = lr[l][0].permute(1, 2, 0)[mm]
                S_xx[l, j] += float((a * a).sum()); S_gg[l, j] += float((c * c).sum())
                S_xg[l, j] += float((a * c).sum()); S_n[l, j] += a.numel()
                R_xx[l, j] += float((e * e).sum()); R_gg[l, j] += float((c * c).sum())
                R_xg[l, j] += float((e * c).sum())
        if (ci + 1) % 5 == 0:
            print(f"  {ci+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)

    def report(name, XX, GG, XG):
        print(f"\n=== {name} : per Laplacian band x GT-gradient region, HCM0181 n={len(stems)} ===")
        print(f"{'band':>6} {'region':>12} {'%ofL0energy':>12} {'amp=sqrt(Ex/Eg)':>16} "
              f"{'rho':>8} {'gstar':>8}")
        for l in range(NLEV):
            for j in range(NM):
                amp = np.sqrt(XX[l, j] / max(GG[l, j], 1e-20))
                rho = XG[l, j] / max(np.sqrt(XX[l, j] * GG[l, j]), 1e-20)
                gs = XG[l, j] / max(XX[l, j], 1e-20)
                frac = 100 * XX[l, j] / XX.sum()
                print(f"L{l:<5d} {MASKS[j]:>12} {frac:12.2f} {amp:16.4f} {rho:8.4f} {gs:8.4f}")

    report("PRODUCTION OUTPUT (restore->warp->jpeg) vs GT", S_xx, S_gg, S_xg)
    report("RAW ENSEMBLE MEAN vs GT", R_xx, R_gg, R_xg)


if __name__ == "__main__":
    main()
