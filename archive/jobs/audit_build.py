#!/usr/bin/env python
"""BUILD-PATH FAULT AUDIT -- measured on the production harness through the FULL shipped chain.

Arms (HCM0181, 60 REAL test poses, REAL test GT, 4-member UT pool, lam=1.0 tower setting):
  prod_q100   exactly what build_r28/r29 does for HCM0539/40/44/74/chair/bonsai:
              mean -> uint8 PNG -> energy restore -> uint8 PNG -> lanczos warp -> uint8 PNG
              -> JPEG q100/ss2/progressive/optimize
  prod_q99    identical, but the JPEG quality the Q dict hard-codes for HCM0421
  float_q100  the same operators with NO intermediate uint8 PNG round-trips
              (isolates the cost of the 3 PNG requantisations the build inserts)

Shared computation (member Laplacian L0 bands) is HOISTED out of the arm loop -- computing it
per arm would be 3x redundant and would starve the two production trainings.
"""
import io, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, upsample, warp
from energy_restore import restore as prod_restore
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg

Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
TAG = "HCM0181"
MEM = [f"/mnt/d/avv/output/{TAG}_gsplatB9ut/test_poses_renders_png",
       f"/mnt/d/avv/output/{TAG}_gsplatB10ut8M/test_poses_renders_png",
       f"/mnt/d/avv/output/{TAG}_gsplatB11ut60k/test_poses_renders_png",
       f"/mnt/d/avv/output/{TAG}_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = f"/mnt/d/avv/data/phase1/public_set/{TAG}/test/images"
LAM = 1.0
CPU = torch.device("cpu")
GPU = torch.device("cuda")
torch.set_num_threads(8)          # leave cores for the two production trainings

K = _K.to(CPU)


def u8(x):                        # float [0,1] HxWx3 -> the exact bytes a PNG stage writes
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8).astype(np.float32) / 255.0


def to_t(a):                      # HxWx3 float -> [1,3,H,W]
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0)


def to_np(t):
    return t[0].permute(1, 2, 0).numpy()


def restore_fast(ens, mem_l0, lam, k, win=3, nlev=5, clamp=4.0):
    """Identical math to energy_restore.restore, but with the member L0 bands passed in."""
    laps, res, sizes = lap_pyr(ens, nlev, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    V = 0.0
    for ml in mem_l0:
        V = V + boxf(((ml - L0) ** 2).sum(1, keepdim=True), win)
    V = V / len(mem_l0) * (k / (k - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)
    return lap_recon([L0 * (1.0 + lam * (r - 1.0))] + laps[1:], res, sizes, K)


def enc_dec(x, q):
    b = io.BytesIO()
    kw = dict(SHIPPED); kw["quality"] = q
    Image.fromarray((np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)).save(b, "JPEG", **kw)
    raw = b.getvalue()
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255.0, len(raw)


def main():
    vgg = lpips_pkg.LPIPS(net="vgg").to(GPU).eval()
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    cache = np.load(f"{HERE}/lens/cache/pub_{TAG}.npz")
    lens = upsample(LooPool(cache["s8"]).pooled("median"), *[int(x) for x in cache["HW"]], "cubic")
    print(f"{TAG}: n={len(stems)} stems, {len(MEM)} members, lam={LAM}", flush=True)

    arms = ["prod_q100", "prod_q99", "float_q100"]
    acc = {a: [0.0, 0.0, 0.0] for a in arms}
    nb = {a: 0 for a in arms}
    t0 = time.time()
    for c, s in enumerate(stems):
        mem_np = [np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                             dtype=np.float32) / 255.0 for d in MEM]
        mem_t = [to_t(a) for a in mem_np]
        mem_l0 = [lap_pyr(m, 5, K)[0][0] for m in mem_t]        # HOISTED: once per image
        mean_f = np.mean(mem_np, axis=0)                        # ensemble_renders float accumulate

        if c == 0:   # prove restore_fast == the production operator, bit level
            a1 = prod_restore(to_t(u8(mean_f)), mem_t, LAM, len(MEM))
            a2 = restore_fast(to_t(u8(mean_f)), mem_l0, LAM, len(MEM))
            print("   restore_fast max|diff| vs production restore:",
                  float((a1 - a2).abs().max()), flush=True)

        # ---- production chain: PNG round-trip after the mean, after ER, after the warp
        er_p = to_np(restore_fast(to_t(u8(mean_f)), mem_l0, LAM, len(MEM)).clamp(0, 1))
        wp_p = u8(np.clip(warp(u8(er_p), lens, "lanczos"), 0, 1))
        # ---- float chain: same operators, no intermediate uint8
        er_f = to_np(restore_fast(to_t(mean_f), mem_l0, LAM, len(MEM)).clamp(0, 1))
        wp_f = np.clip(warp(er_f, lens, "lanczos"), 0, 1)

        outs = {"prod_q100": (wp_p, 100), "prod_q99": (wp_p, 99), "float_q100": (wp_f, 100)}
        g = torch.from_numpy(np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"),
                                        dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(GPU)
        for a in arms:
            x, q = outs[a]
            j, n = enc_dec(x, q)
            nb[a] += n
            r = to_t(j).to(GPU)
            with torch.no_grad():
                acc[a][0] += 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                acc[a][1] += float(repo_ssim(r, g))
                acc[a][2] += float(vgg(r * 2 - 1, g * 2 - 1).item())
            del r
        del g
        if c % 10 == 0:
            torch.cuda.empty_cache()
            print(f"  {c}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    n = len(stems)
    print(f"\nFULL SHIPPED CHAIN, {TAG}, n={n}, lam={LAM}")
    print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs prod_q100':>13} "
          f"{'MB/60':>8} {'dMB':>8}")
    base = bb = None
    for a in arms:
        P, S, L = (x / n for x in acc[a])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        if base is None:
            base, bb = sc, nb[a]
        print(f"{a:>12} {sc:9.4f} {P:8.4f} {S:8.4f} {L:8.4f} {sc-base:+13.4f} "
              f"{nb[a]/1e6:8.2f} {(nb[a]-bb)/1e6:+8.2f}")
    print(f"\nblended (1 of 7 scenes) delta for switching ONE scene q99->q100: "
          f"{-(0.0):+.4f} -- see per-scene numbers above, divide by 7")


if __name__ == "__main__":
    main()
