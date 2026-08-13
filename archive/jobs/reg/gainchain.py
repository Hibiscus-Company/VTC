#!/usr/bin/env python
"""RESIDUAL-REGISTRATION LENS: does the lens field's AMPLITUDE survive the r28 shipped chain,
and is the amplitude deficit isotropic (a scalar gain) or does it carry spatial structure?

Prior art in this job (do not redo):
  * sweep.py 'scale2'/'ship': single-member B9ut renders, 5 public towers, PNG only.
    median field * 1.35 = +0.1156 over *1.0, 5/5 scenes.
  * sweep.py 'k4scale': HCM0181 4-member UNIFORM mean, PNG and shipped JPEG, no energy restore.
    *1.30 = +0.1389 (JPEG) over *1.0.
  * looscale.py: leave-one-view-out on TRAIN views says the optimum is *1.00 and is MONOTONE
    DECREASING after it, on 10/10 towers.  The train protocol therefore MISREADS this lever;
    only real test poses see it.  That is the finding this script builds on.

WHAT IS NEW HERE: the r28 chain is  weighted member mean -> ENERGY RESTORE (lam=1, k) -> field
-> JPEG.  Energy restore multiplies the FINEST Laplacian band by ~1.19 on average; a 1.3x larger
warp destroys more of exactly that band, so the gain and the restore could fight.  Nobody has
measured the scale lever with the restore in the chain.  Arms 1-4 settle that.

Arms 5-8 are the lens question proper: after the scalar gain, is there MORE view-consistent
structure?  Decompose the field into its RADIAL and TANGENTIAL parts about the image centre and
scale them separately (5,6); and give the gain a radial RAMP at fixed mean gain (7,8), which
tests SHAPE at constant magnitude.  If a structured gain beats the scalar, extractable structure
remains; if not, the scalar exhausts it.

CONTROL: arm s1.00 is the exact production field (r28/r29).  Every arm shares one base image
(ensemble + restore computed ONCE per view and hoisted out of the arm loop) so the ONLY thing
that differs between arms is the displacement field.
"""
import os, sys, io, time, json, argparse
import numpy as np
import torch
import cv2
from PIL import Image

HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
from lapfuse import lap_pyr, lap_recon, boxf, _K            # noqa: E402

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(2)
torch.set_num_threads(2)

JPG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
FLOW = f"{HERE}/flow/HCM0181.npz"
DEV = os.environ.get("REGDEV", "cuda")


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEV)


def r8(t):
    return (t.clamp(0, 1) * 255.0 + 0.5).floor().clamp(0, 255) / 255.0


def restore(pyr, memL0, lam, k, K):
    """r28 energy restoration: finest band only, 3x3 box energy, clamp 4."""
    laps, res, sizes = pyr
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), 3)
    V = 0.0
    for ml in memL0:
        V = V + boxf(((ml - L0) ** 2).sum(1, keepdim=True), 3)
    V = V / len(memL0) * (k / (k - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=4.0)
    return lap_recon([L0 * (1.0 + lam * (r - 1.0))] + laps[1:], res, sizes, K)


def build_fields(med, H, W):
    """med: pooled field on the ds8 grid.  Returns {arm: full-res (H,W,2) field}."""
    h, w, _ = med.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx, dy = xx - cx, yy - cy
    rr = np.sqrt(dx * dx + dy * dy)
    ux, uy = dx / (rr + 1e-9), dy / (rr + 1e-9)          # unit radial
    proj = med[..., 0] * ux + med[..., 1] * uy            # signed radial component
    rad = np.stack([proj * ux, proj * uy], -1)            # radial part
    tan = med - rad                                       # tangential part
    rho = rr / rr.max()
    rho_bar = float(rho.mean())

    def up(f):
        return cv2.resize(f.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)

    F = {}
    for a in (1.00, 1.15, 1.30, 1.45):
        F[f"s{a:.2f}"] = up(med * a)
    F["rad1.45_tan1.00"] = up(1.45 * rad + 1.00 * tan)
    F["rad1.00_tan1.45"] = up(1.00 * rad + 1.45 * tan)
    for sgn, nm in ((+1.0, "ramp_up"), (-1.0, "ramp_dn")):
        g = (1.30 + sgn * 0.50 * (rho - rho_bar))[..., None]
        F[nm] = up(med * g)
    stats = {k: dict(mean_d=float(np.linalg.norm(v, axis=2).mean()),
                     max_d=float(np.linalg.norm(v, axis=2).max())) for k, v in F.items()}
    stats["_energy"] = dict(rad_frac=float((rad ** 2).sum() / (med ** 2).sum()),
                            tan_frac=float((tan ** 2).sum() / (med ** 2).sum()))
    return F, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default=f"{HERE}/reg/gainchain.json")
    args = ap.parse_args()

    K = _K.to(DEV)
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))[:args.n]
    z = np.load(FLOW)
    med = np.median(z["ds8"].astype(np.float32), axis=0)
    probe = np.asarray(Image.open(os.path.join(MEM[0], stems[0] + ".png")))
    H, W = probe.shape[:2]
    FLD, fstats = build_fields(med, H, W)
    arms = list(FLD)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    MAPS = {a: ((xx + FLD[a][..., 0]).astype(np.float32),
                (yy + FLD[a][..., 1]).astype(np.float32)) for a in arms}
    del FLD

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(DEV).eval()

    per = {a: [] for a in arms}
    t0 = time.time()
    for c, s in enumerate(stems):
        mem = [load(os.path.join(d, s + ".png")) for d in MEM]
        memL0 = [lap_pyr(m, 5, K)[0][0] for m in mem]                       # HOISTED
        g = load(os.path.join(GTD, gt_by[s]))
        # r28 tower combiner: 0.8 * (3-member mean) + 0.2 * 4th, uint8 at each written stage
        e_p = r8(r8((mem[0] + mem[1] + mem[2]) / 3.0) * 0.8 + 0.2 * mem[3])
        base = restore(lap_pyr(e_p, 5, K), memL0, args.lam, 4, K).clamp(0, 1)
        base_u8 = (base[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        del mem, memL0, e_p, base
        bf = np.ascontiguousarray(base_u8.astype(np.float32) / 255.0)
        for a in arms:
            x = np.clip(cv2.remap(bf, *MAPS[a], cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REFLECT), 0, 1)
            u8 = (x * 255.0 + 0.5).astype(np.uint8)
            b = io.BytesIO(); Image.fromarray(u8).save(b, "JPEG", **JPG)
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32) / 255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(DEV)
            with torch.no_grad():
                P = 10 * np.log10(1.0 / max(float(((r - g) ** 2).mean()), 1e-12))
                S = float(repo_ssim(r, g))
                L = float(vgg(r * 2 - 1, g * 2 - 1).item())
            per[a].append((P, S, L, 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))))
        del g
        if c % 5 == 0:
            print(f"  {c+1}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    n = len(stems)
    A = {a: np.array(per[a]) for a in arms}
    base_s = A["s1.00"][:, 3]
    print(f"\nHCM0181 FULL r28 CHAIN (weighted 4-member mean -> ENERGY RESTORE lam={args.lam} k=4"
          f" -> field*gain lanczos4 -> JPEG q100/ss2), n={n}")
    print(f"{'arm':>18} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} "
          f"{'d(s1.00)':>9} {'se':>7} {'wins':>7} {'mean|d|':>8}")
    res = {}
    for a in arms:
        m = A[a].mean(0); d = A[a][:, 3] - base_s
        se = d.std(ddof=1) / np.sqrt(n) if a != "s1.00" else 0.0
        res[a] = dict(score=m[3], psnr=m[0], ssim=m[1], lpips=m[2],
                      d=float(d.mean()), se=float(se), win=int((d > 0).sum()),
                      **fstats[a])
        print(f"{a:>18} {m[3]:9.4f} {m[0]:8.4f} {m[1]:8.5f} {m[2]:8.5f} "
              f"{d.mean():+9.4f} {se:7.4f} {int((d>0).sum()):4d}/{n} {fstats[a]['mean_d']:8.4f}")
    print(f"\nfield energy split: radial {fstats['_energy']['rad_frac']:.3f}  "
          f"tangential {fstats['_energy']['tan_frac']:.3f}")
    json.dump(dict(res=res, n=n, lam=args.lam, energy=fstats["_energy"]),
              open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
