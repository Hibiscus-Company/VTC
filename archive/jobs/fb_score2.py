#!/usr/bin/env python
"""FREEBIE probe, full run: every zero-cost lever in the shipped artefact, one shared pass.

HCM0181, 60 REAL test poses, REAL test GT, models trained on 100% of train photos.
Shipped chain = 4-member UT pixel mean -> energy restore (lam, k=4) -> median lens field
                (production apply_field, INTER_LANCZOS4) -> JPEG q100/ss2/optimize/progressive.

ARMS  (all ship-able with zero GPU and zero retraining)
  A0_float      float end to end; one final uint8 cast before JPEG   <- a FUSED build script
  A1_qens       + uint8 PNG round-trip after the ensemble mean
  A2_qens_qer   + uint8 PNG round-trip after energy restore          <- PRODUCTION tower chain
  Q99           A0 pixels at quality=99                              <- HCM0421's shipped profile
  SS0           A0 pixels at subsampling=0     (known control, recorded +0.0094)
  ORDER         warp FIRST, then energy restore on warped ens + warped members
                (the lanczos warp throws away 2.4% of the Laplacian energy the restore just put
                 back; doing the restore last means the boost survives)

NULL CONTROL (asserted, not scored): a uint8 round after the warp is idempotent with the cast the
JPEG encoder needs, so it must be bit-free -- if the harness reported a delta there it would be
measuring noise.

SHARED COMPUTATION HOISTED: the members' finest Laplacian band is identical across arms and across
both unwarped restore calls, so it is computed once per image; likewise the warped members' band.
"""
import argparse, io, os, sys, time, json
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track")
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fit_field import apply_field

Image.MAX_IMAGE_PIXELS = None
SHIPPED = dict(quality=100, subsampling=2, optimize=True, progressive=True)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
FIELD = os.path.join(HERE, "HCM0181_median.npy")


def restore_pre(ens, mem_L0, lam, k, win=3, nlev=5, clamp=4.0):
    K = _K.to(ens.device)
    laps, res, sizes = lap_pyr(ens, nlev, K)
    L0 = laps[0]
    Eb = boxf((L0 ** 2).sum(1, keepdim=True), win)
    V = 0.0
    for ml in mem_L0:
        V = V + boxf(((ml - L0) ** 2).sum(1, keepdim=True), win)
    V = V / len(mem_L0) * (k / (k - 1.0))
    r = torch.sqrt(1.0 + V / (Eb + 1e-10)).clamp(max=clamp)
    out = [L0 * (1.0 + lam * (r - 1.0))] + laps[1:]
    return lap_recon(out, res, sizes, K)


def q8(x):
    return np.clip(np.clip(x, 0, 1) * 255.0 + 0.5, 0, 255).astype(np.uint8).astype(np.float32) / 255.0


def enc_dec(x, **kw):
    a = (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", **kw)
    buf = b.getvalue()
    return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"), dtype=np.float32) / 255.0, len(buf)


def t2n(t):
    return np.ascontiguousarray(t[0].permute(1, 2, 0).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "fb_full.json"))
    args = ap.parse_args()

    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    dev = args.device if torch.cuda.is_available() else "cpu"
    vgg = lpips_pkg.LPIPS(net="vgg").to(dev).eval()

    field = np.load(FIELD)
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s + ".png")) for d in MEM))
    if args.n:
        stems = stems[:args.n]

    ARMS = ["A0_float", "A1_qens", "A2_qens_qer", "Q99", "SS0", "ORDER"]
    PROF = {a: SHIPPED for a in ARMS}
    PROF["Q99"] = dict(SHIPPED, quality=99)
    PROF["SS0"] = dict(SHIPPED, subsampling=0)
    acc = {a: [0.0, 0.0, 0.0] for a in ARMS}
    per = {a: [] for a in ARMS}
    nb = {a: 0 for a in ARMS}
    t0 = time.time()
    null_ok = True
    K = _K.to(dev)
    for c, s in enumerate(stems):
        mem = []
        for d in MEM:
            a = np.asarray(Image.open(os.path.join(d, s + ".png")).convert("RGB"),
                           dtype=np.float32) / 255.0
            mem.append(torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev))
        with torch.no_grad():
            mem_L0 = [lap_pyr(m, 5, K)[0][0] for m in mem]                  # HOISTED
            ens_f = torch.stack(mem).mean(0)

            er_f = restore_pre(ens_f, mem_L0, args.lam, len(mem)).clamp(0, 1)
            w_f = np.clip(apply_field(t2n(er_f), field), 0, 1)

            ens_q = torch.from_numpy(q8(t2n(ens_f))).permute(2, 0, 1).unsqueeze(0).to(dev)
            er_q = restore_pre(ens_q, mem_L0, args.lam, len(mem)).clamp(0, 1)
            w_q1 = np.clip(apply_field(t2n(er_q), field), 0, 1)
            w_q2 = np.clip(apply_field(q8(t2n(er_q)), field), 0, 1)

            # ORDER: warp everything first, restore last
            wm = []
            for m in mem:
                a = np.clip(apply_field(t2n(m), field), 0, 1)
                wm.append(torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev))
            wm_L0 = [lap_pyr(m, 5, K)[0][0] for m in wm]
            w_ens = torch.stack(wm).mean(0)
            w_ord = t2n(restore_pre(w_ens, wm_L0, args.lam, len(wm)).clamp(0, 1))

            if c < 3:
                a2 = (np.clip(w_q2, 0, 1) * 255 + 0.5).astype(np.uint8)
                a3 = (np.clip(q8(w_q2), 0, 1) * 255 + 0.5).astype(np.uint8)
                null_ok &= bool(np.array_equal(a2, a3))

            outs = {"A0_float": w_f, "A1_qens": w_q1, "A2_qens_qer": w_q2,
                    "Q99": w_f, "SS0": w_f, "ORDER": w_ord}
            g = torch.from_numpy(np.asarray(Image.open(os.path.join(GTD, gt_by[s])).convert("RGB"),
                                            dtype=np.float32) / 255.0
                                 ).permute(2, 0, 1).unsqueeze(0).to(dev)
            for a in ARMS:
                j, nbytes = enc_dec(outs[a], **PROF[a])
                nb[a] += nbytes
                r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(dev)
                P = 10 * np.log10(1.0 / max(((r - g) ** 2).mean().item(), 1e-12))
                S = float(repo_ssim(r, g))
                L = float(vgg(r * 2 - 1, g * 2 - 1).item())
                acc[a][0] += P; acc[a][1] += S; acc[a][2] += L
                per[a].append(100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0)))
        del mem, mem_L0, wm, wm_L0
        if c % 10 == 0:
            print(f"  {c}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)

    n = len(stems)
    print(f"\nHCM0181 n={n} lam={args.lam}  FULL SHIPPED CHAIN")
    print(f"{'arm':>13} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs A0':>9} "
          f"{'paired t':>9} {'MB/60':>8}")
    res, base, basev = {}, None, None
    for a in ARMS:
        P, S, L = (x / n for x in acc[a])
        sc = 100 * (0.4 * (1 - L) + 0.3 * S + 0.3 * min(P / 50.0, 1.0))
        v = np.asarray(per[a])
        if a == "A0_float":
            base, basev = sc, v
        d = v - basev
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n)) if d.std() > 0 else 0.0
        res[a] = dict(score=sc, psnr=P, ssim=S, lpips=L, mb60=nb[a] / n * 60 / 1e6,
                      delta=sc - base, t=float(t))
        print(f"{a:>13} {sc:9.4f} {P:8.4f} {S:8.4f} {L:8.4f} {sc-base:+9.4f} {t:9.2f} "
              f"{nb[a]/n*60/1e6:8.2f}")
    print(f"\nNULL CONTROL (post-warp uint8 round idempotent): {'PASS' if null_ok else 'FAIL'}")
    json.dump(dict(n=n, lam=args.lam, res=res, null_ok=null_ok), open(args.out, "w"), indent=2)
    print(f"wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
