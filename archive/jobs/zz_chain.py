#!/usr/bin/env python
"""AUDIT: numeric defects in the shipped post-render chain, production harness HCM0181.

Arms (all scored through the FULL shipped chain against real test GT):
  prod        : shipped chain exactly -- FOUR uint8 quantisations after the members
                (stage1 6-mean PNG, stage2 weighted-blend PNG, stage3 energy-restore PNG,
                 stage4 warped PNG) then JPEG q100/ss2
  float1      : identical arithmetic, but only the ONE uint8 round JPEG forces
  debias      : prod, with the analytic uint8-quantisation variance removed from the
                energy-restore disagreement map V (and from Eb)
  float1_deb  : both
"""
import os, sys, io, time, json, argparse
import numpy as np, torch, cv2
from PIL import Image
HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, upsample
Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(8)
torch.set_num_threads(16)
SHIPPED_JPEG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q8 = (1.0/255.0)**2/12.0
G0 = 0.888668          # white-noise gain of the finest Burt-Adelson Laplacian band

MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def r8(t):                      # float tensor -> uint8 round-trip (the production write)
    a = (t.clamp(0, 1).numpy()*255.0 + 0.5).astype(np.uint8)
    return torch.from_numpy(a.astype(np.float32)/255.0)


def restore(ens_l0pyr, memL0, lam, k, debias=0.0, clamp=4.0):
    laps, res, sizes = ens_l0pyr
    L0 = laps[0]
    Eb = boxf((L0**2).sum(1, keepdim=True), 3)
    V = 0.0
    for ml in memL0:
        V = V + boxf(((ml - L0)**2).sum(1, keepdim=True), 3)
    V = V/len(memL0) * (k/(k-1.0))
    if debias:
        V = (V - debias).clamp_min(0.0)
        Eb = (Eb - debias/k).clamp_min(0.0)
    r = torch.sqrt(1.0 + V/(Eb + 1e-10)).clamp(max=clamp)
    out = [L0*(1.0 + lam*(r - 1.0))] + laps[1:]
    return lap_recon(out, res, sizes, _K), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_lpips", action="store_true")
    args = ap.parse_args()

    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s+".png")) for d in MEM))
    stems = stems[args.start:args.start+args.n]
    cache = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
    H, W = [int(x) for x in cache["HW"]]
    lens = upsample(LooPool(cache["s8"]).pooled("median"), H, W, "cubic")
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    MX = (xx + lens[..., 0]).astype(np.float32); MY = (yy + lens[..., 1]).astype(np.float32)

    from utils.loss_utils import ssim as repo_ssim
    if not args.no_lpips:
        import lpips as lpips_pkg
        vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).eval()

    DEB = 3.0*Q8*G0
    arms = ["prod", "float1", "debias", "float1_deb"]
    acc = {a: np.zeros(3) for a in arms}; nby = {a: 0 for a in arms}
    diag = dict(clip_hi=0.0, clip_lo=0.0, npx=0, Eb_q=[], V_q=[], r_prod=[], r_deb=[])
    t0 = time.time()
    for c, s in enumerate(stems):
        mem = [load(os.path.join(d, s+".png")) for d in MEM]
        memL0 = [lap_pyr(m, 5, _K)[0][0] for m in mem]          # HOISTED: arm-independent
        g = load(os.path.join(GTD, gt_by[s]))
        # ---- the two ensemble constructions
        e_float = 0.8*(mem[0]+mem[1]+mem[2])/3.0 + 0.2*mem[3]
        e_prod = r8(r8((mem[0]+mem[1]+mem[2])/3.0)*0.8 + 0.2*mem[3])
        pyr_f = lap_pyr(e_float, 5, _K); pyr_p = lap_pyr(e_prod, 5, _K)
        outs = {}
        o, rP = restore(pyr_p, memL0, args.lam, 4, 0.0);        outs["prod"] = o
        o, _ = restore(pyr_f, memL0, args.lam, 4, 0.0);         outs["float1"] = o
        o, rD = restore(pyr_p, memL0, args.lam, 4, DEB);        outs["debias"] = o
        o, _ = restore(pyr_f, memL0, args.lam, 4, DEB);         outs["float1_deb"] = o
        # clipping + map diagnostics (prod arm)
        raw = outs["prod"]
        diag["clip_hi"] += float((raw > 1.0).sum()); diag["clip_lo"] += float((raw < 0.0).sum())
        diag["npx"] += raw.numel()
        Eb = boxf((pyr_p[0][0]**2).sum(1, keepdim=True), 3)
        diag["Eb_q"].append(np.quantile(Eb.numpy(), [0.01, 0.1, 0.5, 0.9]))
        diag["r_prod"].append(float(rP.mean())); diag["r_deb"].append(float(rD.mean()))
        for a in arms:
            x = outs[a].clamp(0, 1)[0].permute(1, 2, 0).numpy()
            if a.startswith("prod") or a == "debias":
                x = (np.clip(x, 0, 1)*255.0 + 0.5).astype(np.uint8).astype(np.float32)/255.0
            x = np.clip(cv2.remap(np.ascontiguousarray(x), MX, MY, cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REFLECT), 0, 1)
            u8 = (x*255.0 + 0.5).astype(np.uint8)
            if a.startswith("prod") or a == "debias":
                u8 = (np.clip(u8.astype(np.float32)/255.0, 0, 1)*255.0 + 0.5).astype(np.uint8)
            b = io.BytesIO(); Image.fromarray(u8).save(b, "JPEG", **SHIPPED_JPEG)
            nby[a] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"),
                           dtype=np.float32)/255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                acc[a][0] += 10*np.log10(1.0/max(float(((r-g)**2).mean()), 1e-12))
                acc[a][1] += float(repo_ssim(r, g))
                acc[a][2] += 0.0 if args.no_lpips else float(vgg(r*2-1, g*2-1).item())
        if c % 5 == 0:
            print(f"  {c+1}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)
    n = len(stems)
    print(f"\nHCM0181 full shipped chain, n={n}, lam={args.lam}"
          f"{'  (NO LPIPS)' if args.no_lpips else ''}")
    print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs prod':>9} {'MB':>7}")
    base = None; res = {}
    for a in arms:
        P, S, L = acc[a]/n
        sc = 100*(0.4*(1-L) + 0.3*S + 0.3*min(P/50.0, 1.0))
        if a == "prod": base = sc
        res[a] = dict(score=sc, psnr=P, ssim=S, lpips=L, mb=nby[a]/1e6, d=sc-base)
        print(f"{a:>12} {sc:9.4f} {P:8.4f} {S:8.5f} {L:8.5f} {sc-base:+9.4f} {nby[a]/1e6:7.2f}")
    Eq = np.mean(diag["Eb_q"], 0)
    print(f"\nclip>1 {100*diag['clip_hi']/diag['npx']:.4f}%  clip<0 "
          f"{100*diag['clip_lo']/diag['npx']:.4f}%")
    print(f"Eb quantiles (1/10/50/90%): {Eq}   analytic quant floor in V = {DEB:.4g}")
    print(f"mean r  prod {np.mean(diag['r_prod']):.4f}   debiased {np.mean(diag['r_deb']):.4f}")
    if args.out:
        json.dump(dict(res=res, n=n, clip_hi=diag['clip_hi']/diag['npx'],
                       clip_lo=diag['clip_lo']/diag['npx'], Eb_q=list(Eq),
                       deb=DEB, r_prod=float(np.mean(diag['r_prod'])),
                       r_deb=float(np.mean(diag['r_deb']))), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
