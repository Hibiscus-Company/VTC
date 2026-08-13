#!/usr/bin/env python
"""AUDIT (numeric): uint8 double-rounding + quantisation bias in the disagreement map.
Production harness HCM0181, real test GT, FULL shipped chain.
Shared work (member Laplacians, field maps) is hoisted out of the arm loop.
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
cv2.setNumThreads(2); torch.set_num_threads(2)
JPG = dict(quality=100, subsampling=2, optimize=True, progressive=True)
Q8 = (1.0/255.0)**2/12.0
G0 = 0.888668
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
DEV = "cuda"


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEV)


def r8(t):
    return (t.clamp(0, 1)*255.0 + 0.5).floor().clamp(0, 255)/255.0   # exact uint8 write+read


def restore(pyr, memL0, lam, k, K, debias=0.0):
    laps, res, sizes = pyr
    L0 = laps[0]
    Eb = boxf((L0**2).sum(1, keepdim=True), 3)
    V = 0.0
    for ml in memL0:
        V = V + boxf(((ml - L0)**2).sum(1, keepdim=True), 3)
    V = V/len(memL0)*(k/(k-1.0))
    if debias:
        V = (V - debias).clamp_min(0.0); Eb = (Eb - debias/k).clamp_min(0.0)
    r = torch.sqrt(1.0 + V/(Eb + 1e-10)).clamp(max=4.0)
    return lap_recon([L0*(1.0 + lam*(r - 1.0))] + laps[1:], res, sizes, K), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    K = _K.to(DEV)
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by
                   if all(os.path.exists(os.path.join(d, s+".png")) for d in MEM))[:args.n]
    z = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz")
    H, W = [int(x) for x in z["HW"]]
    lens = upsample(LooPool(z["s8"]).pooled("median"), H, W, "cubic")
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    MX = (xx + lens[..., 0]).astype(np.float32); MY = (yy + lens[..., 1]).astype(np.float32)
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lpips_pkg
    vgg = lpips_pkg.LPIPS(net="vgg", verbose=False).to(DEV).eval()
    DEB = 3.0*Q8*G0
    arms = ["prod", "fuse34", "float1", "deb", "float1_deb"]
    per = {a: [] for a in arms}; nby = {a: 0 for a in arms}
    dg = dict(hi=0.0, lo=0.0, npx=0, rp=[], rd=[], frac_q=[])
    t0 = time.time()
    for c, s in enumerate(stems):
        mem = [load(os.path.join(d, s+".png")) for d in MEM]
        memL0 = [lap_pyr(m, 5, K)[0][0] for m in mem]           # HOISTED
        g = load(os.path.join(GTD, gt_by[s]))
        e_f = 0.8*(mem[0]+mem[1]+mem[2])/3.0 + 0.2*mem[3]
        e_p = r8(r8((mem[0]+mem[1]+mem[2])/3.0)*0.8 + 0.2*mem[3])
        pf = lap_pyr(e_f, 5, K); pp = lap_pyr(e_p, 5, K)
        outs = {}
        o, rp = restore(pp, memL0, args.lam, 4, K);        outs["prod"] = r8(o.clamp(0, 1))
        outs["fuse34"] = o.clamp(0, 1)
        o2, _ = restore(pf, memL0, args.lam, 4, K);        outs["float1"] = o2.clamp(0, 1)
        o3, rd = restore(pp, memL0, args.lam, 4, K, DEB);  outs["deb"] = r8(o3.clamp(0, 1))
        o4, _ = restore(pf, memL0, args.lam, 4, K, DEB);   outs["float1_deb"] = o4.clamp(0, 1)
        dg["hi"] += float((o > 1.0).sum()); dg["lo"] += float((o < 0.0).sum()); dg["npx"] += o.numel()
        dg["rp"].append(float(rp.mean())); dg["rd"].append(float(rd.mean()))
        Eb = boxf((pp[0][0]**2).sum(1, keepdim=True), 3)
        Vv = sum(boxf(((ml-pp[0][0])**2).sum(1, keepdim=True), 3) for ml in memL0)/4*(4/3.)
        dg["frac_q"].append(float((DEB/(Vv+1e-12)).clamp(max=1).mean()))
        for a in arms:
            x = outs[a][0].permute(1, 2, 0).cpu().numpy()
            x = np.clip(cv2.remap(np.ascontiguousarray(x), MX, MY, cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REFLECT), 0, 1)
            u8 = (x*255.0 + 0.5).astype(np.uint8)
            b = io.BytesIO(); Image.fromarray(u8).save(b, "JPEG", **JPG); nby[a] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.float32)/255.0
            r = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(DEV)
            with torch.no_grad():
                P = 10*np.log10(1.0/max(float(((r-g)**2).mean()), 1e-12))
                S = float(repo_ssim(r, g)); L = float(vgg(r*2-1, g*2-1).item())
            per[a].append((P, S, L, 100*(0.4*(1-L)+0.3*S+0.3*min(P/50.0, 1.0))))
        del mem, memL0, outs, pf, pp
        if c % 5 == 0:
            print(f"  {c+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    n = len(stems)
    A = {a: np.array(per[a]) for a in arms}
    print(f"\nHCM0181 FULL SHIPPED CHAIN (2-stage weighted ens -> energy restore lam={args.lam}"
          f" -> median field lanczos4 -> JPEG q100/ss2), n={n}")
    print(f"{'arm':>12} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs prod':>9} {'se':>7} {'MB':>7}")
    base = A["prod"][:, 3]; res = {}
    for a in arms:
        m = A[a].mean(0); d = A[a][:, 3]-base
        se = d.std(ddof=1)/np.sqrt(n) if a != "prod" else 0.0
        res[a] = dict(score=m[3], psnr=m[0], ssim=m[1], lpips=m[2], d=float(d.mean()),
                      se=float(se), mb=nby[a]/1e6, win=int((d > 0).sum()))
        print(f"{a:>12} {m[3]:9.4f} {m[0]:8.4f} {m[1]:8.5f} {m[2]:8.5f} {d.mean():+9.4f} "
              f"{se:7.4f} {nby[a]/1e6:7.2f}   wins {int((d>0).sum())}/{n}")
    print(f"\nrestore clip >1 {100*dg['hi']/dg['npx']:.4f}%  <0 {100*dg['lo']/dg['npx']:.4f}%")
    print(f"mean r prod {np.mean(dg['rp']):.4f}  debiased {np.mean(dg['rd']):.4f}")
    print(f"mean fraction of V that is uint8 quantisation noise: {np.mean(dg['frac_q']):.4f}")
    if args.out:
        json.dump(dict(res=res, n=n, clip_hi=dg['hi']/dg['npx'], clip_lo=dg['lo']/dg['npx'],
                       r_prod=float(np.mean(dg['rp'])), r_deb=float(np.mean(dg['rd'])),
                       frac_q=float(np.mean(dg['frac_q']))), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
