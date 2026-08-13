#!/usr/bin/env python
"""AUDIT: HCM0421 still ships at JPEG quality=99 (a leftover from when the 350MB cap was
read as decimal). Everything else ships q100. Measure q99 vs q100 through the full shipped
chain on the production harness."""
import os, sys, io, time, json, argparse
import numpy as np, torch, cv2
from PIL import Image
HERE = "/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "lens"))
from lapfuse import lap_pyr, lap_recon, boxf, _K
from fieldlib import LooPool, upsample
Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(2); torch.set_num_threads(2)
MEM = ["/mnt/d/avv/output/HCM0181_gsplatB9ut/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB10ut8M/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB11ut60k/test_poses_renders_png",
       "/mnt/d/avv/output/HCM0181_gsplatB12ut8Ms7/test_poses_renders_png"]
GTD = "/mnt/d/avv/data/phase1/public_set/HCM0181/test/images"
DEV = "cuda"
ARMS = {"q100": dict(quality=100, subsampling=2, optimize=True, progressive=True),
        "q99":  dict(quality=99,  subsampling=2, optimize=True, progressive=True),
        "q98":  dict(quality=98,  subsampling=2, optimize=True, progressive=True)}


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEV)


def r8(t): return (t.clamp(0, 1)*255.0 + 0.5).floor().clamp(0, 255)/255.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default=None); a = ap.parse_args()
    K = _K.to(DEV)
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(GTD)}
    stems = sorted(s for s in gt_by if all(os.path.exists(os.path.join(d, s+".png")) for d in MEM))[:a.n]
    z = np.load(f"{HERE}/lens/cache/pub_HCM0181.npz"); H, W = [int(x) for x in z["HW"]]
    lens = upsample(LooPool(z["s8"]).pooled("median"), H, W, "cubic")
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    MX = (xx+lens[..., 0]).astype(np.float32); MY = (yy+lens[..., 1]).astype(np.float32)
    from utils.loss_utils import ssim as repo_ssim
    import lpips as lp
    vgg = lp.LPIPS(net="vgg", verbose=False).to(DEV).eval()
    per = {k: [] for k in ARMS}; nby = {k: 0 for k in ARMS}; t0 = time.time()
    for c, s in enumerate(stems):
        mem = [load(os.path.join(d, s+".png")) for d in MEM]
        memL0 = [lap_pyr(m, 5, K)[0][0] for m in mem]
        g = load(os.path.join(GTD, gt_by[s]))
        e = r8(r8((mem[0]+mem[1]+mem[2])/3.0)*0.8 + 0.2*mem[3])
        laps, res, sizes = lap_pyr(e, 5, K); L0 = laps[0]
        Eb = boxf((L0**2).sum(1, keepdim=True), 3)
        V = sum(boxf(((m-L0)**2).sum(1, keepdim=True), 3) for m in memL0)/4*(4/3.)
        r = torch.sqrt(1.0+V/(Eb+1e-10)).clamp(max=4.0)
        o = r8(lap_recon([L0*r]+laps[1:], res, sizes, K).clamp(0, 1))
        x = o[0].permute(1, 2, 0).cpu().numpy()
        x = np.clip(cv2.remap(np.ascontiguousarray(x), MX, MY, cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REFLECT), 0, 1)
        u8 = (x*255.0+0.5).astype(np.uint8)
        for k, kw in ARMS.items():
            b = io.BytesIO(); Image.fromarray(u8).save(b, "JPEG", **kw); nby[k] += len(b.getvalue())
            j = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.float32)/255.
            t = torch.from_numpy(np.ascontiguousarray(j)).permute(2, 0, 1).unsqueeze(0).to(DEV)
            with torch.no_grad():
                P = 10*np.log10(1.0/max(float(((t-g)**2).mean()), 1e-12))
                S = float(repo_ssim(t, g)); L = float(vgg(t*2-1, g*2-1).item())
            per[k].append((P, S, L, 100*(0.4*(1-L)+0.3*S+0.3*min(P/50., 1.))))
        if c % 5 == 0: print(f"  {c+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    n = len(stems); A = {k: np.array(per[k]) for k in ARMS}
    base = A["q100"][:, 3]; res_j = {}
    print(f"\nJPEG quality through the shipped chain, HCM0181, n={n}")
    print(f"{'arm':>6} {'SCORE':>9} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'vs q100':>9} {'se':>7} {'MB/60im':>8}")
    for k in ARMS:
        m = A[k].mean(0); d = A[k][:, 3]-base
        se = d.std(ddof=1)/np.sqrt(n)
        res_j[k] = dict(score=m[3], psnr=m[0], ssim=m[1], lpips=m[2], d=float(d.mean()),
                        se=float(se), mb=nby[k]/1e6*60/n)
        print(f"{k:>6} {m[3]:9.4f} {m[0]:8.4f} {m[1]:8.5f} {m[2]:8.5f} {d.mean():+9.4f} {se:7.4f} "
              f"{nby[k]/1e6*60/n:8.2f}")
    if a.out: json.dump(dict(res=res_j, n=n), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
