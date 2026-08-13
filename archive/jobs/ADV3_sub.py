import os, sys, io, numpy as np, torch, lpips
from PIL import Image
from scipy.ndimage import gaussian_filter

def ssim_map(g, r):
    C1=(0.01*255)**2; C2=(0.03*255)**2; f=lambda x: gaussian_filter(x,1.5)
    out=[]
    for c in range(3):
        x=g[:,:,c].astype(np.float64); y=r[:,:,c].astype(np.float64)
        mx=f(x); my=f(y); mxx=f(x*x)-mx*mx; myy=f(y*y)-my*my; mxy=f(x*y)-mx*my
        out.append(((2*mx*my+C1)*(2*mxy+C2))/((mx*mx+my*my+C1)*(mxx+myy+C2)))
    return np.mean(out,0)


GY, GX = 8, 12
dev = "cuda"
LP = lpips.LPIPS(net="vgg").to(dev).eval()

def cellize(a):
    H, W = a.shape
    ys = np.linspace(0, H, GY+1).astype(int); xs = np.linspace(0, W, GX+1).astype(int)
    return np.array([[a[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean() for j in range(GX)] for i in range(GY)])

GT = "/mnt/d/avv/evalsplit/bonsai/eval_gt"
ARMD = "/mnt/d/avv/bonsai_eval"
arms = sorted(os.listdir(ARMD))
files = sorted(os.listdir(GT))

# --- per-arm mean error ---
per_arm = {a: [] for a in arms}
for fn in files[:8]:
    st = os.path.splitext(fn)[0]
    g = np.asarray(Image.open(os.path.join(GT, fn)).convert("RGB"), np.float32)
    for a in arms:
        p = os.path.join(ARMD, a, "eval_render", st + ".jpg")
        if os.path.exists(p):
            r = np.asarray(Image.open(p).convert("RGB"), np.float32)
            per_arm[a].append(np.abs(r-g).mean() if r.shape == g.shape else np.nan)
print("=== per-arm mean |err| (0-255), 8 eval views ===")
for a in arms: print("  %-16s %.3f  shape_ok=%s" % (a, np.nanmean(per_arm[a]), not np.isnan(per_arm[a]).any()))

GOOD = [a for a in arms if np.nanmean(per_arm[a]) < 1.25*min(np.nanmean(per_arm[x]) for x in arms)]
print("homogeneous pool:", GOOD)

def region(m, rows):
    return m[np.array(rows)].mean()

RG = [("top 0-1 bg", [0,1]), ("row2", [2]), ("mid 3-4 plant", [3,4]), ("bottom 5-7 GLASS", [5,6,7])]

for cond in ("single_K4_pC_seed7", "ens_%d" % len(GOOD)):
    A = np.zeros((GY,GX)); LO = np.zeros((GY,GX)); HI = np.zeros((GY,GX))
    LPm = np.zeros((GY,GX)); SS = np.zeros((GY,GX)); HFm = np.zeros((GY,GX)); n=0
    for fn in files:
        st = os.path.splitext(fn)[0]
        g = np.asarray(Image.open(os.path.join(GT, fn)).convert("RGB"), np.float32)
        if cond.startswith("single"):
            r = np.asarray(Image.open(os.path.join(ARMD,"K4_pC_seed7","eval_render",st+".jpg")).convert("RGB"), np.float32)
        else:
            rs = [np.asarray(Image.open(os.path.join(ARMD,a,"eval_render",st+".jpg")).convert("RGB"), np.float32) for a in GOOD]
            r = np.mean(rs, 0)
        # shipped-chain tail: JPEG q100 ss2 round-trip
        buf = io.BytesIO(); Image.fromarray(np.clip(r+0.5,0,255).astype(np.uint8)).save(buf, "JPEG", quality=100, subsampling=2, optimize=True, progressive=True)
        r = np.asarray(Image.open(buf).convert("RGB"), np.float32)
        d = (r-g).mean(2)
        dl = gaussian_filter(d, 8.0)
        A += cellize(np.abs(d)); LO += cellize(np.abs(dl)); HI += cellize(np.abs(d-dl))
        gg = g.mean(2); HFm += cellize(np.abs(np.diff(gg,axis=0,prepend=gg[:1]))+np.abs(np.diff(gg,axis=1,prepend=gg[:,:1])))
        with torch.no_grad():
            t = lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))[None]).to(dev).float()/127.5-1.0
            LP.spatial = True
            sm = LP(t(g), t(r)).squeeze().cpu().numpy()
            LP.spatial = False
        LPm += cellize(sm)
        SS += cellize(1.0-ssim_map(g, r))
        n += 1
    A/=n; LO/=n; HI/=n; LPm/=n; SS/=n; HFm/=n
    print("\n===== %s  n=%d  (after JPEG q100/ss2) =====" % (cond, n))
    print(" region                 area%   L1     LFerr  HFerr  L1share LPIPSshare (1-SSIM)share  LPIPSdens/hf")
    for nm, rows in RG:
        af = len(rows)/GY
        print("  %-20s %5.1f  %6.3f %6.3f %6.3f  %6.1f%%  %6.1f%%      %6.1f%%      %.4f" %
              (nm, 100*af, region(A,rows), region(LO,rows), region(HI,rows),
               100*af*region(A,rows)/A.mean(), 100*af*region(LPm,rows)/LPm.mean(),
               100*af*region(SS,rows)/SS.mean(), region(LPm,rows)/region(HFm,rows)))
    for tag, M in (("L1", A), ("LPIPS", LPm), ("1-SSIM", SS)):
        h = HFm.ravel(); y = M.ravel(); b,a0 = np.polyfit(h,y,1); res=(y-(a0+b*h)).reshape(GY,GX)
        s = "  fit %-7s = %.4g + %.4g*hf  |  resid%%: " % (tag, a0, b)
        for nm, rows in RG: s += "%s %+.1f%%  " % (nm.split()[0], 100*region(res,rows)/region(M,rows))
        print(s)
    np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADV3_%s.npy" % cond, dict(A=A,LO=LO,HI=HI,LP=LPm,SS=SS,HF=HFm), allow_pickle=True)
