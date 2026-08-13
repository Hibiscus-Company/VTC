import os, io, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"; ARMD="/mnt/d/avv/bonsai_eval"
GOOD=['K1_noUT_aa','K4_pC_seed1k','K4_pC_seed7','eps10','eps20','ppisp_pc']
files=sorted(os.listdir(GT))
H,W=1080,1920
E=np.zeros((H,W)); G=np.zeros((H,W)); HF=np.zeros((H,W)); n=0
for fn in files:
    st=os.path.splitext(fn)[0]
    g=np.asarray(Image.open(os.path.join(GT,fn)).convert("RGB"),np.float32)
    r=np.mean([np.asarray(Image.open(os.path.join(ARMD,a,"eval_render",st+".jpg")).convert("RGB"),np.float32) for a in GOOD],0)
    E+=np.abs(r-g).mean(2); G+=g.mean(2)
    gg=g.mean(2); HF+=np.abs(np.diff(gg,axis=0,prepend=gg[:1]))+np.abs(np.diff(gg,axis=1,prepend=gg[:,:1])); n+=1
E/=n; G/=n; HF/=n
# fine grid 24x32
gy,gx=24,32
def cz(a):
    ys=np.linspace(0,H,gy+1).astype(int); xs=np.linspace(0,W,gx+1).astype(int)
    return np.array([[a[ys[i]:ys[i+1],xs[j]:xs[j+1]].mean() for j in range(gx)] for i in range(gy)])
e,h,l=cz(E),cz(HF),cz(G)
b,a0=np.polyfit(h.ravel(),e.ravel(),1); res=e-(a0+b*h)
print("fine grid %dx%d  fit err=%.3f+%.3f*hf  R2=%.3f"%(gy,gx,a0,b,np.corrcoef((a0+b*h).ravel(),e.ravel())[0,1]**2))
rows_bottom=np.arange(int(gy*5/8),gy)
print("\nrow-band residual profile (each band = 1/8 frame height):")
for k in range(8):
    r0,r1=int(gy*k/8),int(gy*(k+1)/8)
    print("  band %d y=%4d-%4d  err=%.2f hf=%.2f lum=%.0f  resid=%+.3f (%+.1f%%)"%(k,int(H*k/8),int(H*(k+1)/8),
      e[r0:r1].mean(),h[r0:r1].mean(),l[r0:r1].mean(),res[r0:r1].mean(),100*res[r0:r1].mean()/e[r0:r1].mean()))
print("\nWITHIN bottom 3/8 (n=%d cells): resid vs GT luminance"%res[rows_bottom].size)
rb=res[rows_bottom].ravel(); lb=l[rows_bottom].ravel(); hb=h[rows_bottom].ravel()
print("  corr(resid, lum)=%+.3f   corr(resid, hf)=%+.3f"%(np.corrcoef(rb,lb)[0,1],np.corrcoef(rb,hb)[0,1]))
q=np.argsort(lb)
for name,idx in (("darkest third (glass/specular)",q[:len(q)//3]),("middle third",q[len(q)//3:2*len(q)//3]),("brightest third",q[2*len(q)//3:])):
    print("  %-32s lum=%5.1f  resid=%+.3f  err=%.2f"%(name,lb[idx].mean(),rb[idx].mean(),e[rows_bottom].ravel()[idx].mean()))
# column profile of residual in the bottom band
print("\nbottom-band residual by column octile:")
cb=res[rows_bottom]
for k in range(8):
    c0,c1=int(gx*k/8),int(gx*(k+1)/8)
    print("   x=%4d-%4d resid=%+.3f"%(int(W*k/8),int(W*(k+1)/8),cb[:,c0:c1].mean()))
# dump visuals
def png(a,path,lo=None,hi=None):
    lo=a.min() if lo is None else lo; hi=a.max() if hi is None else hi
    Image.fromarray(np.clip((a-lo)/(hi-lo)*255,0,255).astype(np.uint8)).resize((640,360)).save(path)
png(G,"/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADV3_gt.png",0,255)
png(gaussian_filter(E,12),"/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADV3_err.png",0,25)
RESpix=np.kron(res,np.ones((H//gy,W//gx)))
png(RESpix,"/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADV3_res.png",-3,3)
print("\nsaved ADV3_gt.png ADV3_err.png ADV3_res.png")
