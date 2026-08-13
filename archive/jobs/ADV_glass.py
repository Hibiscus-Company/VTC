import os, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"
ARM="/mnt/d/avv/bonsai_eval/K4_pC_seed7/eval_render"
PNG="/mnt/d/avv/bonsai_eval/K4_pC_seed7/eval_png"
GY,GX=8,12
def cells(a,ys,xs):
    return np.array([[a[ys[i]:ys[i+1],xs[j]:xs[j+1]].mean() for j in range(GX)] for i in range(GY)])
E=[];Hf=[];L=[];Elf=[];Ehf=[];Rtv=[];Mse=[];names=[]
Ejpg=[];Epng=[]
for fn in sorted(os.listdir(GT)):
    st=os.path.splitext(fn)[0]
    rp=os.path.join(ARM,st+".jpg")
    if not os.path.exists(rp): continue
    g=np.asarray(Image.open(os.path.join(GT,fn)).convert("RGB"),np.float32)
    r=np.asarray(Image.open(rp).convert("RGB"),np.float32)
    H,W,_=g.shape
    ys=np.linspace(0,H,GY+1).astype(int); xs=np.linspace(0,W,GX+1).astype(int)
    d=g-r
    e=np.abs(d).mean(2)
    dl=gaussian_filter(d,(8,8,0))
    elf=np.abs(dl).mean(2); ehf=np.abs(d-dl).mean(2)
    gg=g.mean(2); rr=r.mean(2)
    gh=np.abs(np.diff(gg,axis=0,prepend=gg[:1]))+np.abs(np.diff(gg,axis=1,prepend=gg[:,:1]))
    rh=np.abs(np.diff(rr,axis=0,prepend=rr[:1]))+np.abs(np.diff(rr,axis=1,prepend=rr[:,:1]))
    E.append(cells(e,ys,xs)); Hf.append(cells(gh,ys,xs)); L.append(cells(gg,ys,xs))
    Elf.append(cells(elf,ys,xs)); Ehf.append(cells(ehf,ys,xs)); Rtv.append(cells(rh,ys,xs))
    Mse.append(cells((d**2).mean(2),ys,xs)); names.append(st)
    pp=os.path.join(PNG,st+".png")
    if os.path.exists(pp):
        rp2=np.asarray(Image.open(pp).convert("RGB"),np.float32)
        Ejpg.append(np.abs(g-r).mean()); Epng.append(np.abs(g-rp2).mean())
E=np.array(E);Hf=np.array(Hf);L=np.array(L);Elf=np.array(Elf);Ehf=np.array(Ehf);Rtv=np.array(Rtv);Mse=np.array(Mse)
n=len(E); print("n images",n)
np.set_printoptions(precision=2,suppress=True,linewidth=250)
print("\nGT mean luminance per cell (0-255):"); print(L.mean(0))
print("\nGT TV per cell:"); print(Hf.mean(0))
print("\nRENDER TV per cell:"); print(Rtv.mean(0))
print("\nrender/GT TV ratio:"); print(Rtv.mean(0)/Hf.mean(0))
print("\nerr LOW-freq part (sigma8):"); print(Elf.mean(0))
print("\nerr HIGH-freq part:"); print(Ehf.mean(0))
bands={"top01":[0,1],"mid34":[3,4],"bot567":[5,6,7],"row5":[5],"row6":[6],"row7":[7],"row2":[2]}
print("\n%-8s %6s %6s %6s %6s %6s %6s %6s"%("band","err","hf","ratio","elf","ehf","lum","rtv/gtv"))
for k,rows in bands.items():
    e=E[:,rows].mean(); h=Hf[:,rows].mean()
    print("%-8s %6.3f %6.3f %6.3f %6.3f %6.3f %6.1f %6.3f"%(k,e,h,e/h,Elf[:,rows].mean(),Ehf[:,rows].mean(),L[:,rows].mean(),Rtv[:,rows].mean()/h))
# per-image bootstrap of ratio difference bot - mid
rb=np.array([E[i,[5,6,7]].mean()/Hf[i,[5,6,7]].mean() for i in range(n)])
rm=np.array([E[i,[3,4]].mean()/Hf[i,[3,4]].mean() for i in range(n)])
d=rb-rm
print("\nper-image ratio bot %.3f+-%.3f  mid %.3f+-%.3f  diff %.4f  sd %.4f  se %.4f  t=%.2f"%(
 rb.mean(),rb.std(ddof=1),rm.mean(),rm.std(ddof=1),d.mean(),d.std(ddof=1),d.std(ddof=1)/np.sqrt(n),d.mean()/(d.std(ddof=1)/np.sqrt(n))))
# linear model err = a + b*hf on 96 cell means
a=np.polyfit(Hf.mean(0).ravel(),E.mean(0).ravel(),1)
print("\nfit err = %.4f*hf + %.4f  (R2=%.3f)"%(a[0],a[1],np.corrcoef(Hf.mean(0).ravel(),E.mean(0).ravel())[0,1]**2))
res=E.mean(0)-np.polyval(a,Hf.mean(0))
print("residual of linear fit per band:", {k:float(res[rows].mean()) for k,rows in bands.items()})
# LB arithmetic: shrink bot567 error to ratio 1.158
mse_tot=Mse.mean(); mse_bot=Mse[:,[5,6,7]].mean()
s=(1.158/1.264)**2
new=(mse_tot*8 - mse_bot*3 + mse_bot*3*s)/8
print("\nMSE tot %.2f bot %.2f  PSNR now %.4f  after %.4f  dPSNR %.4f dB"%(
 mse_tot,mse_bot,-10*np.log10(mse_tot/255**2),-10*np.log10(new/255**2),-10*np.log10(new/255**2)+10*np.log10(mse_tot/255**2)))
if Ejpg: print("\njpg vs png render MAE: %.4f vs %.4f  (n=%d)"%(np.mean(Ejpg),np.mean(Epng),len(Ejpg)))
