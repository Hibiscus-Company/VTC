import os, numpy as np, torch, lpips
from PIL import Image
from math import log10
def ssim_fn(a,b,channel_axis=None,data_range=1.0):
    import numpy as np
    from scipy.ndimage import uniform_filter
    C1=(0.01*data_range)**2; C2=(0.03*data_range)**2
    out=[]
    for c in range(a.shape[2]):
        x=a[:,:,c].astype(np.float64); y=b[:,:,c].astype(np.float64)
        ux=uniform_filter(x,7); uy=uniform_filter(y,7)
        uxx=uniform_filter(x*x,7); uyy=uniform_filter(y*y,7); uxy=uniform_filter(x*y,7)
        vx=uxx-ux*ux; vy=uyy-uy*uy; vxy=uxy-ux*uy
        n=49.0/48.0
        vx*=n; vy*=n; vxy*=n
        S=((2*ux*uy+C1)*(2*vxy+C2))/((ux*ux+uy*uy+C1)*(vx+vy+C2))
        out.append(S[3:-3,3:-3].mean())
    return float(np.mean(out))
dev="cuda:0" if torch.cuda.is_available() else "cpu"
ln=lpips.LPIPS(net='vgg').to(dev)
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"
arms={"depth_prior_0.05":"/mnt/d/avv/depth/bonsai_dp05/eval_render",
      "baseline_K4_pC_seed7":"/mnt/d/avv/bonsai_eval/K4_pC_seed7/eval_render",
      "baseline_K1_noUT_aa":"/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_render"}
for nm,rd in arms.items():
    if not os.path.isdir(rd): print(nm,"MISSING",rd); continue
    P=[];S=[];L=[]
    for fn in sorted(os.listdir(GT)):
        st=os.path.splitext(fn)[0]
        rp=None
        for e in (".jpg",".png",".JPG"):
            p=os.path.join(rd,st+e)
            if os.path.exists(p): rp=p;break
        if rp is None: continue
        g=np.asarray(Image.open(os.path.join(GT,fn)).convert("RGB"),np.float32)/255.
        r=np.asarray(Image.open(rp).convert("RGB"),np.float32)/255.
        if g.shape!=r.shape: continue
        P.append(10*log10(1.0/max(((g-r)**2).mean(),1e-12)))
        S.append(ssim_fn(g,r,channel_axis=2,data_range=1.0))
        tg=torch.from_numpy(g).permute(2,0,1)[None].to(dev)*2-1
        tr=torch.from_numpy(r).permute(2,0,1)[None].to(dev)*2-1
        with torch.no_grad(): L.append(ln(tg,tr).item())
    p,s,l=np.mean(P),np.mean(S),np.mean(L)
    sc=100*(0.4*(1-l)+0.3*s+0.3*p/50)
    print("%-22s n=%d PSNR %.4f SSIM %.4f LPIPS %.4f SCORE %.4f"%(nm,len(P),p,s,l,sc))
