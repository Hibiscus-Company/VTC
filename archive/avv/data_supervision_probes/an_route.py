import os, sys, numpy as np, cv2, torch
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from PIL import Image
import lpips
from utils.loss_utils import ssim as fs
Image.MAX_IMAGE_PIXELS=None
dev='cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
LP=lpips.LPIPS(net='vgg').to(dev).eval()
def t(a): return torch.from_numpy(a).float().permute(2,0,1)[None].to(dev)/255.
def vl(p,ms=1024):
    im=Image.open(p).convert('L'); s=ms/max(im.size)
    if s<1: im=im.resize((int(im.size[0]*s),int(im.size[1]*s)),Image.LANCZOS)
    return float(cv2.Laplacian(np.asarray(im,dtype=np.float32),cv2.CV_32F).var())
R='/mnt/d/avv/data/phase1/private_set2'
for sc,root,es in [('chair','/mnt/d/avv/chair_eval','chair'),('bonsai','/mnt/d/avv/bonsai_eval','bonsai')]:
    gtd=f'/mnt/d/avv/evalsplit/{es}/eval_gt'
    mems=[m for m in sorted(os.listdir(root)) if os.path.isdir(f'{root}/{m}/eval_png')]
    gts=sorted(os.listdir(gtd))
    # predicted sharpness from TRAIN neighbours (legal: no eval pixels)
    imgs=read_extrinsics_binary(f'{R}/{sc}/train/sparse/0/images.bin')
    D={im.name:qvec2rotmat(im.qvec).T@np.array([0,0,1.]) for im in imgs.values()}
    tsub=sorted(os.listdir(f'/mnt/d/avv/evalsplit/{es}/train_sub/images'))
    tv=np.array([vl(f'/mnt/d/avv/evalsplit/{es}/train_sub/images/{n}') for n in tsub])
    tD=np.array([D[n] for n in tsub])
    S=np.full((len(gts),len(mems)),np.nan); pred=np.zeros(len(gts)); act=np.zeros(len(gts))
    Amean=[]
    for i,g in enumerate(gts):
        A=np.asarray(Image.open(f'{gtd}/{g}').convert('RGB'),dtype=np.uint8)
        At=t(A.astype(np.float32))
        ang=np.degrees(np.arccos(np.clip(tD@D[g],-1,1))); o=np.argsort(ang)[:5]
        pred[i]=np.exp(np.log(tv[o]).mean()); act[i]=vl(f'{gtd}/{g}')
        Bs=[]
        for j,m in enumerate(mems):
            p=f'{root}/{m}/eval_png/{os.path.splitext(g)[0]}.png'
            if not os.path.exists(p): continue
            B=np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)
            if B.shape!=A.shape: continue
            Bt=t(B); Bs.append(B)
            with torch.no_grad(): lp=float(LP(At*2-1,Bt*2-1)); ss=float(fs(Bt,At))
            ps=10*np.log10(255.**2/np.mean((A.astype(np.float64)-B.astype(np.float64))**2))
            S[i,j]=100*(0.4*(1-lp)+0.3*ss+0.3*ps/50)
        if Bs:
            M=np.mean(Bs,0); Mt=t(M)
            with torch.no_grad(): lp=float(LP(At*2-1,Mt*2-1)); ss=float(fs(Mt,At))
            ps=10*np.log10(255.**2/np.mean((A.astype(np.float64)-M.astype(np.float64))**2))
            Amean.append(100*(0.4*(1-lp)+0.3*ss+0.3*ps/50))
    ok=~np.isnan(S).any(1)
    S=S[ok]; pred=pred[ok]; act=act[ok]; Am=np.array(Amean)[ok]
    best_fixed=np.nanmean(S,0).max(); bi=int(np.nanargmax(np.nanmean(S,0)))
    oracle=np.nanmax(S,1).mean()
    print(f"== {sc}: {len(mems)} members {mems}, n={len(S)}")
    print(f"   best FIXED member: {mems[bi]} {best_fixed:.3f} | uniform MEAN of all {Am.mean():.3f} | per-frame ORACLE {oracle:.3f} (+{oracle-best_fixed:.3f} over best fixed)")
    # sharpness-routed: split frames at median PREDICTED sharpness, pick best member per half
    for nm,key in [('predicted (train-neighbour)',pred),('actual GT',act)]:
        hi=key>=np.median(key)
        r=np.array([S[i, int(np.nanargmax(np.nanmean(S[hi if hi[i] else ~hi],0)))] for i in range(len(S))])
        # honest 2-fold CV
        idx=np.arange(len(S)); cv=[]
        for f in range(2):
            te=idx%2==f; trn=~te
            sel_hi=int(np.nanargmax(np.nanmean(S[trn&hi],0))); sel_lo=int(np.nanargmax(np.nanmean(S[trn&~hi],0)))
            cv+= [S[i, sel_hi if hi[i] else sel_lo] for i in idx[te]]
        print(f"   route on {nm:28s}: in-sample {r.mean():.3f} (+{r.mean()-best_fixed:+.3f})  2-fold CV {np.mean(cv):.3f} ({np.mean(cv)-best_fixed:+.3f} vs best fixed)")
    print(f"   corr(predicted, actual sharpness) = {np.corrcoef(np.log(pred),np.log(act))[0,1]:+.3f}")
    print("   per-member mean on SHARP half / BLURRY half:")
    hi=act>=np.median(act)
    for j,m in enumerate(mems): print(f"      {m:16s} sharp {np.nanmean(S[hi,j]):.3f}  blurry {np.nanmean(S[~hi,j]):.3f}")
