import sys, os, numpy as np, cv2, torch
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from PIL import Image
import lpips
from utils.loss_utils import ssim as fastgs_ssim
Image.MAX_IMAGE_PIXELS=None
dev='cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
LP=lpips.LPIPS(net='vgg').to(dev).eval()
R='/mnt/d/avv/data/phase1/private_set2'
def vl(p, maxside=1024):
    im=Image.open(p).convert('L'); s=maxside/max(im.size)
    if s<1: im=im.resize((int(im.size[0]*s),int(im.size[1]*s)), Image.LANCZOS)
    return float(cv2.Laplacian(np.asarray(im,dtype=np.float32),cv2.CV_32F).var())
def t(a): return torch.from_numpy(a).float().permute(2,0,1)[None].to(dev)/127.5-1
cfg={'chair':('/mnt/d/avv/evalsplit/chair/eval_gt','/mnt/d/avv/chair_eval/base60k/eval_png','/mnt/d/avv/evalsplit/chair/train_sub'),
     'bonsai':('/mnt/d/avv/evalsplit/bonsai/eval_gt','/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png','/mnt/d/avv/evalsplit/bonsai/train_sub')}
for sc,(gtd,rd,tsd) in cfg.items():
    imgs=read_extrinsics_binary(f'{R}/{sc}/train/sparse/0/images.bin')
    pose={im.name:(qvec2rotmat(im.qvec).T@np.array([0,0,1.0])) for im in imgs.values()}
    tsub=sorted(os.listdir(tsd+'/images'))
    tv=np.array([vl(f'{tsd}/images/{n}') for n in tsub]); tD=np.array([pose[n] for n in tsub])
    rows=[]
    for g in sorted(os.listdir(gtd)):
        stem=os.path.splitext(g)[0]; r=os.path.join(rd,stem+'.png')
        if not os.path.exists(r): continue
        gp=os.path.join(gtd,g)
        A=np.asarray(Image.open(gp).convert('RGB')); B=np.asarray(Image.open(r).convert('RGB'))
        with torch.no_grad(): lp=float(LP(t(A),t(B)))
        ss=float(fastgs_ssim((t(A)+1)/2,(t(B)+1)/2))
        ps=10*np.log10(255.0**2/np.mean((A.astype(np.float64)-B.astype(np.float64))**2))
        ang=np.degrees(np.arccos(np.clip(tD@pose[g],-1,1))); o=np.argsort(ang)[:5]
        rows.append((vl(gp), vl(r), ps, ss, lp, float(np.exp(np.log(tv[o]).mean()))))
    a=np.array(rows); gtv,rev,ps,ss,lp,nbv=a.T
    scr=100*(0.4*(1-lp)+0.3*ss+0.3*ps/50)
    L=np.log
    def pcorr(x,y,z):
        Z=np.c_[np.ones_like(z),z]
        rx=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]; ry=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
        return np.corrcoef(rx,ry)[0,1]
    print(f"== {sc} n={len(a)} mean score {scr.mean():.3f} (PSNR {ps.mean():.2f} SSIM {ss.mean():.4f} LPIPS {lp.mean():.4f})")
    for nm,v in [('PSNR',ps),('SSIM',ss),('LPIPS',lp),('SCORE',scr)]:
        print(f"   partial corr({nm:5s}, log nbTrainVoL | log ownGTVoL) = {pcorr(v,L(nbv),L(gtv)):+.3f}   raw corr w/ ownGTVoL {np.corrcoef(v,L(gtv))[0,1]:+.3f}")
    X=np.c_[np.ones(len(a)),L(nbv),L(gtv)]
    b=np.linalg.lstsq(X,scr,rcond=None)[0]
    print(f"   d(score)/d(log2 nbTrainVoL) holding content = {b[1]*np.log(2):+.3f} pts per doubling of local train sharpness")
    o=np.argsort(gtv); k=len(o)//4
    print(f"   blurriest-GT quartile score {scr[o[:k]].mean():.2f}  |  sharpest-GT quartile {scr[o[-k:]].mean():.2f}")
