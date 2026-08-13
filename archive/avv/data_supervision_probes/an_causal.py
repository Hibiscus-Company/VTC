import sys, os, numpy as np, cv2
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
R='/mnt/d/avv/data/phase1/private_set2'
def vl(p, maxside=1024):
    im=Image.open(p).convert('L'); s=maxside/max(im.size)
    if s<1: im=im.resize((int(im.size[0]*s),int(im.size[1]*s)), Image.LANCZOS)
    return float(cv2.Laplacian(np.asarray(im,dtype=np.float32),cv2.CV_32F).var())
def psnr(a,b):
    m=np.mean((a.astype(np.float64)-b.astype(np.float64))**2); return 10*np.log10(255.0**2/m)
cfg={'chair':('/mnt/d/avv/evalsplit/chair/eval_gt','/mnt/d/avv/chair_eval/base60k/eval_png','/mnt/d/avv/evalsplit/chair/train_sub'),
     'bonsai':('/mnt/d/avv/evalsplit/bonsai/eval_gt','/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png','/mnt/d/avv/evalsplit/bonsai/train_sub')}
for sc,(gtd,rd,tsd) in cfg.items():
    sp=f'{R}/{sc}/train/sparse/0'
    imgs=read_extrinsics_binary(sp+'/images.bin')
    pose={}
    for im in imgs.values():
        Rm=qvec2rotmat(im.qvec); pose[im.name]=(-Rm.T@im.tvec, Rm.T@np.array([0,0,1.0]))
    tsub=sorted(os.listdir(tsd+'/images'))
    tv=np.array([vl(f'{tsd}/images/{n}') for n in tsub])
    tC=np.array([pose[n][0] for n in tsub]); tD=np.array([pose[n][1] for n in tsub])
    rows=[]
    for g in sorted(os.listdir(gtd)):
        stem=os.path.splitext(g)[0]; r=os.path.join(rd,stem+'.png')
        if not os.path.exists(r): continue
        gp=os.path.join(gtd,g)
        A=np.asarray(Image.open(gp).convert('RGB')); B=np.asarray(Image.open(r).convert('RGB'))
        C,D=pose[g]
        ang=np.degrees(np.arccos(np.clip(tD@D,-1,1)))
        o=np.argsort(ang)[:5]
        rows.append((vl(gp), vl(r), psnr(A,B), np.exp(np.log(tv[o]).mean()), ang[o].mean()))
    a=np.array(rows)
    gtv,rev,ps,nbv,nba=a.T
    ratio=rev/gtv
    L=lambda x: np.log(x)
    def pcorr(x,y,z):  # partial corr of x,y controlling z
        import numpy.linalg as la
        Z=np.c_[np.ones_like(z),z]
        rx=x-Z@la.lstsq(Z,x,rcond=None)[0]; ry=y-Z@la.lstsq(Z,y,rcond=None)[0]
        return np.corrcoef(rx,ry)[0,1]
    print(f"== {sc}  n={len(a)}   (nbv = geo-mean VoL of 5 nearest TRAIN frames)")
    print(f"   corr(log nbVoL, log ownGTVoL)      = {np.corrcoef(L(nbv),L(gtv))[0,1]:+.3f}   <- collinearity")
    print(f"   corr(log ren/gt ratio, log ownGT)  = {np.corrcoef(L(ratio),L(gtv))[0,1]:+.3f}")
    print(f"   corr(log ren/gt ratio, log nbVoL)  = {np.corrcoef(L(ratio),L(nbv))[0,1]:+.3f}")
    print(f"   PARTIAL corr(ratio, nbVoL | ownGT) = {pcorr(L(ratio),L(nbv),L(gtv)):+.3f}   <- neighbour-blur causal term")
    print(f"   PARTIAL corr(ratio, ownGT | nbVoL) = {pcorr(L(ratio),L(gtv),L(nbv)):+.3f}")
    print(f"   PARTIAL corr(PSNR,  nbVoL | ownGT) = {pcorr(ps,L(nbv),L(gtv)):+.3f}")
    # slope of render sharpness on neighbour sharpness holding own content
    X=np.c_[np.ones(len(a)),L(nbv),L(gtv)]
    b=np.linalg.lstsq(X,L(rev),rcond=None)[0]
    print(f"   log renVoL = {b[0]:.2f} + {b[1]:+.3f}*log nbVoL {b[2]:+.3f}*log ownGTVoL")
    print(f"   -> a 2x sharper local TRAIN neighbourhood raises render detail energy by {100*(2**b[1]-1):.0f}%")
