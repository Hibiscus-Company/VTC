import os,sys,re,math
import numpy as np, cv2
from scipy.stats import rankdata
sys.path.insert(0,'/home/bkai/.claude/jobs/1c9cf7e9/tmp')
sys.argv=['x','bonsai']
from blurmotion import read_images_bin,qrot,read_cam_focal
def pear(a,b):
    a=(a-a.mean())/(a.std()+1e-12);b=(b-b.mean())/(b.std()+1e-12);return float((a*b).mean())
def spear(a,b): return pear(rankdata(a),rankdata(b))

for scene in ['bonsai','chair']:
    root=f'/mnt/d/avv/data/phase1/private_set2/{scene}/train'
    imgs=read_images_bin(os.path.join(root,'sparse/0/images.bin'))
    f_px,W,H=read_cam_focal(os.path.join(root,'sparse/0/cameras.bin'))
    names=sorted(imgs.keys(),key=lambda n:int(re.findall(r'(\d+)',n)[-1]))
    idx=np.array([int(re.findall(r'(\d+)',n)[-1]) for n in names])
    R={};T={};C={}
    for nm in names:
        q,t=imgs[nm];Rm=qrot(q);R[nm]=Rm;T[nm]=t;C[nm]=-Rm.T@t
    Cs=np.array([C[n] for n in names]);depth=np.median(np.linalg.norm(Cs-Cs.mean(0),axis=1))
    cx,cy=W/2.,H/2.
    def proj(nm,P):
        p=R[nm]@P+T[nm]
        if p[2]<=1e-6: return np.array([np.nan,np.nan])
        return np.array([f_px*p[0]/p[2]+cx,f_px*p[1]/p[2]+cy])
    # grid of virtual points at median depth -> mean |flow| over the frame (not just centre)
    gx,gy=np.meshgrid(np.linspace(-0.4,0.4,5),np.linspace(-0.4,0.4,5))
    flow={}
    for i,nm in enumerate(names):
        a=names[max(0,i-1)];b=names[min(len(names)-1,i+1)]
        di=idx[min(len(names)-1,i+1)]-idx[max(0,i-1)]
        ds=[]
        for u,v in zip(gx.ravel(),gy.ravel()):
            dirc=R[nm].T@np.array([u*W/f_px,v*H/f_px,1.0]); dirc/=np.linalg.norm(dirc)
            P=C[nm]+depth*dirc
            d=(proj(b,P)-proj(a,P))/max(di,1)
            if np.isfinite(d).all(): ds.append(np.linalg.norm(d))
        flow[nm]=np.median(ds) if ds else np.nan
    vols=[];hfs=[];fl=[]
    for nm in names:
        p=os.path.join(root,'images',nm)
        im=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        g=im.astype(np.float32)/255.
        gs=cv2.resize(g,(512,512))
        Fm=np.abs(np.fft.fftshift(np.fft.fft2(gs*np.hanning(512)[:,None]*np.hanning(512)[None,:])))**2
        yy,xx=np.mgrid[0:512,0:512];r=np.sqrt((yy-256)**2+(xx-256)**2)/256.
        hfs.append(Fm[(r>0.30)&(r<0.55)].mean()/(Fm[(r>0.10)&(r<0.20)].mean()+1e-30))
        vols.append(cv2.Laplacian(g,cv2.CV_32F).var()/(g.var()+1e-12))
        fl.append(flow[nm])
    hfs=np.array(hfs);vols=np.array(vols);fl=np.array(fl)
    print(f'\n== {scene} n={len(fl)}  TRUE projected flow px/video-frame: p5={np.percentile(fl,5):.2f} med={np.median(fl):.2f} p95={np.percentile(fl,95):.2f}')
    for lbl,b in [('HF/MID',hfs),('VoL/var',vols)]:
        s=np.argsort(b);q=len(s)//4
        print(f'   corr({lbl}, flow) pearson={pear(b,fl):+.3f} spearman={spear(b,fl):+.3f} '
              f'| blurriest-Q med flow={np.median(fl[s[:q]]):.2f} sharpest-Q med flow={np.median(fl[s[-q:]]):.2f}')
