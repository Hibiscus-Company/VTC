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

for sc in ['chair','bonsai']:
    sp=f'{R}/{sc}/train/sparse/0'; imd=f'{R}/{sc}/train/images'
    imgs=read_extrinsics_binary(sp+'/images.bin')
    ondisk=set(os.listdir(imd))
    tr={}; te={}
    for im in imgs.values():
        Rm=qvec2rotmat(im.qvec); C=-Rm.T@im.tvec; d=Rm.T@np.array([0,0,1.0])
        (tr if im.name in ondisk else te)[im.name]=(C,d)
    vs={n: vl(os.path.join(imd,n)) for n in tr}
    names=sorted(tr); V=np.array([vs[n] for n in names])
    # angular distance from each test pose to nearest kept train pose
    tenames=sorted(te)
    def cover(keep_names):
        Cs=np.array([tr[n][0] for n in keep_names]); Ds=np.array([tr[n][1] for n in keep_names])
        out=[]
        for tn in tenames:
            C,d=te[tn]
            ang=np.degrees(np.arccos(np.clip(Ds@d,-1,1)))
            dist=np.linalg.norm(Cs-C,axis=1)
            sc_=ang+ 0  # angle proxy
            out.append((ang.min(), dist.min()))
        return np.array(out)
    base=cover(names)
    print(f"== {sc}: N_train={len(names)} N_test={len(tenames)}")
    print(f"   ALL frames:  nearest-train angle med {np.median(base[:,0]):.2f} deg  p90 {np.percentile(base[:,0],90):.2f}  max {base[:,0].max():.2f}")
    for frac in [0.75,0.5,0.35]:
        thr=np.quantile(V,1-frac); keep=[n for n in names if vs[n]>=thr]
        c=cover(keep)
        print(f"   keep top {frac:.0%} sharpest (n={len(keep)}, VoL>={thr:.0f}): angle med {np.median(c[:,0]):.2f} p90 {np.percentile(c[:,0],90):.2f} max {c[:,0].max():.2f}")
    # is sharpness clustered in time?
    idx=np.arange(len(V)); lv=np.log(V)
    ac=np.corrcoef(lv[:-1],lv[1:])[0,1]
    print(f"   log-VoL lag-1 autocorr along frame order: {ac:.3f}  (1 = long blurry runs, 0 = isolated blurry frames)")
    # longest run of below-median frames
    b=(V<np.median(V)).astype(int); runs=[];c_=0
    for x in b:
        c_=c_+1 if x else 0; runs.append(c_)
    print(f"   longest consecutive run of below-median-sharpness frames: {max(runs)}")
