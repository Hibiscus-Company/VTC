import os, sys, numpy as np, cv2
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
R='/mnt/d/avv/data/phase1/private_set2'
# (a) tower render/GT detail ratio  (natural experiment: uniform-sharpness supervision)
def vl(p,ms=1024):
    im=Image.open(p).convert('L'); s=ms/max(im.size)
    if s<1: im=im.resize((int(im.size[0]*s),int(im.size[1]*s)),Image.LANCZOS)
    return float(cv2.Laplacian(np.asarray(im,dtype=np.float32),cv2.CV_32F).var())
gtd='/mnt/d/avv/evalsplit/HCM0421/eval_gt'; rd='/mnt/d/avv/evalgen/HCM0421/eval_png'
rows=[]
for g in sorted(os.listdir(gtd)):
    p=os.path.join(rd,os.path.splitext(g)[0]+'.png')
    if os.path.exists(p): rows.append((vl(os.path.join(gtd,g)),vl(p)))
a=np.array(rows); o=np.argsort(a[:,0]); k=len(o)//4
print(f"(a) HCM0421 tower  ren/gt detail ratio: blurQ {a[o[:k],1].mean()/a[o[:k],0].mean():.2f}  sharpQ {a[o[-k:],1].mean()/a[o[-k:],0].mean():.2f}  overall {(a[:,1]/a[:,0]).mean():.2f}")
print(f"    (videos for comparison: chair 0.95->0.40, bonsai 1.20->0.20)")
# (b) motion vs defocus gate: content-controlled sharpness rho_f vs per-frame angular velocity
for sc in ['chair','bonsai']:
    rel={}
    for L in open(f'/tmp/relsharp_{sc}.txt'):
        n,v=L.split('\t'); rel[n]=float(v)
    imgs=read_extrinsics_binary(f'{R}/{sc}/train/sparse/0/images.bin')
    ondisk=sorted(rel)
    Rm={im.name:qvec2rotmat(im.qvec) for im in imgs.values()}
    Cc={im.name:-qvec2rotmat(im.qvec).T@im.tvec for im in imgs.values()}
    om=[];tr=[];r=[]
    for i,n in enumerate(ondisk):
        j=min(i+1,len(ondisk)-1); i0=max(i-1,0)
        if j==i0: continue
        dR=Rm[ondisk[j]]@Rm[ondisk[i0]].T
        om.append(np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2,-1,1)))/(j-i0))
        tr.append(np.linalg.norm(Cc[ondisk[j]]-Cc[ondisk[i0]])/(j-i0))
        r.append(rel[n])
    om=np.array(om);tr=np.array(tr);r=np.array(r)
    print(f"(b) {sc}: corr(log rho_f, log angular-vel) = {np.corrcoef(np.log(r),np.log(om+1e-6))[0,1]:+.3f}   "
          f"corr(log rho_f, log translational-vel) = {np.corrcoef(np.log(r),np.log(tr+1e-9))[0,1]:+.3f}   "
          f"[negative & strong => MOTION blur; ~0 => DEFOCUS/focus-hunt]")
