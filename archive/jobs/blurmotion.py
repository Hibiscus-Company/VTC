import sys, os, glob, struct, math
import numpy as np, cv2

def read_images_bin(path):
    out={}
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            iid,qw,qx,qy,qz,tx,ty,tz,cid=struct.unpack('<idddddddi',f.read(64))
            name=b''
            while True:
                c=f.read(1)
                if c==b'\x00': break
                name+=c
            np2=struct.unpack('<Q',f.read(8))[0]
            f.seek(24*np2,1)
            out[name.decode()]=(np.array([qw,qx,qy,qz]),np.array([tx,ty,tz]))
    return out

def qrot(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def read_cam_focal(path):
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        cid,mid,w,h=struct.unpack('<iiQQ',f.read(24))
        npar={0:3,1:4,2:4,3:5,4:8,5:8,6:12,7:5,8:4,9:5,10:12,11:5}[mid]
        p=struct.unpack('<'+'d'*npar,f.read(8*npar))
    return p[0], w, h

scene=sys.argv[1]
root=f'/mnt/d/avv/data/phase1/private_set2/{scene}/train'
imgs=read_images_bin(os.path.join(root,'sparse/0/images.bin'))
f_px,W,H=read_cam_focal(os.path.join(root,'sparse/0/cameras.bin'))
names=sorted(imgs.keys())
# camera centers and rotations
C={}; R={}
for nm in names:
    q,t=imgs[nm]; Rm=qrot(q); R[nm]=Rm; C[nm]=-Rm.T@t

# scene scale: median pairwise distance of centers -> use median distance to point cloud centroid
Cs=np.array([C[n] for n in names])
cen=Cs.mean(0)
scale=np.median(np.linalg.norm(Cs-cen,axis=1))  # proxy for typical depth

rows=[]
for i,nm in enumerate(names):
    p=os.path.join(root,'images',nm)
    im=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    if im is None: continue
    vol=cv2.Laplacian(im.astype(np.float32)/255.,cv2.CV_32F).var()
    # neighbours
    j0=names[max(0,i-1)]; j1=names[min(len(names)-1,i+1)]
    dR=R[nm].T@R[j1] if i+1<len(names) else R[j0].T@R[nm]
    ang=math.acos(max(-1,min(1,(np.trace(dR)-1)/2)))          # rad per frame-step
    dC=np.linalg.norm(C[j1]-C[nm]) if i+1<len(names) else np.linalg.norm(C[nm]-C[j0])
    # px motion: rotation term f*ang ; translation term f*dC/depth (depth~scale)
    px_rot=f_px*ang
    px_tr=f_px*dC/max(scale,1e-9)
    rows.append((nm,vol,ang,dC,px_rot,px_tr,px_rot+px_tr))

vol=np.array([r[1] for r in rows]); tot=np.array([r[6] for r in rows])
prot=np.array([r[4] for r in rows]); ptr=np.array([r[5] for r in rows])
def pear(a,b):
    a=(a-a.mean())/(a.std()+1e-12); b=(b-b.mean())/(b.std()+1e-12); return float((a*b).mean())
def spear(a,b):
    from scipy.stats import rankdata
    return pear(rankdata(a),rankdata(b))
print(f'== {scene}  n={len(rows)}  f={f_px:.1f}px  {W}x{H}  scale(depth proxy)={scale:.3f}')
print(f'VoL: p5={np.percentile(vol,5):.5f} med={np.median(vol):.5f} p95={np.percentile(vol,95):.5f} ratio_med/p5={np.median(vol)/np.percentile(vol,5):.2f}')
print(f'px_motion/frame: med rot={np.median(prot):.2f} tr={np.median(ptr):.2f} tot={np.median(tot):.2f}  p95tot={np.percentile(tot,95):.2f}')
for lbl,x in [('rot',prot),('tr',ptr),('tot',tot)]:
    print(f'  corr(VoL, {lbl}) pearson={pear(vol,x):+.3f} spearman={spear(vol,x):+.3f}   corr(log VoL,log x)={pear(np.log(vol+1e-9),np.log(x+1e-9)):+.3f}')
# sharpest vs blurriest quartile motion
o=np.argsort(vol); q=len(o)//4
print(f'  blurriest quartile median tot_px={np.median(tot[o[:q]]):.2f} | sharpest quartile median tot_px={np.median(tot[o[-q:]]):.2f}')
np.save(f'/home/bkai/.claude/jobs/1c9cf7e9/tmp/{scene}_blurmotion.npy',np.array([vol,prot,ptr,tot]))
