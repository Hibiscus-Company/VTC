import numpy as np, csv, os
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
# --- read binary ply ---
p="/mnt/d/avv/evalsplit/bonsai/train_sub/sparse/0/points3D.ply"
with open(p,'rb') as f:
    hdr=b''
    while b'end_header' not in hdr: hdr+=f.readline()
    props=[l for l in hdr.decode('ascii','ignore').split('\n')]
    n=[int(l.split()[-1]) for l in props if l.startswith('element vertex')][0]
    fields=[l.split()[-1] for l in props if l.startswith('property')]
    types=[l.split()[1] for l in props if l.startswith('property')]
    tm={'float':'f4','float32':'f4','double':'f8','uchar':'u1','uint8':'u1','int':'i4'}
    dt=np.dtype([(fields[i],tm[types[i]]) for i in range(len(fields))])
    arr=np.frombuffer(f.read(n*dt.itemsize),dtype=dt,count=n)
XYZ=np.stack([arr['x'],arr['y'],arr['z']],1).astype(np.float64)
print("points:",XYZ.shape, "fields:",fields)
def q2R(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
rows=list(csv.DictReader(open("/mnt/d/avv/evalsplit/bonsai/eval_poses.csv")))
CELL=40
gtd="/mnt/d/avv/evalsplit/bonsai/eval_gt"
rd="/mnt/d/avv/bonsai_perc/pC_lpearly/eval_png"
D_all=[];E_all=[]
for r in rows:
    W,H=int(r['width']),int(r['height']); fx,fy,cx,cy=float(r['fx']),float(r['fy']),float(r['cx']),float(r['cy'])
    R=q2R([float(r['qw']),float(r['qx']),float(r['qy']),float(r['qz])'.replace(')','')]) if False else float(r['qz'])])
    t=np.array([float(r['tx']),float(r['ty']),float(r['tz'])])
    Xc=XYZ@R.T+t
    m=Xc[:,2]>1e-3; Xc=Xc[m]
    u=fx*Xc[:,0]/Xc[:,2]+cx; v=fy*Xc[:,1]/Xc[:,2]+cy
    k=(u>=0)&(u<W)&(v>=0)&(v<H); u,v=u[k],v[k]
    gh,gw=H//CELL,W//CELL
    dens=np.zeros((gh,gw))
    np.add.at(dens,(np.clip((v//CELL).astype(int),0,gh-1),np.clip((u//CELL).astype(int),0,gw-1)),1)
    s=os.path.splitext(r['image_name'])[0]
    g=np.asarray(Image.open(os.path.join(gtd,s+'.jpg')).convert('RGB'),dtype=np.float64)/255.
    q=np.asarray(Image.open(os.path.join(rd,s+'.png')).convert('RGB'),dtype=np.float64)/255.
    err=((q-g)**2).mean(2)[:gh*CELL,:gw*CELL].reshape(gh,CELL,gw,CELL).mean((1,3))
    D_all.append(dens.ravel()); E_all.append(err.ravel())
D=np.concatenate(D_all); E=np.concatenate(E_all)
print(f"cells={len(D)} pts-in-view mean/frame={D.reshape(len(rows),-1).sum(1).mean():.0f}  empty-cell frac={np.mean(D==0):.3f}")
print(f"spearman-ish corr(density, err) = {np.corrcoef(np.argsort(np.argsort(D)),np.argsort(np.argsort(E)))[0,1]:.4f}")
o=np.argsort(D)
q=len(D)//10
for name,idx in [("lowest-density decile",o[:q]),("median decile",o[len(D)//2-q//2:len(D)//2+q//2]),("highest-density decile",o[-q:])]:
    print(f"  {name:24s} mean pts/cell {D[idx].mean():7.2f}  MSE {E[idx].mean():.5f}  -> PSNR {10*np.log10(1/E[idx].mean()):.2f} dB")
