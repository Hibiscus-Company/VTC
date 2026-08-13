import numpy as np, csv, os
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
p="/mnt/d/avv/evalsplit/bonsai/train_sub/sparse/0/points3D.ply"
with open(p,'rb') as f:
    hdr=b''
    while b'end_header' not in hdr: hdr+=f.readline()
    L=hdr.decode('ascii','ignore').split('\n')
    n=[int(l.split()[-1]) for l in L if l.startswith('element vertex')][0]
    fields=[l.split()[-1] for l in L if l.startswith('property')]
    types=[l.split()[1] for l in L if l.startswith('property')]
    tm={'float':'f4','double':'f8','uchar':'u1','int':'i4'}
    dt=np.dtype([(fields[i],tm[types[i]]) for i in range(len(fields))])
    arr=np.frombuffer(f.read(n*dt.itemsize),dtype=dt,count=n)
XYZ=np.stack([arr['x'],arr['y'],arr['z']],1).astype(np.float64)
def q2R(w,x,y,z):
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
rows=list(csv.DictReader(open("/mnt/d/avv/evalsplit/bonsai/eval_poses.csv")))
CELL=40; gtd="/mnt/d/avv/evalsplit/bonsai/eval_gt"
runs={"pC":"/mnt/d/avv/bonsai_perc/pC_lpearly/eval_png","dp05":"/mnt/d/avv/depth/bonsai_dp05/eval_png"}
D=[];E={k:[] for k in runs};V=[]
for r in rows:
    W,H=int(r['width']),int(r['height']); fx,fy,cx,cy=float(r['fx']),float(r['fy']),float(r['cx']),float(r['cy'])
    R=q2R(float(r['qw']),float(r['qx']),float(r['qy']),float(r['qz']))
    t=np.array([float(r['tx']),float(r['ty']),float(r['tz'])])
    Xc=XYZ@R.T+t; m=Xc[:,2]>1e-3; Xc=Xc[m]
    u=fx*Xc[:,0]/Xc[:,2]+cx; v=fy*Xc[:,1]/Xc[:,2]+cy
    k=(u>=0)&(u<W)&(v>=0)&(v<H); u,v=u[k],v[k]
    gh,gw=H//CELL,W//CELL; dens=np.zeros((gh,gw))
    np.add.at(dens,((v//CELL).astype(int),(u//CELL).astype(int)),1)
    s=os.path.splitext(r['image_name'])[0]
    g=np.asarray(Image.open(os.path.join(gtd,s+'.jpg')).convert('RGB'),dtype=np.float64)/255.
    gg=g.mean(2)[:gh*CELL,:gw*CELL]
    # local GT detail = mean squared gradient inside the cell
    gx=np.zeros_like(gg); gy=np.zeros_like(gg)
    gx[:,1:]=np.diff(gg,axis=1); gy[1:,:]=np.diff(gg,axis=0)
    det=((gx**2+gy**2).reshape(gh,CELL,gw,CELL).mean((1,3)))
    D.append(dens.ravel()); V.append(det.ravel())
    for kk,rdir in runs.items():
        q=np.asarray(Image.open(os.path.join(rdir,s+'.png')).convert('RGB'),dtype=np.float64)/255.
        E[kk].append(((q-g)**2).mean(2)[:gh*CELL,:gw*CELL].reshape(gh,CELL,gw,CELL).mean((1,3)).ravel())
D=np.concatenate(D); V=np.concatenate(V); E={k:np.concatenate(v) for k,v in E.items()}
print(f"corr(rank density, rank GT-detail) = {np.corrcoef(np.argsort(np.argsort(D)),np.argsort(np.argsort(V)))[0,1]:.4f}  <-- confound size")
# stratify by GT detail, then look at density WITHIN each stratum
qs=np.quantile(V,[0,.25,.5,.75,1.0])
print(f"{'GT-detail stratum':>20s} | {'low-dens MSE':>12s} {'high-dens MSE':>13s} | pC dB low/high | dp05 dB low/high")
for i in range(4):
    sel=(V>=qs[i])&(V<=qs[i+1]); d=D[sel]
    lo=d<=np.quantile(d,0.33); hi=d>=np.quantile(d,0.67)
    e=E['pC'][sel]; e2=E['dp05'][sel]
    print(f"  Q{i+1} det<= {qs[i+1]:.5f} | {e[lo].mean():12.5f} {e[hi].mean():13.5f} | "
          f"{10*np.log10(1/e[lo].mean()):5.2f}/{10*np.log10(1/e[hi].mean()):5.2f} | "
          f"{10*np.log10(1/e2[lo].mean()):5.2f}/{10*np.log10(1/e2[hi].mean()):5.2f}  "
          f"(pts/cell {d[lo].mean():.1f} vs {d[hi].mean():.1f})")
# global: how much of total squared error lives in ZERO-point cells?
z=D==0
print(f"\nzero-SfM-point cells: {z.mean()*100:.1f}% of area, {E['pC'][z].sum()/E['pC'].sum()*100:.1f}% of total squared error")
print(f"dp05-vs-pC MSE delta in zero-point cells: {(E['dp05'][z].mean()-E['pC'][z].mean())/E['pC'][z].mean()*100:+.1f}% ; in nonzero: {(E['dp05'][~z].mean()-E['pC'][~z].mean())/E['pC'][~z].mean()*100:+.1f}%")

# ---- ORACLE accounting for candidate (b): lift sparse-but-textured cells to their dense peers ----
tot=E['pC'].mean()
print(f"\nORACLE (b): pC baseline mean MSE {tot:.6f} = {10*np.log10(1/tot):.4f} dB (matches scorer)")
new=E['pC'].copy()
for i in range(4):
    sel=np.where((V>=qs[i])&(V<=qs[i+1]))[0]; d=D[sel]
    lo=sel[d<=np.quantile(d,0.33)]; hi=sel[d>=np.quantile(d,0.67)]
    tgt=E['pC'][hi].mean()
    share=E['pC'][lo].sum()/E['pC'].sum()*100
    if E['pC'][lo].mean()>tgt:
        new[lo]=tgt
        print(f"  Q{i+1}: sparse tertile = {len(lo)/len(D)*100:4.1f}% of area, {share:4.1f}% of total sq-err, "
              f"MSE {E['pC'][lo].mean():.5f} -> {tgt:.5f} (its dense peers)")
    else:
        print(f"  Q{i+1}: sparse tertile already <= dense peers ({share:4.1f}% of err) - no lift")
print(f"  => oracle PSNR {10*np.log10(1/new.mean()):.4f} dB  (delta {10*np.log10(1/new.mean())-10*np.log10(1/tot):+.4f} dB)")
