import numpy as np, struct, csv, os, sys
from PIL import Image

P3D='/mnt/d/avv/data/phase1/private_set2/bonsai/train/sparse/0/points3D.bin'
GT='/mnt/d/avv/evalsplit/bonsai/eval_gt'
POSES='/mnt/d/avv/evalsplit/bonsai/eval_poses.csv'

def read_points3D(path):
    xyz=[]
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            f.read(8)
            x,y,z=struct.unpack('<3d',f.read(24))
            f.read(3); f.read(8)
            tl=struct.unpack('<Q',f.read(8))[0]
            f.read(8*tl)
            xyz.append((x,y,z))
    return np.array(xyz)

def qvec2R(q):
    w,x,y,z=q
    return np.array([
     [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
     [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
     [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])

xyz=read_points3D(P3D)
print('points3D:',xyz.shape)

rows=list(csv.DictReader(open(POSES)))
CS=int(sys.argv[2]) if len(sys.argv)>2 else 64
REN=sys.argv[1]

recs=[]
for r in rows:
    name=r['image_name']
    gp=os.path.join(GT,name); rp=os.path.join(REN,name)
    if not os.path.exists(rp):
        rp2=rp.replace('.jpg','.png')
        if os.path.exists(rp2): rp=rp2
        else: print('MISS',name); continue
    g=np.asarray(Image.open(gp).convert('RGB'),dtype=np.float32)/255.
    d=np.asarray(Image.open(rp).convert('RGB'),dtype=np.float32)/255.
    H,W,_=g.shape
    assert d.shape==g.shape,(d.shape,g.shape)
    q=np.array([float(r[k]) for k in ('qw','qx','qy','qz')])
    t=np.array([float(r[k]) for k in ('tx','ty','tz')])
    R=qvec2R(q)
    fx,fy,cx,cy=(float(r[k]) for k in ('fx','fy','cx','cy'))
    Xc=xyz@R.T+t
    m=Xc[:,2]>1e-6
    u=fx*Xc[m,0]/Xc[m,2]+cx; v=fy*Xc[m,1]/Xc[m,2]+cy
    ok=(u>=0)&(u<W)&(v>=0)&(v<H)
    u=u[ok]; v=v[ok]
    ny,nx=H//CS, W//CS
    cnt=np.zeros((ny,nx))
    ci=(v//CS).astype(int); cj=(u//CS).astype(int)
    kk=(ci<ny)&(cj<nx)
    np.add.at(cnt,(ci[kk],cj[kk]),1)
    gy=g[:ny*CS,:nx*CS]; dy=d[:ny*CS,:nx*CS]
    err=((gy-dy)**2).mean(axis=2)
    gray=gy.mean(axis=2)
    gx=np.zeros_like(gray); gyv=np.zeros_like(gray)
    gx[:,1:]=np.diff(gray,axis=1); gyv[1:,:]=np.diff(gray,axis=0)
    grad=gx**2+gyv**2
    ec=err.reshape(ny,CS,nx,CS).mean(axis=(1,3))
    gc=grad.reshape(ny,CS,nx,CS).mean(axis=(1,3))
    for a in range(ny):
        for b in range(nx):
            recs.append((cnt[a,b],ec[a,b],gc[a,b]))
A=np.array(recs)
np.save('/home/bkai/.claude/jobs/1c9cf7e9/tmp/cells_%s_%d.npy'%(os.path.basename(os.path.dirname(REN.rstrip('/')))or'x',CS),A)
cnt,err,grad=A[:,0],A[:,1],A[:,2]
print('cells',len(A),'cellsize',CS)
print('overall PSNR %.4f'%(-10*np.log10(err.mean())))
from scipy.stats import spearmanr
print('spearman(cnt,err)=%.4f  spearman(cnt,grad)=%.4f  spearman(grad,err)=%.4f'%(
   spearmanr(cnt,err).statistic, spearmanr(cnt,grad).statistic, spearmanr(grad,err).statistic))
print('zero-point cells: area %.1f%%  err-share %.1f%%'%(100*(cnt==0).mean(),100*err[cnt==0].sum()/err.sum()))

def oracle(nq, label):
    qe=np.quantile(grad,np.linspace(0,1,nq+1)); qe[0]=-1; qe[-1]=np.inf
    qi=np.digitize(grad,qe[1:-1])
    tot=err.sum(); saved=0.0; rowinfo=[]
    for q in range(nq):
        m=qi==q
        c=cnt[m]; e=err[m]
        t1,t2=np.quantile(c,[1/3,2/3])
        sp=c<=t1; dn=c>=t2
        if sp.sum()<5 or dn.sum()<5: continue
        mse_sp=e[sp].mean(); mse_dn=e[dn].mean()
        if mse_sp>mse_dn:
            saved+= e[sp].sum()*(1-mse_dn/mse_sp)
        rowinfo.append((q,sp.sum(),dn.sum(),c[sp].mean(),c[dn].mean(),
            -10*np.log10(mse_sp),-10*np.log10(mse_dn),100*e[sp].sum()/tot,100*sp.sum()/len(A)))
    base=-10*np.log10(err.mean()); new=-10*np.log10((tot-saved)/len(err))
    print('  %s nbins=%d: oracle %+.4f dB (%.4f -> %.4f), err removed %.1f%%'%(label,nq,new-base,base,new,100*saved/tot))
    return rowinfo,new-base

ri,_=oracle(4,'quartile')
print('  bin  nsp ndn  pts_sp pts_dn  PSNR_sp PSNR_dn  errshare_sp% area_sp%')
for r in ri: print('  Q%d %5d %5d %7.1f %7.1f  %6.2f %6.2f   %5.1f  %5.1f'%(r[0]+1,r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8]))
for nq in (8,16,32,64):
    oracle(nq,'finer')
