import os,sys,re,math,shutil,subprocess
import numpy as np, cv2
sys.path.insert(0,'/home/bkai/.claude/jobs/1c9cf7e9/tmp')
from blurmotion import read_images_bin,qrot,read_cam_focal

scene='bonsai'
root=f'/mnt/d/avv/data/phase1/private_set2/{scene}/train'
REN='/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png'
GT='/mnt/d/avv/evalsplit/bonsai/eval_gt'
OUT='/mnt/d/avv/tmp_blurgate'

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
    return np.array([f_px*p[0]/p[2]+cx, f_px*p[1]/p[2]+cy])

evalnames=sorted(os.listdir(REN))
mv={}
for fn in evalnames:
    nm=fn.replace('.png','.jpg')
    i=names.index(nm)
    a=names[max(0,i-1)];b=names[min(len(names)-1,i+1)]
    di=idx[min(len(names)-1,i+1)]-idx[max(0,i-1)]
    P=C[nm]+depth*(R[nm].T@np.array([0,0,1.0]))
    d=(proj(b,P)-proj(a,P))/max(di,1)
    mv[fn]=d
mags=np.array([np.linalg.norm(mv[f]) for f in evalnames])
print(f'eval-frame motion px/video-frame: med={np.median(mags):.2f} p5={np.percentile(mags,5):.2f} p95={np.percentile(mags,95):.2f}',flush=True)

def linekernel(L,theta):
    L=max(L,1e-6); n=int(math.ceil(L))|1; n=max(n,3)
    k=np.zeros((n,n),np.float32); c=n//2
    S=64
    for s in range(S):
        u=(s/(S-1)-0.5)*L
        x=c+u*math.cos(theta); y=c+u*math.sin(theta)
        x0,y0=int(math.floor(x)),int(math.floor(y)); fx,fy=x-x0,y-y0
        for dy in (0,1):
            for dx in (0,1):
                xx,yy=x0+dx,y0+dy
                if 0<=xx<n and 0<=yy<n:
                    k[yy,xx]+=(fx if dx else 1-fx)*(fy if dy else 1-fy)
    return k/k.sum()

def run(tag,taus,shuffle=False):
    rng=np.random.RandomState(0)
    sh=rng.permutation(len(evalnames))
    for tau in taus:
        d=os.path.join(OUT,f'{tag}_{tau}')
        os.makedirs(d,exist_ok=True)
        for j,fn in enumerate(evalnames):
            im=cv2.imread(os.path.join(REN,fn))
            v=mv[fn]; L=tau*np.linalg.norm(v)
            if L<0.3:
                out=im
            else:
                th=math.atan2(v[1],v[0])
                if shuffle:
                    v2=mv[evalnames[sh[j]]]; th=math.atan2(v2[1],v2[0])
                out=cv2.filter2D(im,-1,linekernel(L,th),borderType=cv2.BORDER_REFLECT)
            cv2.imwrite(os.path.join(d,fn),out)
        r=subprocess.run(['python','/mnt/c/Users/BKAI/an_plaza2/FastGS/scripts/eval_score.py','--render_dir',d,'--gt_dir',GT,'--tag',f'{tag}{tau}'],capture_output=True,text=True)
        print(f'{tag} tau={tau}: '+r.stdout.strip().replace('\n',' | ')+' ERR:'+r.stderr.strip()[-300:],flush=True)
        shutil.rmtree(d)

run('true',[0.0,0.15,0.30,0.50,0.75])
run('shuf',[0.30],shuffle=True)
