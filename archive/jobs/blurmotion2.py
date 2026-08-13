import sys, os, re, struct, math
import numpy as np, cv2
from scipy.stats import rankdata
sys.path.insert(0,'/home/bkai/.claude/jobs/1c9cf7e9/tmp')
from blurmotion import read_images_bin, qrot, read_cam_focal

def pear(a,b):
    a=(a-a.mean())/(a.std()+1e-12); b=(b-b.mean())/(b.std()+1e-12); return float((a*b).mean())
def spear(a,b): return pear(rankdata(a),rankdata(b))

for scene in sys.argv[1:]:
    root=f'/mnt/d/avv/data/phase1/private_set2/{scene}/train'
    imgs=read_images_bin(os.path.join(root,'sparse/0/images.bin'))
    f_px,W,H=read_cam_focal(os.path.join(root,'sparse/0/cameras.bin'))
    names=sorted(imgs.keys())
    idx=np.array([int(re.findall(r'(\d+)',n)[-1]) for n in names])
    o=np.argsort(idx); names=[names[i] for i in o]; idx=idx[o]
    C={};R={}
    for nm in names:
        q,t=imgs[nm]; Rm=qrot(q); R[nm]=Rm; C[nm]=-Rm.T@t
    Cs=np.array([C[n] for n in names]); scale=np.median(np.linalg.norm(Cs-Cs.mean(0),axis=1))
    # per-unit-index motion, central difference over the FULL sequence (incl. test poses)
    mot={}
    for i,nm in enumerate(names):
        a=max(0,i-1); b=min(len(names)-1,i+1)
        di=idx[b]-idx[a]
        dR=R[names[a]].T@R[names[b]]
        ang=math.acos(max(-1,min(1,(np.trace(dR)-1)/2)))/max(di,1)
        dC=np.linalg.norm(C[names[b]]-C[names[a]])/max(di,1)
        mot[nm]=(f_px*ang, f_px*dC/max(scale,1e-9))
    have=[]; vol=[]; nvol=[]; hf=[]; prot=[]; ptr=[]
    for nm in names:
        p=os.path.join(root,'images',nm)
        im=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        g=im.astype(np.float32)/255.
        v=cv2.Laplacian(g,cv2.CV_32F).var()
        var=g.var()+1e-12
        # radial spectrum band ratio: high(0.3-0.5 nyq) / mid(0.1-0.2 nyq), content-normalised
        gs=cv2.resize(g,(512,512))
        F=np.abs(np.fft.fftshift(np.fft.fft2(gs*np.hanning(512)[:,None]*np.hanning(512)[None,:])))**2
        yy,xx=np.mgrid[0:512,0:512]; r=np.sqrt((yy-256)**2+(xx-256)**2)/256.
        mid=F[(r>0.10)&(r<0.20)].mean(); hi=F[(r>0.30)&(r<0.55)].mean()
        have.append(nm); vol.append(v); nvol.append(v/var); hf.append(hi/(mid+1e-30))
        prot.append(mot[nm][0]); ptr.append(mot[nm][1])
    vol=np.array(vol); nvol=np.array(nvol); hf=np.array(hf)
    prot=np.array(prot); ptr=np.array(ptr); tot=prot+ptr
    print(f'\n== {scene} n_photo={len(have)} n_poses={len(names)} f={f_px:.0f}px {W}x{H} idx_step_med={np.median(np.diff(idx)):.0f}')
    print(f'   px motion PER VIDEO FRAME: rot med={np.median(prot):.2f} tr med={np.median(ptr):.2f} tot med={np.median(tot):.2f} p95={np.percentile(tot,95):.2f}')
    for bl,b in [('VoL',vol),('VoL/var',nvol),('HF/MID',hf)]:
        print(f'   {bl:8s} p5={np.percentile(b,5):.4g} med={np.median(b):.4g} p95={np.percentile(b,95):.4g} spread(med/p5)={np.median(b)/np.percentile(b,5):.2f}')
        for ml,m in [('rot',prot),('tr',ptr),('tot',tot)]:
            print(f'      corr({bl},{ml}) pearson={pear(b,m):+.3f} spearman={spear(b,m):+.3f}')
        s=np.argsort(b); q=max(1,len(s)//4)
        print(f'      blurriest-quartile med tot_px={np.median(tot[s[:q]]):.2f}  sharpest-quartile med tot_px={np.median(tot[s[-q:]]):.2f}')
