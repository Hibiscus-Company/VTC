import sys, os, numpy as np
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from PIL import Image
import cv2

R='/mnt/d/avv/data/phase1/private_set2'

def vol(p, maxside=1024):
    im=Image.open(p).convert('L')
    s=maxside/max(im.size)
    if s<1: im=im.resize((int(im.size[0]*s),int(im.size[1]*s)), Image.LANCZOS)
    a=np.asarray(im,dtype=np.float32)
    return float(cv2.Laplacian(a,cv2.CV_32F).var()), float(np.asarray(im,dtype=np.float32).mean())

for sc in ['chair','bonsai','HCM0421','HCM0644']:
    d=f'{R}/{sc}/train/images'
    names=sorted(os.listdir(d))
    vs=[];lum=[]
    for n in names:
        v,l=vol(os.path.join(d,n)); vs.append(v); lum.append(l)
    vs=np.array(vs); lum=np.array(lum)
    q=np.percentile(vs,[1,5,25,50,75,95,99])
    print(f"{sc}: N={len(vs)} VoL p1/p5/p25/med/p75/p95/p99 = {np.round(q,0)}  spread(p95/p5)={q[5]/q[1]:.1f}x  frac<0.5*med={np.mean(vs<0.5*np.median(vs)):.3f}")
    print(f"   lum mean {lum.mean():.1f} std {lum.std():.1f} range {lum.max()-lum.min():.1f}")
    np.save(f'/tmp/vol_{sc}.npy', np.array([vs,lum]))
    with open(f'/tmp/names_{sc}.txt','w') as f: f.write('\n'.join(names))
