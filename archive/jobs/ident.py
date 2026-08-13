import os, numpy as np
from PIL import Image
ROOT='/mnt/d/avv/output'
vars_=sorted([d for d in os.listdir(ROOT) if d.startswith('HCM0181_') and os.path.isdir(os.path.join(ROOT,d,'test_poses_renders_png'))])
k4d='/mnt/d/avv/prodharness/k4/png'
fs=sorted(os.listdir(k4d))
picks=[fs[0],fs[10],fs[25]]
X=[];Y=[]
for f in picks:
    y=np.asarray(Image.open(os.path.join(k4d,f)),dtype=np.float64)/255.
    cols=[]
    for v in vars_:
        p=os.path.join(ROOT,v,'test_poses_renders_png',f)
        cols.append(np.asarray(Image.open(p),dtype=np.float64).ravel()/255.)
    X.append(np.stack(cols,1)[::37]); Y.append(y.ravel()[::37])
X=np.concatenate(X); Y=np.concatenate(Y)
w,res,rank,sv=np.linalg.lstsq(X,Y,rcond=None)
order=np.argsort(-w)
for i in order[:8]:
    print(f'{vars_[i]:32s} {w[i]:+.4f}')
pred=X@w
print('resid rms', np.sqrt(((pred-Y)**2).mean()))
