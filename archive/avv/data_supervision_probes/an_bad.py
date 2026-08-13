import os, numpy as np, sys
sys.path.insert(0,'/mnt/c/Users/BKAI/an_plaza2/FastGS')
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
g='DJI_20241230093301_0003_V.JPG'
A=np.asarray(Image.open('/mnt/d/avv/evalsplit/HCM0421/eval_gt/'+g).convert('RGB'))
B=np.asarray(Image.open('/mnt/d/avv/evalgen/HCM0421/eval_png/'+os.path.splitext(g)[0]+'.png').convert('RGB'))
print('GT mean',A.reshape(-1,3).mean(0).round(1),'std',A.reshape(-1,3).std(0).round(1))
print('REN mean',B.reshape(-1,3).mean(0).round(1),'std',B.reshape(-1,3).std(0).round(1))
# where is the eval-frame pose relative to the train camera cloud?
sp='/mnt/d/avv/data/phase1/private_set2/HCM0421/train/sparse/0'
imgs=read_extrinsics_binary(sp+'/images.bin')
ondisk=set(os.listdir('/mnt/d/avv/evalsplit/HCM0421/train_sub/images'))
allon=set(os.listdir('/mnt/d/avv/data/phase1/private_set2/HCM0421/train/images'))
C={}; D={}
for im in imgs.values():
    Rm=qvec2rotmat(im.qvec); C[im.name]=-Rm.T@im.tvec; D[im.name]=Rm.T@np.array([0,0,1.0])
tc=np.array([C[n] for n in sorted(ondisk)]); td=np.array([D[n] for n in sorted(ondisk)])
allc=np.array([C[n] for n in sorted(allon)])
ctr=allc.mean(0); rad=np.linalg.norm(allc-ctr,axis=1)
for nm in [g,'DJI_20241230093733_0018_V.JPG']:
    d=np.linalg.norm(tc-C[nm],axis=1); ang=np.degrees(np.arccos(np.clip(td@D[nm],-1,1)))
    print(f"{nm}: dist to scene centre {np.linalg.norm(C[nm]-ctr):.3f} (train radius med {np.median(rad):.3f} max {rad.max():.3f}) "
          f"| nearest train-sub cam {d.min():.3f} (med NN among train {np.median(np.sort(np.linalg.norm(allc[:,None]-allc[None],axis=2),axis=1)[:,1]):.3f}) | min view-angle {ang.min():.1f} deg")
# how many FULL train frames are further out than median+3*mad
print("frames further from centre than 1.5x median radius:", int((rad>1.5*np.median(rad)).sum()), "/", len(rad))
