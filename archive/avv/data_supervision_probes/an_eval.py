import sys, os, numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
def vol_l(p, maxside=1024):
    im=Image.open(p).convert('L'); s=maxside/max(im.size)
    if s<1: im=im.resize((int(im.size[0]*s),int(im.size[1]*s)), Image.LANCZOS)
    a=np.asarray(im,dtype=np.float32); return float(cv2.Laplacian(a,cv2.CV_32F).var()), float(a.mean())
def psnr(a,b): 
    m=np.mean((a.astype(np.float64)-b.astype(np.float64))**2); return 10*np.log10(255.0**2/m)

cfg={'chair':('/mnt/d/avv/evalsplit/chair/eval_gt','/mnt/d/avv/chair_eval/base60k/eval_png'),
     'bonsai':('/mnt/d/avv/evalsplit/bonsai/eval_gt','/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png')}
for sc,(gtd,rd) in cfg.items():
    if not os.path.isdir(rd): print('miss',rd); continue
    gts=sorted(os.listdir(gtd)); rows=[]
    for g in gts:
        stem=os.path.splitext(g)[0]; r=os.path.join(rd,stem+'.png')
        if not os.path.exists(r): continue
        A=np.asarray(Image.open(os.path.join(gtd,g)).convert('RGB'))
        B=np.asarray(Image.open(r).convert('RGB'))
        if A.shape!=B.shape: continue
        v,l=vol_l(os.path.join(gtd,g))
        rows.append((stem,psnr(A,B),v,l))
    arr=np.array([[r[1],r[2],r[3]] for r in rows])
    print(f"== {sc}: n={len(rows)} meanPSNR={arr[:,0].mean():.3f}")
    lv=np.log(arr[:,1])
    print(f"   corr(PSNR, log VoL_gt) = {np.corrcoef(arr[:,0],lv)[0,1]:.3f}")
    o=np.argsort(arr[:,1])
    k=max(4,len(o)//4)
    print(f"   blurriest quartile GT: PSNR {arr[o[:k],0].mean():.2f} (VoL {arr[o[:k],1].mean():.0f}) | sharpest quartile: PSNR {arr[o[-k:],0].mean():.2f} (VoL {arr[o[-k:],1].mean():.0f})")
    np.save(f'/tmp/ev_{sc}.npy',arr)
    with open(f'/tmp/evnames_{sc}.txt','w') as f: f.write('\n'.join(r[0] for r in rows))
