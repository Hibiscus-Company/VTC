import sys, os, numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
def vl(p, maxside=1024):
    im=Image.open(p).convert('L'); s=maxside/max(im.size)
    if s<1: im=im.resize((int(im.size[0]*s),int(im.size[1]*s)), Image.LANCZOS)
    return float(cv2.Laplacian(np.asarray(im,dtype=np.float32),cv2.CV_32F).var())
def psnr(a,b):
    m=np.mean((a.astype(np.float64)-b.astype(np.float64))**2); return 10*np.log10(255.0**2/m)
cfg={'chair':('/mnt/d/avv/evalsplit/chair/eval_gt','/mnt/d/avv/chair_eval/base60k/eval_png'),
     'bonsai':('/mnt/d/avv/evalsplit/bonsai/eval_gt','/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png'),
     'HCM0421':('/mnt/d/avv/evalsplit/HCM0421/eval_gt',None)}
import glob
# find a tower eval render dir
cands=glob.glob('/mnt/d/avv/*HCM0421*/**/eval_png',recursive=True)+glob.glob('/mnt/d/avv/**/HCM0421*/eval_png',recursive=True)
print('tower eval render dirs:',cands[:6])
for sc,(gtd,rd) in cfg.items():
    if rd is None or not os.path.isdir(rd): continue
    rows=[]
    for g in sorted(os.listdir(gtd)):
        stem=os.path.splitext(g)[0]; r=os.path.join(rd,stem+'.png')
        if not os.path.exists(r): continue
        A=np.asarray(Image.open(os.path.join(gtd,g)).convert('RGB')); B=np.asarray(Image.open(r).convert('RGB'))
        rows.append((vl(os.path.join(gtd,g)), vl(r), psnr(A,B)))
    a=np.array(rows)
    o=np.argsort(a[:,0]); k=max(3,len(o)//4)
    print(f"== {sc} n={len(a)}")
    print(f"   GT VoL   blurQ {a[o[:k],0].mean():8.0f}  sharpQ {a[o[-k:],0].mean():8.0f}   (ratio {a[o[-k:],0].mean()/a[o[:k],0].mean():.1f}x)")
    print(f"   REN VoL  blurQ {a[o[:k],1].mean():8.0f}  sharpQ {a[o[-k:],1].mean():8.0f}   (ratio {a[o[-k:],1].mean()/a[o[:k],1].mean():.1f}x)")
    print(f"   ren/gt VoL ratio: blurQ {a[o[:k],1].mean()/a[o[:k],0].mean():.2f}  sharpQ {a[o[-k:],1].mean()/a[o[-k:],0].mean():.2f}")
    print(f"   PSNR     blurQ {a[o[:k],2].mean():.2f}   sharpQ {a[o[-k:],2].mean():.2f}")
