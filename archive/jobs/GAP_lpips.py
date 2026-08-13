"""LPIPS-vgg half of the TRAIN-vs-EVAL gap. CPU only (live sweep arms own both GPUs).
Matched n=28 per side per scene so the two gaps are directly comparable."""
import os,sys,json,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch, lpips as lpips_pkg
torch.set_num_threads(os.cpu_count() or 8)
vgg=lpips_pkg.LPIPS(net="vgg").eval()
def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
def run(rd,gd,tag,limit=28):
    gt={os.path.splitext(f)[0]:f for f in os.listdir(gd)}
    r=[f for f in sorted(os.listdir(rd)) if os.path.splitext(f)[1].lower() in(".png",".jpg",".jpeg")]
    r=[f for f in r if os.path.splitext(f)[0] in gt]
    if limit and len(r)>limit:
        r=[r[i] for i in sorted(set(np.linspace(0,len(r)-1,limit).round().astype(int)))]
    L=[]
    with torch.no_grad():
        for f in r:
            s=os.path.splitext(f)[0]
            a=load(os.path.join(rd,f)); b=load(os.path.join(gd,gt[s]))
            L.append(float(vgg(a*2-1,b*2-1).item()))
    L=np.array(L)
    print(f"{tag:28s} n={len(L):3d}  LPIPSvgg {L.mean():.4f} (sd {L.std(ddof=1):.4f}, sem {L.std(ddof=1)/np.sqrt(len(L)):.4f})",flush=True)
    return L
ES="/mnt/d/avv/evalsplit"; BB="/mnt/d/avv/blurbound"; TW="/mnt/d/avv/tw_test"
out={}
for sc,tr,trg,ev,evg in [("bonsai",f"{BB}/bonsai/train_png",f"{ES}/bonsai/train_sub/images",f"{TW}/bonsai_ema099/eval_png",f"{ES}/bonsai/eval_gt"),
                         ("chair", f"{BB}/chair/train_png", f"{ES}/chair/train_sub/images", f"{TW}/chair_ema099/eval_png", f"{ES}/chair/eval_gt")]:
    a=run(tr,trg,f"{sc} TRAIN (in-sample)"); b=run(ev,evg,f"{sc} EVAL (held-out)")
    d=b.mean()-a.mean(); se=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
    print(f"  >>> {sc} LPIPS GAP (eval-train) {d:+.5f} +/- {se:.5f}   [worth {-40*d:+.3f} score pts]\n",flush=True)
    out[sc]={"train":a.tolist(),"eval":b.tolist()}
json.dump(out,open("/home/bkai/.claude/jobs/1c9cf7e9/tmp/GAP_lpips.json","w"))
print("DONE")
