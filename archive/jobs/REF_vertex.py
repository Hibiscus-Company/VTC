"""REFUTE-lens: the +0.0151 vertex is 100% interpolation. MEASURE t=0.21 and t=0.35 directly.
Same scenes/stems/oracle gains/encode as LENS3.py. Per-image paired deltas -> SE."""
import os, sys, io
import numpy as np, cv2, torch
sys.path.insert(0, "/mnt/d/avv/metric_probe")
import mlib, lpips as L
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(6)
dev="cuda"
GTD="/mnt/d/avv/data/phase1/public_set/{s}/test/images"
SCENES=[("HCM0181","/mnt/d/avv/prodharness/k4/png"),
        ("HCM0193","/mnt/d/avv/output/HCM0193_gsplatB9ut/test_poses_renders_png"),
        ("HCM0204","/mnt/d/avv/output/HCM0204_gsplatB9ut/test_poses_renders_png"),
        ("hcm0031","/mnt/d/avv/output/hcm0031_gsplatB9ut/test_poses_renders_png"),
        ("hcm0034","/mnt/d/avv/output/hcm0034_gsplatB9ut/test_poses_renders_png")]
NIMG=12; NRB=8; SIG=[0.8,1.6]
lp=L.LPIPS(net='vgg',verbose=False).to(dev).eval()
def apply_field(img8,field):
    H,W,_=img8.shape
    fu=cv2.resize(field,(W,H),interpolation=cv2.INTER_CUBIC)
    yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
    return np.clip(cv2.remap(img8.astype(np.float32)/255.0,(xx+fu[...,0]).astype(np.float32),
        (yy+fu[...,1]).astype(np.float32),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)*255.,0,255).astype(np.uint8)
def jpg(a):
    b=io.BytesIO()
    Image.fromarray(np.clip(a*255+0.5,0,255).astype(np.uint8)).save(b,"JPEG",quality=100,subsampling=2,optimize=True,progressive=True)
    b.seek(0); return np.asarray(Image.open(b).convert("RGB"),np.float32)/255.
def bands(img):
    out,cur=[],img
    for s in SIG:
        lo=cv2.GaussianBlur(img,(0,0),s); out.append(cur-lo); cur=lo
    return out,cur
def rmask(H,W):
    yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
    r=np.sqrt(((xx-W/2)/(W/2))**2+((yy-H/2)/(H/2))**2); r=r/r.max()
    return np.clip((r*NRB).astype(np.int32),0,NRB-1),r
def load(scene,rdir,st):
    FLD=np.load(f"/mnt/d/avv/fields/{scene}.npy")
    gd=GTD.format(s=scene); gmap={os.path.splitext(f)[0]:f for f in sorted(os.listdir(gd))}
    G8=mlib.load_u8(os.path.join(gd,gmap[st]))
    R8=apply_field(mlib.load_u8(os.path.join(rdir,st+".png")),FLD)
    return G8.astype(np.float32)/255., R8.astype(np.float32)/255.
STEMS={}
for scene,rdir in SCENES:
    gd=GTD.format(s=scene); gm=sorted(os.path.splitext(f)[0] for f in os.listdir(gd))
    s2=[s for s in gm if os.path.exists(os.path.join(rdir,s+".png"))]
    STEMS[scene]=s2[::max(1,len(s2)//NIMG)][:NIMG]
# PASS1: power only (need pw for g_en)
pw={s:np.zeros((len(SIG),NRB,2)) for s,_ in SCENES}
for scene,rdir in SCENES:
    for st in STEMS[scene]:
        Gf,Rf=load(scene,rdir,st); idx,_=rmask(*Gf.shape[:2])
        bR,_=bands(Rf); bG,_=bands(Gf)
        for k in range(len(SIG)):
            a=bR[k].mean(2); b=bG[k].mean(2)
            for j in range(NRB):
                m=idx==j
                pw[scene][k,j,0]+=float((a[m]**2).sum()); pw[scene][k,j,1]+=float((b[m]**2).sum())
    print("p1",scene,flush=True)
TS=[0.21,0.35,0.50]
ARMS=["base"]+[f"t{t:.2f}" for t in TS]
per={a:[] for a in ARMS}
for scene,rdir in SCENES:
    g_en=np.stack([np.sqrt(pw[scene][k,:,1]/pw[scene][k,:,0]) for k in range(len(SIG))])
    for st in STEMS[scene]:
        Gf,Rf=load(scene,rdir,st); idx,rn=rmask(*Gf.shape[:2]); bR,lo=bands(Rf)
        tg=torch.from_numpy(np.ascontiguousarray(Gf)).permute(2,0,1)[None].to(dev)
        V={"base":Rf}
        for t in TS:
            gp=1.0+t*(g_en-1.0); out=lo.copy()
            for k in range(len(SIG)):
                gm=np.interp(rn*NRB-0.5,np.arange(NRB),gp[k]).astype(np.float32)
                out=out+bR[k]*gm[...,None]
            V[f"t{t:.2f}"]=np.clip(out,0,1)
        for a,arr in V.items():
            tt=torch.from_numpy(np.ascontiguousarray(jpg(arr))).permute(2,0,1)[None].to(dev)
            with torch.no_grad():
                per[a].append([mlib.psnr(tt,tg),float(mlib.ssim(tt,tg)),float(lp(tt*2-1,tg*2-1).item())])
    print("p2",scene,flush=True)
P={a:np.array(v) for a,v in per.items()}
sc={a:100*(0.4*(1-P[a][:,2])+0.3*P[a][:,1]+0.3*P[a][:,0]/50.) for a in ARMS}
print("\n=== DIRECT MEASUREMENT AT THE CLAIMED VERTEX (n=%d paired) ==="%len(sc["base"]))
print(" parabola predicts: t=0.21 +0.0148 | t=0.35 +0.0083 | t=0.50 -0.0137(measured anchor)")
for a in ARMS:
    v=P[a].mean(0); d=sc[a]-sc["base"]
    print("  %-6s PSNR %8.4f SSIM %8.5f LPIPS %8.5f  dScore %+8.4f  paired SE %.4f  win %2d/%d"%(
        a,v[0],v[1],v[2],d.mean(),d.std(ddof=1)/np.sqrt(len(d)) if a!="base" else 0.0,(d>0).sum(),len(d)))
np.save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/REF_vertex.npy",np.stack([sc[a] for a in ARMS]))
print("DONEREF")
