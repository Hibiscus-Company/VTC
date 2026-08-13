"""Is the harness's lam=1.0 optimum the SAME OPERATOR as our private towers' lam=1.0?
Energy restoration is POOL-DEPENDENT. Compare DELIVERED BOOST at lam=1.0 across pools."""
import os,sys,numpy as np,torch
from PIL import Image
HERE="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,HERE)
from lapfuse import lap_pyr, boxf, _K
Image.MAX_IMAGE_PIXELS=None
NLEV=1; WIN=3; CLAMP=4.0; NS=6
ROOT="/mnt/d/avv/output"
ORDER=["gsplatB11ut60k","sh3","m31b_nolpips","m31b_taillpips","gsplatB10ut8M","gsplatB12ut8Ms7",
       "e17visnorm","e15ceil95","gsplatB9ut","gsplatB8pure","gsplatB2","gsplatB4warm"]
def harness(tag,k): return [f"{ROOT}/{tag}_{m}/test_poses_renders_png" for m in ORDER[:k]]
def tower(T):
    return ["/mnt/d/avv/r2r9/models/%s_ut7/test_png"%T,"/mnt/d/avv/r2r9/models/%s_ut13/test_png"%T,
            "/mnt/d/avv/r2r9/models/%s_ut42/test_png"%T,"/mnt/d/avv/r2r9/models/%s_ut77/test_png"%T,
            "/mnt/d/avv/r22_seed101/%s/test_png"%T,"/mnt/d/avv/r25_mip3d/%s/test_png"%T,
            "/mnt/d/avv/r28_members/%s/test_png"%T]
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
SC=[("HARNESS HCM0181 k=10",harness("HCM0181",10)),("HARNESS HCM0181 k=8",harness("HCM0181",8)),
    ("HARNESS HCM0193 k=10",harness("HCM0193",10)),
    ("PRIV HCM0421 k=7",tower("HCM0421")),("PRIV HCM0539 k=7",tower("HCM0539")),
    ("PRIV HCM0644 k=7",tower("HCM0644")),("PRIV HCM0674 k=7",tower("HCM0674"))]
print(f"{'pool':>22} {'k':>3} {'disagree/255':>12} {'boost%':>8} {'wboost%':>8} {'%clamp':>7}")
out={}
for name,dirs in SC:
    dirs=[d for d in dirs if os.path.isdir(d)]
    if len(dirs)<4: print(f"{name:>22}  MISSING ({len(dirs)} dirs)"); continue
    ex=lambda d: set(f[:-4] for f in os.listdir(d) if f.endswith(".png"))
    stems=sorted(set.intersection(*[ex(d) for d in dirs]))
    stems=stems[::max(1,len(stems)//NS)][:NS]
    k=len(dirs); acc=[]
    for s in stems:
        mem=[ld(os.path.join(d,s+".png")) for d in dirs]
        ens=torch.stack(mem).mean(0)
        dis=float(torch.stack([ (m-ens).abs().mean() for m in mem]).mean())*255
        L0=lap_pyr(ens,NLEV,_K)[0][0]
        Eb=boxf((L0**2).sum(1,keepdim=True),WIN)
        V=sum(boxf(((lap_pyr(m,NLEV,_K)[0][0]-L0)**2).sum(1,keepdim=True),WIN) for m in mem)/k*(k/(k-1.0))
        r=torch.sqrt(1.0+V/(Eb+1e-10)); rc=r.clamp(max=CLAMP)
        w=L0.abs().sum(1,keepdim=True)
        acc.append((dis,float((rc-1).mean())*100,float(((rc-1)*w).sum()/w.sum())*100,
                    float((r>=CLAMP).float().mean())*100))
    a=np.array(acc).mean(0); out[name]=a
    print(f"{name:>22} {k:3d} {a[0]:12.3f} {a[1]:8.2f} {a[2]:8.2f} {a[3]:7.2f}",flush=True)
h=[v for kk,v in out.items() if kk.startswith("HARNESS HCM0181 k=10")]
p=[v for kk,v in out.items() if kk.startswith("PRIV")]
if h and p:
    hb=h[0][2]; pb=np.mean([x[2] for x in p])
    print(f"\nweighted boost at lam=1.0:  harness {hb:.2f}%   private towers {pb:.2f}%   ratio {hb/pb:.2f}x")
    print(f"lambda that would match the harness-proven delivered boost on private towers: ~{hb/pb:.2f}")
