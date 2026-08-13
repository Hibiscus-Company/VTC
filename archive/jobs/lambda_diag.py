"""SET THE VIDEO-SCENE LAMBDA BY MEASUREMENT, NOT GUESS -- no GT required.
The energy operator is r = sqrt(1 + (k/(k-1)) * V/E), driven ONLY by member disagreement V over
local energy E. Both are computable without ground truth. Towers are KNOWN to want lam=1.0
(harness-swept: 0.75 +0.1756 / 1.0 +0.1883 / 1.25 +0.1820 / 1.5 +0.1570). So: compare the r-map
statistics of chair and bonsai against the towers. If a video scene's map looks like a tower's,
lam~1.0 is right for it and shipping lam=0 is leaving score on the table. If it SATURATES at the
r<=4 clamp, the operator is over-boosting there and lam must come down.
Known prior: r28's post-mortem found bonsai saturating at 4.5% of pixels. Chair was never checked."""
import os, sys, time
import numpy as np, torch
from PIL import Image
HERE="/home/bkai/.claude/jobs/1c9cf7e9/tmp"
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS"); sys.path.insert(0,HERE)
from lapfuse import lap_pyr, boxf, _K
Image.MAX_IMAGE_PIXELS=None
NLEV=5; WIN=3; CLAMP=4.0
CHAIR=["/mnt/d/avv/r14/chair_aa42/test_png","/mnt/d/avv/r14/chair_aa7/test_png",
       "/mnt/d/avv/r14/chair_aa13/test_png","/mnt/d/avv/r17/chair_ema099_seed42/test_png",
       "/mnt/d/avv/r17/chair_ema099_seed7/test_png","/mnt/d/avv/r17/chair_ema099_seed13/test_png",
       "/mnt/d/avv/r17/chair_depth_seed42/test_png","/mnt/d/avv/r28_members/chair/test_png"]
BONSAI=["/mnt/d/avv/r14/bonsai_aa42/test_png","/mnt/d/avv/r14/bonsai_aa7/test_png",
        "/mnt/d/avv/r14/bonsai_aa13/test_png","/mnt/d/avv/r24_bonsai/aa101/test_png",
        "/mnt/d/avv/r24_bonsai/aa202/test_png","/mnt/d/avv/r24_bonsai/aa303/test_png",
        "/mnt/d/avv/r28_members/bonsai/test_png"]
def tower(T):
    return ["/mnt/d/avv/r2r9/models/%s_ut7/test_png"%T,"/mnt/d/avv/r2r9/models/%s_ut13/test_png"%T,
            "/mnt/d/avv/r2r9/models/%s_ut42/test_png"%T,"/mnt/d/avv/r2r9/models/%s_ut77/test_png"%T,
            "/mnt/d/avv/r22_seed101/%s/test_png"%T,"/mnt/d/avv/r25_mip3d/%s/test_png"%T,
            "/mnt/d/avv/r28_members/%s/test_png"%T]
SCENES=[("HCM0421(tower,lam=1.0)",tower("HCM0421")),("HCM0644(tower,lam=1.0)",tower("HCM0644")),
        ("chair (SHIPS lam=0)",CHAIR),("bonsai(SHIPS lam=0)",BONSAI)]
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
NS=8
print(f"{'scene':>22} {'k':>3} {'mean r':>8} {'p99 r':>8} {'%clamped':>9} {'mean V/E':>9} {'boost%':>8}")
for name,dirs in SCENES:
    dirs=[d for d in dirs if os.path.isdir(d)]
    stems=sorted(f for f in os.listdir(dirs[0]) if f.endswith(".png"))
    stems=[s for s in stems if all(os.path.exists(os.path.join(d,s)) for d in dirs)]
    stems=stems[::max(1,len(stems)//NS)][:NS]
    k=len(dirs); acc=[]
    for s in stems:
        mem=[ld(os.path.join(d,s)) for d in dirs]
        ens=torch.stack(mem).mean(0)
        L0=lap_pyr(ens,NLEV,_K)[0][0]
        Eb=boxf((L0**2).sum(1,keepdim=True),WIN)
        V=sum(boxf(((lap_pyr(m,NLEV,_K)[0][0]-L0)**2).sum(1,keepdim=True),WIN) for m in mem)/k*(k/(k-1.0))
        ratio=(V/(Eb+1e-10))
        r=torch.sqrt(1.0+ratio)
        rc=r.clamp(max=CLAMP)
        acc.append((float(rc.mean()), float(np.percentile(r.numpy(),99)),
                    float((r>=CLAMP).float().mean())*100., float(ratio.mean()),
                    float((rc-1).mean())*100.))
    a=np.array(acc).mean(0)
    print(f"{name:>22} {k:3d} {a[0]:8.4f} {a[1]:8.3f} {a[2]:9.3f} {a[3]:9.4f} {a[4]:8.3f}", flush=True)
print("\nREAD: towers want lam=1.0. A video scene whose r-map RESEMBLES a tower's should also want")
print("      lam~1.0 -> shipping it at 0 costs score. A SATURATING map (high %clamped) means the")
print("      operator over-boosts there and lam must stay low.")
