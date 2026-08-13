#!/bin/bash
# Fires when the cap-8M eval arm lands. Scores it RAW (no encode) on the same 28 held-out frames
# the 72.013 six-member-ensemble baseline was measured on. Bar = 72.013 + 0.30 single-seed noise
# floor = 72.31 to REPLACE the ensemble; below that it can still be evaluated as a 7th member.
until [ -f /mnt/d/avv/r33_bonsai/eval_s42.DONE ]; do sleep 30
  pgrep -f train_gsplat >/dev/null || pgrep -f render_gsplat >/dev/null || { echo "arms gone without DONE"; break; }
done
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
cd /home/bkai/.claude/jobs/1c9cf7e9/tmp
OMP_NUM_THREADS=6 python - <<'PY'
import os,sys,numpy as np,torch
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
dev="cpu"; torch.set_num_threads(6)
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
ES="/mnt/d/avv/evalsplit/bonsai"; NEW="/mnt/d/avv/r33_bonsai/eval_s42/eval_png"
B="/mnt/d/avv/bonsai_eval"
OLD=[d for d in sorted(os.listdir(B)) if os.path.isdir(f"{B}/{d}/eval_png") and d!="K2_clip_full"]
gt={os.path.splitext(f)[0]:f"{ES}/eval_gt/{f}" for f in os.listdir(f"{ES}/eval_gt")}
ld=lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).unsqueeze(0)
sc=lambda p,s,l:100*(0.4*(1-l)+0.3*s+0.3*p/50)
if not os.path.isdir(NEW): print("NO RENDERS at",NEW); raise SystemExit
st=sorted(s for s in gt if os.path.exists(f"{NEW}/{s}.png") and all(os.path.exists(f"{B}/{a}/eval_png/{s}.png") for a in OLD))
print(f"n={len(st)} frames, {len(OLD)} old members\n")
def score(get):
    P=S=L=0.
    for s in st:
        o=get(s); g=ld(gt[s])
        if g.shape!=o.shape: g=torch.nn.functional.interpolate(g,size=o.shape[-2:],mode="bilinear",align_corners=False)
        P+=10*np.log10(1/max(((o-g)**2).mean().item(),1e-12)); S+=float(repo_ssim(o,g)); L+=float(vgg(o*2-1,g*2-1).mean())
    n=len(st); return P/n,S/n,L/n
cases=[("NEW cap8M alone",lambda s: ld(f"{NEW}/{s}.png")),
       ("OLD 6-member mean",lambda s: torch.stack([ld(f"{B}/{a}/eval_png/{s}.png") for a in OLD]).mean(0)),
       ("OLD6 + NEW (7-way)",lambda s: torch.stack([ld(f"{B}/{a}/eval_png/{s}.png") for a in OLD]+[ld(f"{NEW}/{s}.png")]).mean(0))]
print(f"{'arm':>20} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'SCORE':>8}")
res={}
for nm,f in cases:
    P,S,L=score(f); res[nm]=sc(P,S,L)
    print(f"{nm:>20} {P:7.3f} {S:7.4f} {L:7.4f} {res[nm]:8.3f}",flush=True)
base=res["OLD 6-member mean"]
print(f"\nbaseline (what ships) {base:.3f}   replace-bar {base+0.30:.3f}")
for nm in ("NEW cap8M alone","OLD6 + NEW (7-way)"):
    d=res[nm]-base
    print(f"  {nm:>20}: {d:+.3f} scene pts = {d/7:+.4f} LB   {'PASS' if d>0.30 else 'below bar'}")
PY
