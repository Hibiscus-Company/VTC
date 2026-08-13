#!/bin/bash
# Strategy-audit moves V1 + T1, chained after the r14 trainings free the GPUs.
#   V1 kill-test (~20 min GPU): cross-family video ensemble on EVAL HOLES — pixel-mean of
#     existing aa + UT eval renders at w_aa in {1.0, 0.65, 0.5, 0.35}, exact metric.
#     Bar: >= +0.3/scene over the aa single to justify a video-swap slot.
#   T1 (~8h, both GPUs): third UT seed (13) on the 5 towers at 60k/8M -> test renders.
#     Proven 1/N ensemble mechanism; ships later as an all-tower swap after LB A/B.
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2

# wait for ALL training (r14 chain) to clear the GPUs; compose/zip afterwards is CPU-only
while pgrep -f "train_gsplat.py" >/dev/null; do sleep 120; done
sleep 60

conda activate fastgs2
python - <<'PY'
import numpy as np, os, sys
from PIL import Image
sys.path.insert(0,"/mnt/c/Users/BKAI/an_plaza2/FastGS")
import torch
from utils.loss_utils import ssim as repo_ssim
import lpips as lpips_pkg
Image.MAX_IMAGE_PIXELS=None
dev="cuda"
vgg=lpips_pkg.LPIPS(net="vgg").to(dev).eval()
def load(p): return np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255
CASES={"chair":("/mnt/d/avv/chair_eval/K3_noUT_aa/eval_png","/mnt/d/avv/chair_eval/lpearly60k/eval_png","/mnt/d/avv/evalsplit/chair/eval_gt"),
       "bonsai":("/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png","/mnt/d/avv/bonsai_perc/pC_lpearly/eval_png","/mnt/d/avv/evalsplit/bonsai/eval_gt")}
for sc,(aad,utd,gtd) in CASES.items():
    gt={os.path.splitext(f)[0]:f for f in os.listdir(gtd)}
    fs=[f for f in sorted(os.listdir(aad)) if os.path.splitext(f)[0] in gt and os.path.isfile(os.path.join(utd,f))]
    with torch.no_grad():
        for w in (1.0,0.65,0.5,0.35):
            P=S=L=0.0;n=0
            for f in fs:
                a=load(os.path.join(aad,f)); u=load(os.path.join(utd,f)); g=load(os.path.join(gtd,gt[os.path.splitext(f)[0]]))
                if a.shape!=g.shape or u.shape!=g.shape: continue
                m=np.clip(w*a+(1-w)*u,0,1)
                r=torch.from_numpy(m).permute(2,0,1).unsqueeze(0).to(dev); t=torch.from_numpy(g).permute(2,0,1).unsqueeze(0).to(dev)
                mse=((r-t)**2).mean().item()
                P+=10*np.log10(1/max(mse,1e-12)); S+=float(repo_ssim(r,t)); L+=float(vgg(r*2-1,t*2-1)); n+=1
            P,S,L=P/n,S/n,L/n
            print(f"V1 {sc:7s} w_aa={w:.2f} n={n:3d} PSNR {P:7.4f} SSIM {S:.4f} LPIPS {L:.4f} SCORE {100*(0.4*(1-L)+0.3*S+0.3*min(P/50,1)):.4f}")
PY
echo "=== V1 SWEEP DONE ==="

# ---- T1: seed-13 tower members
t1() {  # $1 gpu, rest scenes
  local G=$1; shift
  conda activate gsplat
  for s in "$@"; do
    local M=/mnt/d/avv/r2r9/models/${s}_ut13
    CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
      --source $PRI/$s/train --images images --ut --seed 13 --out $M \
      --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 2>&1 | tail -1
    echo "--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000" > $M/train_args.txt
    CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
      --ckpt $M/ckpt.pt --csv $PRI/$s/test/test_poses.csv \
      --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 | tail -1
    echo "=== T1 [$s] seed13 DONE (gpu$G) ==="
  done
}
( t1 0 HCM0421 HCM0540 HCM0674 ) &
( t1 1 HCM0539 HCM0644 ) &
wait
echo "=== T1 DONE ==="
