#!/bin/bash
# scale_reg sweep on bonsai -- an axis that has NEVER been swept (not in the closed list).
# Motivation: bonsai gaussians are 7x flatter than a tower's (s2/s3 = 355 vs 51.5) on the scene
# whose content is the LEAST surface-like in the set. Whatever drives that flatness, scale_reg is
# the cheapest knob touching it. NOTE the mechanism is NOT the usual "L1 zeroes the smallest axis":
# the penalty is scale_reg*exp(log_scales).mean(), so the gradient is lambda*s -- strongest on the
# LARGEST axis. It pushes toward isotropy and leaves the small axis alone. Sweeping it therefore
# tests whether flatness is regulariser-driven at all.
# Everything else = the shipped bonsai recipe, verbatim. Single variable.
# Controls: shipped-recipe singles on this split score 71.22-71.31 (K4_pC_*), best single 71.9029,
#           6-member mean 72.2644 (today's scorer). A single arm must be read against the SINGLES.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai
OUT=/mnt/d/avv/r35_scalereg
mkdir -p $OUT
BASE="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"

arm(){ # $1 gpu  $2 scale_reg
  local G=$1 SR=$2 M=$OUT/sr$2
  mkdir -p $M
  conda activate gsplat
  echo ">>> scale_reg=$SR (gpu$G) START $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --out $M --seed 42 $BASE --scale_reg $SR 2>&1 \
    | grep -aE "Baked|Saved|Error|error" | tail -3
  [ -f $M/ckpt.pt ] || { echo "!!! sr$SR NO CKPT"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  echo "--- shape census sr=$SR ---"
  python - "$M/ckpt.pt" <<'PY'
import sys,torch,numpy as np
c=torch.load(sys.argv[1],map_location="cpu",weights_only=False);sp=c["splats"]
S=torch.exp(sp["scales"]);o=torch.sigmoid(sp["opacities"]);S=S[o>0.05].double().numpy()
Ss=np.sort(S,axis=1)[:,::-1]
print(f"   N={len(S):,}  s2/s1 med {np.median(Ss[:,1]/Ss[:,0]):.4f}   s2/s3 med {np.median(Ss[:,1]/np.maximum(Ss[:,2],1e-30)):.1f}")
PY
  python /home/bkai/.claude/jobs/1c9cf7e9/tmp/bar.py $M/eval_png
  rm -f $M/ckpt.pt
  echo "<<< sr$SR DONE $(date +%H:%M)"
  touch $OUT/sr$SR.DONE
}

arm 0 0     > $OUT/sr0.log 2>&1 &
sleep 40
arm 1 0.003 > $OUT/sr0003.log 2>&1 &
wait
arm 0 0.03  > $OUT/sr003.log 2>&1     # 3x the default, the other direction
echo "=== SWEEP DONE $(date +%H:%M) ==="; touch $OUT/ALL.DONE
