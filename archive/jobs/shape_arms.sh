#!/bin/bash
# Shape-axis arms. All single-variable vs the shipped bonsai recipe, all on the eval split,
# all scored by bar.py with today's scorer. Controls: shipped-recipe singles 71.22-71.31,
# best single 71.9029, 6-member mean (THE BAR) 72.2644.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/bonsai
OUT=/mnt/d/avv/r36_shape
mkdir -p $OUT
BASE="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000 --seed 42"

arm(){ # $1 gpu  $2 tag  $3 extra-args
  local G=$1 T=$2 M=$OUT/$2; shift 2; local EX="$*"
  mkdir -p $M
  conda activate gsplat
  echo ">>> $T (gpu$G) START $(date +%H:%M)  [$EX]"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --out $M $BASE $EX 2>&1 \
    | grep -aE "Baked|Saved|Error|error" | tail -3
  [ -f $M/ckpt.pt ] || { echo "!!! $T NO CKPT"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  echo "--- census $T ---"
  python - "$M/ckpt.pt" <<'PY'
import sys,torch,numpy as np
c=torch.load(sys.argv[1],map_location="cpu",weights_only=False);sp=c["splats"]
S=torch.exp(sp["scales"]);o=torch.sigmoid(sp["opacities"]);S=S[o>0.05].double().numpy()
Ss=np.sort(S,axis=1)[:,::-1]
print(f"   N={len(S):,}  s2/s1 {np.median(Ss[:,1]/Ss[:,0]):.4f}  s2/s3 {np.median(Ss[:,1]/np.maximum(Ss[:,2],1e-30)):.1f}  s1/s3 {np.median(Ss[:,0]/np.maximum(Ss[:,2],1e-30)):.1f}")
PY
  python /home/bkai/.claude/jobs/1c9cf7e9/tmp/bar.py $M/eval_png
  rm -f $M/ckpt.pt
  echo "<<< $T DONE $(date +%H:%M)"; touch $OUT/$T.DONE
}

# GPU1 is free now -> the most direct test of the shape hypothesis: an existing, never-used flag
arm 1 aniso001 --aniso_reg 0.01 > $OUT/aniso001.log 2>&1 &
# wait for the scale_reg sweep to release GPU0
while pgrep -f "r35_scalereg" >/dev/null || [ ! -f /mnt/d/avv/r35_scalereg/ALL.DONE ]; do sleep 60; done
arm 0 sr001seed42 --scale_reg 0.01 > $OUT/sr001seed42.log 2>&1 &   # MATCHED-SEED control
wait
arm 0 regstop15k --reg_stop 15000 > $OUT/regstop15k.log 2>&1 &
arm 1 aniso0003  --aniso_reg 0.003 > $OUT/aniso0003.log 2>&1 &
wait
echo "=== SHAPE ARMS DONE $(date +%H:%M) ==="; touch $OUT/ALL.DONE
