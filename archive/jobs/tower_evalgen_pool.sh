#!/bin/bash
# Generate (held-out novel-view render, GT) pairs for all 5 set2 towers -- the training data the
# restoration head needs, per scene. Claim-based across both GPUs, each waits for its GPU to free.
# Per tower: make eval split (40 held out) -> train production recipe on train_sub -> render the
# 40 held-out poses natively -> delete ckpt. ~2h/tower, 10 GPU-h total, ~5h wall on 2 GPUs.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2
CLAIMS=/mnt/d/avv/evalgen_claims
mkdir -p "$CLAIMS"
TOWERS="HCM0421 HCM0539 HCM0540 HCM0644 HCM0674"

run_one() {
  local G=$1 T=$2 ES=/mnt/d/avv/evalsplit/$T M=/mnt/d/avv/evalgen/$T
  conda activate gsplat
  if [ ! -d "$ES/train_sub" ]; then
    echo ">>> gpu$G $T: creating eval split"
    python scripts/make_eval_split.py --scene $DATA/$T --out $ES --n_eval 40 2>&1 | tail -1 \
      || { echo "!!! $T SPLIT FAIL"; return 1; }
  fi
  echo ">>> gpu$G $T train $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
    --seed 42 --ut --out $M --ema_decay 0.999 \
    --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 2>&1 \
    | tail -2 || { echo "!!! $T TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 \
    | tail -1 || { echo "!!! $T RENDER FAIL"; return 1; }
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png \
    --gt_dir $ES/eval_gt --tag ${T}_evalgen
  rm -f $M/ckpt.pt
  echo "<<< gpu$G $T done $(date +%H:%M)"
  touch /mnt/d/avv/evalgen/${T}.DONE
}

worker() {
  local G=$1
  for T in $TOWERS; do
    if mkdir "$CLAIMS/$T" 2>/dev/null; then run_one "$G" "$T"; fi
  done
}

mkdir -p /mnt/d/avv/evalgen
gpu_free() {  # $1 = gpu index; free == under 1500 MiB in use (numeric, not regex)
  local m
  m=$(nvidia-smi --id=$1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-99999}" -lt 1500 ]
}
# each worker waits for ITS gpu to be free before starting (no stacking on a live trainer --
# the HCM0674 contention lesson)
( while ! gpu_free 0; do sleep 60; done; worker 0 ) &
( while ! gpu_free 1; do sleep 60; done; worker 1 ) &
wait
echo "=== TOWER EVALGEN POOL DONE $(date) ==="
touch /mnt/d/avv/tower_evalgen_pool.DONE
