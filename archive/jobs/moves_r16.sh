#!/bin/bash
# NO set -u — conda activate/deactivate hooks reference unbound vars (NVCC_*, _CONDA_*);
# three deaths from that. pipefail + explicit || checks instead.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2

# ---- B1: eps2d video A/B on eval holes (both GPUs). aa base (no --ut), sweep eps2d.
b1() {  # $1 gpu $2 scene $3 name $4 eps  rest = recipe args
  local G=$1 SC=$2 N=$3 EPS=$4; shift 4
  local ES=/mnt/d/avv/evalsplit/$SC M=/mnt/d/avv/${SC}_eval/$N
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --seed 42 --out $M --eps2d $EPS "$@" \
    || { echo "B1 $SC $N TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "${SC}_$N"
}
BON="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"
CHA="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"
# GPU0: bonsai eps {0.2,0.1} (aa baseline eps=0.3 already known = K1 71.90)
( b1 0 bonsai eps20 0.2 $BON
  b1 0 bonsai eps10 0.1 $BON
  # then T1 towers on GPU0
  conda activate gsplat
  for s in HCM0421 HCM0540 HCM0674; do
    M=/mnt/d/avv/r2r9/models/${s}_ut13
    CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py --source $PRI/$s/train --images images --ut --seed 13 \
      --out $M --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 2>&1 | tail -1 || { echo "T1 $s FAIL"; continue; }
    echo "--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000" > $M/train_args.txt
    CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $PRI/$s/test/test_poses.csv \
      --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 | tail -1
    echo "=== T1 [$s] DONE ==="
  done ) &
# GPU1: chair eps {0.2,0.1}, then T1 remaining towers
( b1 1 chair eps20 0.2 $CHA
  b1 1 chair eps10 0.1 $CHA
  conda activate gsplat
  for s in HCM0539 HCM0644; do
    M=/mnt/d/avv/r2r9/models/${s}_ut13
    CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py --source $PRI/$s/train --images images --ut --seed 13 \
      --out $M --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 2>&1 | tail -1 || { echo "T1 $s FAIL"; continue; }
    echo "--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000" > $M/train_args.txt
    CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $PRI/$s/test/test_poses.csv \
      --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 | tail -1
    echo "=== T1 [$s] DONE ==="
  done ) &
wait
echo "=== MOVES_R16 DONE ==="
