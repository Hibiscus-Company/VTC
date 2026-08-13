#!/bin/bash
# Phase-2 wave-1 structural candidates (eval-split). Waits on the config_batch2 SENTINEL FILE
# (not pgrep -- that self-match bug cost us the overnight batch). Baselines cached: chair 69.37,
# bonsai 71.90, HCM0181 tower = absgrad0 74.9453.
# Candidates: min_opacity (dead-gaussian prune), aniso_reg (needle/sheet penalty), sky_dome (tower).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
rm -f /mnt/d/avv/phase2_wave1.DONE

CHA="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"
TOW="--ut --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"

vid_test() { local G=$1 SC=$2 N=$3; shift 3
  local ES=/mnt/d/avv/evalsplit/$SC M=/mnt/d/avv/tw_test/${SC}_$N
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images --seed 42 --out $M "$@" 2>&1 | tail -2 || { echo "$SC $N TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "${SC}_$N"; }
tow_test() { local G=$1 N=$2; shift 2
  local ES=/mnt/d/avv/evalsplit/HCM0181 M=/mnt/d/avv/tw_test/HCM0181_$N
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images --seed 42 --out $M "$@" 2>&1 | tail -2 || { echo "HCM0181 $N TRAIN FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "HCM0181_$N"; }

# wait for the config batch to finish via its sentinel file
while [ ! -f /mnt/d/avv/config_batch2.DONE ]; do sleep 60; done
echo "=== config_batch2 sentinel seen, starting phase2_wave1 $(date) ==="

# GPU0 lane (3 runs): chair min_opacity, chair aniso, tower sky_dome
( vid_test 0 chair minop02 $CHA --min_opacity 0.02
  vid_test 0 chair aniso01 $CHA --aniso_reg 0.1
  tow_test 0 skydome50k $TOW --sky_dome 50000
  echo "=== GPU0 phase2_wave1 done ===" ) &
# GPU1 lane (2 runs): tower min_opacity, tower aniso
( tow_test 1 minop02 $TOW --min_opacity 0.02
  tow_test 1 aniso01 $TOW --aniso_reg 0.1
  echo "=== GPU1 phase2_wave1 done ===" ) &
wait
echo "=== PHASE2_WAVE1 DONE $(date) ==="
touch /mnt/d/avv/phase2_wave1.DONE
