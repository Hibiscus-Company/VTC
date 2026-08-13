#!/bin/bash
# ADOPTER for the two orphaned rollout trainers (HCM0644 pid 254712 gpu1, HCM0674 pid 254586
# gpu0). Their parent subshells died in the 09:20 kill-cleanup, so nobody will run their render
# step or touch their DONE markers when training exits -- which would strand the makeup script
# (waits on those markers) forever. This waits for each PID to exit (kill -0 poll; can't `wait`
# on a non-child), sanity-checks the ckpt, renders, writes train_args, touches the marker.
# GPU assignment preserved: HCM0644 renders on gpu1, HCM0674 on gpu0 -- and the marker is only
# touched AFTER the render finishes, so the makeup trainer never overlaps the render on its GPU.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
PRI=/mnt/d/avv/data/phase1/private_set2
TOW_ARGS="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"

adopt() {
  local PID=$1
  local G=$2
  local S=$3
  local M=/mnt/d/avv/r17/${S}_ut7_ema999
  while kill -0 "$PID" 2>/dev/null; do sleep 30; done
  echo "=== [$S] trainer (pid $PID) exited $(date) ==="
  if [ ! -f "$M/ckpt.pt" ]; then
    echo "!!! [$S] ckpt.pt MISSING after trainer exit -- training crashed, NOT touching marker"
    return 1
  fi
  conda activate gsplat
  echo "--ut --seed 7 --ema_decay 0.999 $TOW_ARGS" > $M/train_args.txt
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py \
    --ckpt $M/ckpt.pt --csv $PRI/$S/test/test_poses.csv \
    --out $M/test_render --png_dir $M/test_png --ut_render native 2>&1 | tail -1
  local n=$(ls $M/test_png 2>/dev/null | wc -l)
  echo "=== [$S] ADOPTED-COMPLETE $(date), $n test renders ==="
  if [ "$n" -ne 60 ]; then echo "!!! [$S] expected 60 renders, got $n -- NOT touching marker"; return 1; fi
  touch /mnt/d/avv/tower_ema_rollout_${S}.DONE
}

adopt 254712 1 HCM0644 &
adopt 254586 0 HCM0674 &
wait
echo "=== ORPHAN ADOPTER DONE $(date) ==="
