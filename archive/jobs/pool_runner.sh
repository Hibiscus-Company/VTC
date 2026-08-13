#!/bin/bash
# CLAIM-BASED job pool -- guarantees zero GPU idle. Two workers (one per GPU) each grab the
# next unclaimed line from /mnt/d/avv/phase2_jobs.txt when free, run it, repeat. New jobs can be
# APPENDED to phase2_jobs.txt any time; workers exit only when the list is fully claimed AND
# /mnt/d/avv/phase2_jobs.SEALED exists. Waits on config_batch2.DONE sentinel first (not pgrep).
# Job line format: SCENE|NAME|EXTRA_TRAIN_ARGS   (SCENE in chair|bonsai|HCM0181)
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
JOBS=/mnt/d/avv/phase2_jobs.txt
CLAIMS=/mnt/d/avv/phase2_jobs/claims
mkdir -p "$CLAIMS" /mnt/d/avv/phase2_jobs/done
CHA="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"
BON="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"
TOW="--ut --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"

run_one() {  # $1 gpu  $2 scene  $3 name  $4 extra_args
  local G=$1 SCENE=$2 N=$3 EXTRA=$4
  local ES=/mnt/d/avv/evalsplit/$SCENE M=/mnt/d/avv/tw_test/${SCENE}_$N BASE RF=""
  case $SCENE in
    chair) BASE=$CHA;; bonsai) BASE=$BON;; HCM0181) BASE=$TOW; RF="--ut_render native";;
    *) echo "!! unknown scene $SCENE"; return 1;;
  esac
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
    --seed 42 --out $M $BASE $EXTRA 2>&1 | tail -2 || { echo "[$SCENE $N] TRAIN FAIL (gpu$G)"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv \
    --out $M/eval_render --png_dir $M/eval_png $RF 2>&1 | tail -1 || { echo "[$SCENE $N] RENDER FAIL"; return 1; }
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "${SCENE}_$N"
}

worker() {
  local G=$1 ln=0 line
  while true; do
    ln=0; local claimed_something=0
    while IFS= read -r line || [ -n "$line" ]; do
      ln=$((ln+1))
      [ -z "$line" ] && continue
      if mkdir "$CLAIMS/$ln" 2>/dev/null; then      # atomic claim
        claimed_something=1
        local SCENE=${line%%|*} rest=${line#*|} N EXTRA
        N=${rest%%|*}; EXTRA=${rest#*|}
        echo ">>> gpu$G claim job#$ln: $SCENE $N  [$EXTRA]  $(date +%H:%M)"
        run_one "$G" "$SCENE" "$N" "$EXTRA"
        touch "/mnt/d/avv/phase2_jobs/done/${ln}_${SCENE}_${N}"
        break                                        # re-scan from top for next unclaimed
      fi
    done < "$JOBS"
    if [ "$claimed_something" = "0" ]; then           # nothing left to claim this pass
      if [ -f /mnt/d/avv/phase2_jobs.SEALED ]; then echo "=== gpu$G: pool sealed + empty, exiting ==="; return 0; fi
      sleep 30                                        # list not sealed -> wait for appended jobs
    fi
  done
}

while [ ! -f /mnt/d/avv/config_batch2.DONE ]; do sleep 60; done
echo "=== config_batch2 done, POOL RUNNER starting $(date) ==="
worker 0 &
worker 1 &
wait
echo "=== POOL RUNNER DONE $(date) ==="
touch /mnt/d/avv/phase2_pool.DONE
