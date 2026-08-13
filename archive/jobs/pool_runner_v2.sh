#!/bin/bash
# SCREEN-TIER claim-based pool. Jobs run at ~1/4 the iteration count (proportionally scaled
# refine_stop/noise_stop/lpips_from) -> ~4x faster triage. Job line format:
#   MODE|SCENE|NAME|EXTRA_ARGS      MODE = screen | full
# screen = fast triage (kills obvious duds fast, confirms obvious wins fast)
# full   = the real production recipe (used to CONFIRM a screen survivor before trusting it)
# Same atomic mkdir-claim pattern as pool_runner.sh (proven, no pgrep, no self-match risk).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
JOBS=/mnt/d/avv/phase2_screen_jobs.txt
CLAIMS=/mnt/d/avv/phase2_screen_jobs/claims
mkdir -p "$CLAIMS" /mnt/d/avv/phase2_screen_jobs/done

# FULL (production) recipes -- unchanged, used for MODE=full
CHA_FULL="--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000"
BON_FULL="--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"
TOW_FULL="--ut --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000"
# SCREEN recipes -- iters at 1/4, every schedule milestone scaled by the same 1/4 factor so the
# recipe's SHAPE (fraction of training in each phase) is preserved, just compressed in time.
# cap_max also reduced (proportional to iters -> proportional gaussian count -> faster per-step).
CHA_SCREEN="--iters 15000 --cap_max 2000000 --refine_stop 12500 --noise_stop 12500 --lpips_from 7500"
BON_SCREEN="--iters 7500  --cap_max 1250000 --refine_stop 3750  --noise_stop 2000  --lpips_from 3000"
TOW_SCREEN="--ut --iters 15000 --cap_max 2000000 --refine_stop 12500 --noise_stop 12500 --lpips_from 12500"

run_one() {  # $1 gpu $2 mode $3 scene $4 name $5 extra_args
  local G=$1 MODE=$2 SCENE=$3 N=$4 EXTRA=$5
  local ES=/mnt/d/avv/evalsplit/$SCENE M=/mnt/d/avv/tw_test/${SCENE}_${N}_${MODE} BASE RF=""
  case $SCENE in
    chair)   [[ $MODE == screen ]] && BASE=$CHA_SCREEN || BASE=$CHA_FULL;;
    bonsai)  [[ $MODE == screen ]] && BASE=$BON_SCREEN || BASE=$BON_FULL;;
    HCM0181) [[ $MODE == screen ]] && BASE=$TOW_SCREEN || BASE=$TOW_FULL; RF="--ut_render native";;
    *) echo "!! unknown scene $SCENE"; return 1;;
  esac
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
    --seed 42 --out $M $BASE $EXTRA 2>&1 | tail -2 || { echo "[$SCENE $N $MODE] TRAIN FAIL (gpu$G)"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv \
    --out $M/eval_render --png_dir $M/eval_png $RF 2>&1 | tail -1 || { echo "[$SCENE $N $MODE] RENDER FAIL"; return 1; }
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "${SCENE}_${N}_${MODE}"
}

worker() {
  local G=$1 ln
  while true; do
    ln=0; local claimed=0
    while IFS= read -r line || [ -n "$line" ]; do
      ln=$((ln+1)); [ -z "$line" ] && continue
      if mkdir "$CLAIMS/$ln" 2>/dev/null; then
        claimed=1
        IFS='|' read -r MODE SCENE N EXTRA <<< "$line"
        echo ">>> gpu$G claim job#$ln: [$MODE] $SCENE $N  [$EXTRA]  $(date +%H:%M)"
        run_one "$G" "$MODE" "$SCENE" "$N" "$EXTRA"
        touch "/mnt/d/avv/phase2_screen_jobs/done/${ln}_${SCENE}_${N}_${MODE}"
        break
      fi
    done < "$JOBS"
    if [ "$claimed" = "0" ]; then
      if [ -f /mnt/d/avv/phase2_screen_jobs.SEALED ]; then echo "=== gpu$G: sealed+empty, exiting ==="; return 0; fi
      sleep 20
    fi
  done
}

# wait for the FIRST (full-length) pool via ITS sentinel file (phase2_pool.DONE, touched by
# pool_runner.sh after both its workers exit) -- NOT pgrep (self-match bit us once already;
# sentinels are the house rule now).
while [ ! -f /mnt/d/avv/phase2_pool.DONE ]; do sleep 30; done
echo "=== SCREEN POOL starting (GPU0 only -- GPU1 dedicated to EMA production) $(date) ==="
worker 0 &
wait
echo "=== SCREEN POOL DONE $(date) ==="
touch /mnt/d/avv/phase2_screen.DONE
