#!/bin/bash
# Claim-based depth-prior pool -- keeps BOTH GPUs saturated the instant the current bonsai/tower
# depth runs free them (no idle gap). Each worker: precompute Depth-Anything (if missing) ->
# train eval-split with --depth_prior -> render eval -> score. Job file appendable live.
# Job line: SCENE|NAME|WEIGHT|UTFLAG   (UTFLAG = "--ut" for towers, "" for video)
# Gated on BOTH current runs finishing (depth_bonsai.DONE + depth_tower.DONE) so it never
# stacks on a busy GPU (HCM0674 lesson).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
JOBS=/mnt/d/avv/depth_pool_jobs.txt
CLAIMS=/mnt/d/avv/depth_pool/claims
mkdir -p "$CLAIMS" /mnt/d/avv/depth_pool/done

recipe() {  # scene -> eval-split recipe (matches production per-scene)
  case $1 in
    chair)  echo "--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 30000";;
    bonsai) echo "--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000";;
    *)      echo "--ut --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000 --lpips_from 50000";;  # towers
  esac
}

run_one() {  # $1 gpu $2 scene $3 name $4 weight $5 utflag
  local G=$1 SC=$2 N=$3 W=$4 UT=$5
  local ES=/mnt/d/avv/evalsplit/$SC M=/mnt/d/avv/depth/${SC}_${N} DD=/mnt/d/avv/depth_prior/$SC
  local RF=""; [ -n "$UT" ] && RF="--ut_render native"
  conda activate gsplat
  if [ ! -d "$ES" ]; then
    CUDA_VISIBLE_DEVICES=$G python scripts/make_eval_split.py --scene /mnt/d/avv/data/phase1/private_set2/$SC \
      --out $ES --n_eval 40 2>&1 | tail -1 || { echo "[$SC] SPLIT FAIL"; return 1; }
  fi
  if [ ! -f "$DD/$(ls $ES/train_sub/images | head -1 | sed 's/\.[^.]*$//').npy" ]; then
    CUDA_VISIBLE_DEVICES=$G python scripts/precompute_depth.py --images $ES/train_sub/images \
      --out $DD --device cuda:0 2>&1 | tail -1 || { echo "[$SC] DEPTH PRECOMPUTE FAIL"; return 1; }
  fi
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
    --seed 42 $UT --out $M --depth_prior $W --depth_dir $DD $(recipe $SC) 2>&1 | tail -2 \
    || { echo "[$SC $N] TRAIN FAIL (gpu$G)"; return 1; }
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt --csv $ES/eval_poses.csv \
    --out $M/eval_render --png_dir $M/eval_png $RF 2>&1 | tail -1 || { echo "[$SC $N] RENDER FAIL"; return 1; }
  conda activate fastgs2
  CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag "${SC}_${N}"
}

worker() {
  local G=$1 ln
  while true; do
    ln=0; local claimed=0
    while IFS= read -r line || [ -n "$line" ]; do
      ln=$((ln+1)); [ -z "$line" ] && continue
      if mkdir "$CLAIMS/$ln" 2>/dev/null; then
        claimed=1
        IFS='|' read -r SC N W UT <<< "$line"
        echo ">>> gpu$G job#$ln: $SC $N w=$W [$UT] $(date +%H:%M)"
        run_one "$G" "$SC" "$N" "$W" "$UT"
        touch "/mnt/d/avv/depth_pool/done/${ln}_${SC}_${N}"
        break
      fi
    done < "$JOBS"
    if [ "$claimed" = "0" ]; then
      [ -f /mnt/d/avv/depth_pool.SEALED ] && { echo "=== gpu$G sealed+empty, exit ==="; return 0; }
      sleep 30
    fi
  done
}

while [ ! -f /mnt/d/avv/depth_bonsai.DONE ] || [ ! -f /mnt/d/avv/depth_tower.DONE ]; do sleep 60; done
echo "=== both current depth runs done, POOL starting $(date) ==="
worker 0 &
worker 1 &
wait
echo "=== DEPTH POOL DONE $(date) ==="
touch /mnt/d/avv/depth_pool.DONE
