#!/bin/bash
# Coarse OFAT hyperparameter sweep on one public scene (has GT), winning
# undist config as baseline. Each config: train -> render(distort auto) -> eval.
# Baseline (output/<scene>_undist) is reused, not re-run.
#
# Usage: bash sweep.sh <gpu_id> <scene_name>
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
cd "$(dirname "$0")"

GPU=$1; SCENE=$2
DATA=~/data/phase1/public_set/$SCENE
SPARSE=$DATA/train/sparse/0
CSV=$DATA/test/test_poses.csv

# config: "tag|mult|extra_train_args"  (mult applied to both train and render)
CONFIGS=(
  "g2|0.7|--grad_abs_thresh 0.0002"
  "g1|0.7|--grad_abs_thresh 0.0001"
  "m10|1.0|"
  "du20|0.7|--densify_until_iter 20000"
  "combo|1.0|--grad_abs_thresh 0.0002 --densify_until_iter 20000"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag mult extra <<< "$cfg"
  OUT=output/${SCENE}_${tag}
  echo "=== [GPU $GPU] $SCENE / $tag (mult=$mult) $extra ==="
  [ -f "$OUT/eval.json" ] && { echo "  skip (done)"; continue; }

  ok=0
  for attempt in 1 2 3; do
    if CUDA_VISIBLE_DEVICES=$GPU python train.py -s "$DATA/train" -i images_undist -m "$OUT" \
        --densification_interval 100 --optimizer_type default --highfeature_lr 0.04 \
        --grad_abs_thresh 0.0004 --mult "$mult" --data_device cpu \
        --test_iterations 30000 --save_iterations 30000 --quiet $extra; then ok=1; break; fi
    echo "  !! attempt $attempt failed, retry"; rm -rf "$OUT"
  done
  [ "$ok" = 1 ] || { echo "  !! $tag FAILED, skip"; continue; }

  CUDA_VISIBLE_DEVICES=$GPU python render_test_poses.py -m "$OUT" --csv "$CSV" \
      --out "$OUT/test_poses_renders" --mult "$mult" --distort auto --sparse "$SPARSE"
  CUDA_VISIBLE_DEVICES=$GPU python eval_test_renders.py --renders "$OUT/test_poses_renders" \
      --gt "$DATA/test/images" --out_json "$OUT/eval.json" | tail -5
done
echo "=== [GPU $GPU] sweep $SCENE done ==="
