#!/bin/bash
# Train + render test poses (+ eval when GT exists) for a list of scenes.
#
# Usage:
#   bash run_scenes.sh <gpu_id> <config_tag> <scene_dir> [<scene_dir> ...]
#
# Each scene gets: output/<scene_name>_<config_tag>/
#   point_cloud/iteration_30000/point_cloud.ply
#   test_poses_renders/   (named per test_poses.csv)
#   eval.json             (only if <scene_dir>/test/images exists)
#
# Training hyperparameters come from env vars (defaults = outdoor config):
#   MULT (0.7) DENSIFY_INT (100) HIGHFEAT_LR (0.04) GRAD_ABS (0.0004) EXTRA_ARGS ("")
#   IMAGES_DIR (images)  — e.g. images_undist for undistorted training
#   DISTORT (none)       — 'auto' warps rendered test poses back into the
#                          distorted camera geometry (use with images_undist)
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
cd "$(dirname "$0")"

GPU=$1; TAG=$2; shift 2
OUT_ROOT=${OUT_ROOT:-output}   # e.g. /mnt/d/avv/output for D:-backed storage
RENDER_ARGS=${RENDER_ARGS:-}   # extra render_test_poses.py args (e.g. --png_dir ...)
MULT=${MULT:-0.7}
DENSIFY_INT=${DENSIFY_INT:-100}
HIGHFEAT_LR=${HIGHFEAT_LR:-0.04}
GRAD_ABS=${GRAD_ABS:-0.0004}
EXTRA_ARGS=${EXTRA_ARGS:-}
IMAGES_DIR=${IMAGES_DIR:-images}
DISTORT=${DISTORT:-none}

for SCENE_DIR in "$@"; do
    NAME=$(basename "$SCENE_DIR")
    OUT=${OUT_ROOT}/${NAME}_${TAG}
    echo "=== [GPU $GPU] $NAME -> $OUT ==="

    # retry training up to 3 times (rare residual CUDA crashes waste a run,
    # not the queue)
    ok=0
    for attempt in 1 2 3; do
        # --data_device cpu keeps the ~3.7GB of GT images in RAM; the VRAM
        # headroom prevents the near-OOM CUDA crashes at peak gaussian count
        if CUDA_VISIBLE_DEVICES=$GPU python train.py -s "$SCENE_DIR/train" -i "$IMAGES_DIR" -m "$OUT" \
            --densification_interval "$DENSIFY_INT" --optimizer_type default \
            --highfeature_lr "$HIGHFEAT_LR" --grad_abs_thresh "$GRAD_ABS" --mult "$MULT" \
            --data_device cpu \
            --test_iterations 30000 --save_iterations 30000 --quiet $EXTRA_ARGS; then
            ok=1; break
        fi
        echo "!!! [GPU $GPU] $NAME training attempt $attempt failed, retrying"
        rm -rf "$OUT"
    done
    if [ "$ok" != 1 ]; then
        echo "!!! [GPU $GPU] $NAME FAILED after 3 attempts, skipping"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU python render_test_poses.py -m "$OUT" \
        --csv "$SCENE_DIR/test/test_poses.csv" --out "$OUT/test_poses_renders" --mult "$MULT" \
        --distort "$DISTORT" --sparse "$SCENE_DIR/train/sparse/0" \
        --png_dir "$OUT/test_poses_renders_png" $RENDER_ARGS

    if [ -d "$SCENE_DIR/test/images" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python eval_test_renders.py \
            --renders "$OUT/test_poses_renders" --gt "$SCENE_DIR/test/images" \
            --out_json "$OUT/eval.json" | tail -6
    fi
done
echo "=== [GPU $GPU] queue '$TAG' done ==="
