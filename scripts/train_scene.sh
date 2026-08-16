#!/usr/bin/env bash
# Train one scene with a recipe file. Usage:
#   bash scripts/train_scene.sh <scene_dir> <recipe_name> <run_name> [extra args...]
# Example:
#   bash scripts/train_scene.sh $DATA_ROOT/sceneA tower_ut_production sceneA_ut42 --seed 42
set -e
cd "$(dirname "$0")/.."
[ -f configs/paths.sh ] && source configs/paths.sh || source configs/paths.example.sh
SCENE=$1; RECIPE=$2; NAME=$3; shift 3
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate "$CONDA_ENV" || true
mkdir -p "$RUNS_ROOT/$NAME"
python original/train_gsplat.py --source "$SCENE/train" --images images \
  --out "$RUNS_ROOT/$NAME" $(cat "configs/recipes/$RECIPE.args") --ckpt_every 2000 "$@" \
  2>&1 | tee "$RUNS_ROOT/$NAME/train.log"
