#!/usr/bin/env bash
# Render a trained run at its scene's test poses (PNG archive + preview JPEG).
#   bash scripts/render_pipeline.sh <scene_dir> <run_name> [render extra args...]
set -e
cd "$(dirname "$0")/.."
[ -f configs/paths.sh ] && source configs/paths.sh || source configs/paths.example.sh
SCENE=$1; NAME=$2; shift 2
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate "$CONDA_ENV" || true
CKPT="$RUNS_ROOT/$NAME/ckpt.pt"
[ -f "$CKPT" ] || CKPT="$RUNS_ROOT/$NAME/ckpt_latest.pt"   # crashed-run fallback
python original/render_gsplat.py --ckpt "$CKPT" \
  --csv "$SCENE/test/test_poses.csv" \
  --out "$RUNS_ROOT/$NAME/test_render" --png_dir "$RUNS_ROOT/$NAME/test_png" "$@" \
  2>&1 | tee "$RUNS_ROOT/$NAME/render.log"
