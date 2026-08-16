#!/usr/bin/env bash
# Build + verify a submission zip from per-scene PNG dirs.
#   bash scripts/build_zip.sh <out.zip> <data_root> <SCENE=png_dir> [SCENE=png_dir ...]
set -e
cd "$(dirname "$0")/.."
[ -f configs/paths.sh ] && source configs/paths.sh || source configs/paths.example.sh
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate "$CONDA_ENV" || true
OUT=$1; ROOT=$2; shift 2
python original/build_submission_zip.py --out "$OUT" --data_root "$ROOT" --scene_dirs "$@"
python original/verify_zip.py --zip "$OUT" --data_root "$ROOT"
echo "VERIFIED: $OUT — a HUMAN submits this. Never auto-submit."
