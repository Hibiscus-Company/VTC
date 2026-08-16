#!/usr/bin/env bash
# Fast no-dataset verification that the pipeline is intact on this machine.
# GPU/torch parts degrade gracefully if the env is missing.
set -e
cd "$(dirname "$0")/.."
echo "== 1. compile check =="
python3 -m py_compile original/*.py && echo "   all original/*.py compile"
echo "== 2. CLI surface (needs the gsplat env for the torch-importing tools) =="
if source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate "${CONDA_ENV:-gsplat}" 2>/dev/null; then
  for t in train_gsplat render_gsplat render_train fit_field apply_field fov_mask \
           ensemble_renders energy_restore build_submission_zip verify_zip \
           preflight make_eval_split eval_score score_submission; do
    python "original/$t.py" --help > /dev/null && echo "   $t --help OK"
  done
else
  echo "   (conda env not found — compile check only)"
fi
echo "SMOKE TEST PASSED"
