#!/bin/bash
# ONE-COMMAND PIPELINE: scene folders -> verified submission zip.
#
#   scripts/run_dataset.sh --data_root ~/data/phaseN/private --out /mnt/d/avv/pN \
#       --zip /mnt/d/avv/submissions/pN_r1.zip [--tier 2] [--gpus 0,1] [--scenes A B C]
#
# Written for the 16/07 dataset change: every trained model, fitted field and mask is
# scene-specific and dies with the data -- only this script, RUNBOOK.md and the audit
# guards survive.
#
# TIERS:  1 = 1 UT model (~3h/scene) | 2 = 2 UT seeds (~6h, DEFAULT) | 3 = + FastGS gates (~9h)
#
# HARDENED per audit round 10:
#   - PREFLIGHT before any GPU time (a 30s check vs 18 wasted GPU-hours)
#   - atomic-claim workers, NOT a barrier: unequal scenes must not idle a GPU
#   - a failed scene OR SEED is FATAL, never silently shipped as a 1-member "ensemble"
#     (bare `wait` returns 0, so `set -e` cannot see a backgrounded failure)
#   - tier-aware weights: w_UT is 0.5 each at tier<=2, 0.3 each at tier 3 (with gates
#     at 0.1333 x3 -> the tuned w_UT=0.6 family split)
set -euo pipefail

DATA_ROOT=""; OUT_ROOT=""; ZIP_OUT=""; TIER=2; GPUS="0,1"; SCENES=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_root) DATA_ROOT=$2; shift 2;;
    --out)       OUT_ROOT=$2; shift 2;;
    --zip)       ZIP_OUT=$2; shift 2;;
    --tier)      TIER=$2; shift 2;;
    --gpus)      GPUS=$2; shift 2;;
    --scenes)    shift; while [[ $# -gt 0 && $1 != --* ]]; do SCENES="$SCENES $1"; shift; done;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done
[[ -n $DATA_ROOT && -n $OUT_ROOT && -n $ZIP_OUT ]] || {
  echo "usage: $0 --data_root DIR --out DIR --zip FILE [--tier 1|2|3] [--gpus 0,1] [--scenes ...]"; exit 1; }

cd "$(dirname "$0")/.."
DATA_ROOT=$(eval echo "$DATA_ROOT")
[[ -n $SCENES ]] || SCENES=$(ls "$DATA_ROOT")
IFS=',' read -ra GPULIST <<< "$GPUS"
mkdir -p "$OUT_ROOT"/{models,fields,masks,ens,logs,claims,done}
rm -f "$OUT_ROOT"/claims/* "$OUT_ROOT"/done/* 2>/dev/null || true
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1

SEEDS="42"; [[ $TIER -ge 2 ]] && SEEDS="42 7"
NSEED=$(echo "$SEEDS" | wc -w)

# ------------------------------------------------------------------- PREFLIGHT
conda activate fastgs2
python scripts/preflight.py --data_root "$DATA_ROOT" --scenes $SCENES || {
  echo "!!! PREFLIGHT FAILED -- refusing to spend GPU time"; exit 1; }

# ------------------------------------------------------------- per-scene worker
do_scene() {                                   # $1 scene  $2 gpu
  local s=$1 g=$2
  local S="$DATA_ROOT/$s/train" SP="$DATA_ROOT/$s/train/sparse/0"

  # k1 sign decides the render path. Negative k1 is a DEGENERATE COLMAP fit (the real
  # lens is ~+0.009): UT-native forward distortion FOLDS and sprays phantom corner
  # gaussians. The field must be FIT on the SAME path it is APPLIED to -- native vs warp
  # differ by mean 2.4/255 with 27% of px off by >2, far above the 0.2-2px field scale.
  local K1 UTR
  K1=$(conda run -n fastgs2 python -c "
import sys; sys.path.insert(0,'.')
from scene.colmap_loader import read_intrinsics_binary
c=list(read_intrinsics_binary('$SP/cameras.bin').values())[0]
p=list(c.params); print(p[3] if c.model in ('SIMPLE_RADIAL','RADIAL') else 0.0)")
  UTR=native
  if awk -v k="$K1" 'BEGIN{exit !(k<0)}'; then UTR=warp; fi
  echo "[$s] k1=$K1 -> $UTR"

  conda activate gsplat
  for seed in $SEEDS; do
    local M="$OUT_ROOT/models/${s}_ut$seed"
    if [[ ! -f $M/ckpt.pt ]]; then
      CUDA_VISIBLE_DEVICES=$g python gsplat_track/train_gsplat.py \
        --source "$S" --images images --ut --seed "$seed" \
        --out "$M" --iters 60000 --cap_max 8000000 \
        --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 || return 1
    fi
    local RF="--ut_render $UTR"
    [[ $UTR == warp ]] && RF="$RF --distort auto --sparse $SP"
    CUDA_VISIBLE_DEVICES=$g python gsplat_track/render_gsplat.py \
      --ckpt "$M/ckpt.pt" --csv "$DATA_ROOT/$s/test/test_poses.csv" \
      --out "$M/test_renders" --png_dir "$M/test_png" $RF >/dev/null || return 1
  done

  # lens field: fit on TRAIN renders vs TRAIN photos. NEVER test GT (fit_field asserts).
  local M0="$OUT_ROOT/models/${s}_ut42" NTR ST
  NTR=$(ls "$S/images" | wc -l)
  ST=2; [[ $NTR -lt 150 ]] && ST=1     # a thin fit warps by the WRONG amount
  CUDA_VISIBLE_DEVICES=$g python gsplat_track/render_train.py \
    --ckpt "$M0/ckpt.pt" --source "$S" --images images \
    --out "$M0/train_png" --stride $ST --ut_render "$UTR" >/dev/null || return 1

  conda activate fastgs2
  python gsplat_track/fit_field.py --render_dir "$M0/train_png" \
    --gt_dir "$S/images" --out "$OUT_ROOT/fields/$s.npy" || return 1
  python gsplat_track/fov_mask.py --sparse "$SP" --out "$OUT_ROOT/masks/$s.npy" || return 1

  touch "$OUT_ROOT/done/$s"           # the ONLY proof this scene finished
  echo "=== [$s] DONE (gpu$g) ==="
}

# atomic-claim worker: whichever GPU frees first takes the next scene. A barrier-batched
# loop would idle one GPU for hours on unequal scenes (103 vs 240 images).
worker() {
  local g=$1
  for s in $SCENES; do
    mkdir "$OUT_ROOT/claims/$s" 2>/dev/null || continue      # someone else has it
    if ! do_scene "$s" "$g" > "$OUT_ROOT/logs/$s.log" 2>&1; then
      # release the claim so the OTHER gpu can retry (round-9 utq.sh lesson: holding a
      # claim after a failure means nobody retries and the scene vanishes). The hard gate
      # below still catches it if both GPUs fail on it.
      rmdir "$OUT_ROOT/claims/$s" 2>/dev/null || true
      echo "!!! [$s] FAILED on gpu$g, claim released (see $OUT_ROOT/logs/$s.log)"
    fi
  done
}

for g in "${GPULIST[@]}"; do worker "$g" & done
wait                                   # bare `wait` returns 0 -- do NOT trust it

# --------------------------------------------------- HARD GATE: nothing may be missing
conda activate fastgs2
MISSING=""
for s in $SCENES; do
  [[ -f $OUT_ROOT/done/$s ]] || { MISSING="$MISSING $s(no-done)"; continue; }
  for seed in $SEEDS; do
    [[ -d $OUT_ROOT/models/${s}_ut$seed/test_png ]] || MISSING="$MISSING $s(seed$seed)"
  done
  [[ -f $OUT_ROOT/fields/$s.npy ]] || MISSING="$MISSING $s(field)"
  [[ -f $OUT_ROOT/masks/$s.npy ]]  || MISSING="$MISSING $s(mask)"
done
[[ -z $MISSING ]] || {
  echo "!!! ABORT -- incomplete:$MISSING"
  echo "!!! shipping now would silently degrade those scenes (e.g. a 1-member ensemble)"
  exit 1; }

# ------------------------------------------------------------------ compose + zip
if [[ $TIER -ge 3 ]]; then W_UT=0.3; else W_UT=$(awk -v n="$NSEED" 'BEGIN{print 1.0/n}'); fi
PAIRS=""
for s in $SCENES; do
  E="$OUT_ROOT/ens/$s"; DIRS=""; W=""; MK=""
  for seed in $SEEDS; do
    DIRS="$DIRS $OUT_ROOT/models/${s}_ut$seed/test_png"; W="$W $W_UT"; MK="$MK none"
  done
  if [[ $TIER -ge 3 ]]; then
    # gates are UNSUPERVISED in the outer ring on negative-k1 scenes (17.8dB vs the UT
    # member's 21.1dB there) -> mask them. w_UT 0.3x2 + gates 0.1333x3 = the tuned 0.6/0.4.
    for gate in champA memB memC; do
      DIRS="$DIRS $OUT_ROOT/models/${s}_$gate/test_png"; W="$W 0.1333"; MK="$MK $OUT_ROOT/masks/$s.npy"
    done
  fi
  python ensemble_renders.py --dirs $DIRS --weights $W --masks $MK \
    --out "$E/jpg" --png_dir "$E/png_ens" \
    --names_from "$DATA_ROOT/$s/test/test_poses.csv"
  python gsplat_track/apply_field.py --strict \
    --in_dir "$E/png_ens" --field "$OUT_ROOT/fields/$s.npy" --out_dir "$E/png"
  PAIRS="$PAIRS $s=$E/png"
done

python build_submission_zip.py --scene_dirs $PAIRS --data_root "$DATA_ROOT" --out "$ZIP_OUT"
python scripts/verify_zip.py --zip "$ZIP_OUT" --data_root "$DATA_ROOT"
echo "=== PIPELINE DONE -> $ZIP_OUT ==="
