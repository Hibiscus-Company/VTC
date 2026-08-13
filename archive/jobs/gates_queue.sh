#!/bin/bash
# FASTGS GATE FAMILY on set2 (strategy audit lever 5, set1-proven +0.5 family effect):
# 3 gate members per scene (champA=gate1, memB=gate5, memC=gate2; all on the g15+lpips
# champion base) for 5 towers + chair. bonsai EXCLUDED (collapse-prone; keeps 2-seed pC).
# set2 has NO negative-k1 scenes -> no FoV ring, masks all-ones, gates fully supervised.
# ~25-30 min/member (set1) -> 18 runs ≈ 4-5h wall on 2 GPUs.
set -uo pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
D=/mnt/d/avv/data/phase1/private_set2
export OUT_ROOT=/mnt/d/avv/output_s2gates
export GRAD_ABS=0.00015                 # g15 champion base
mkdir -p $OUT_ROOT

run_members() {  # $1 gpu, rest scene dirs
  local G=$1; shift
  for SC in "$@"; do
    for CFG in "champA:1" "memB:5" "memC:2"; do
      TAG=${CFG%%:*}; GATE=${CFG##*:}
      EXTRA_ARGS="--lambda_lpips 0.1 --metric_gate $GATE" \
      RENDER_ARGS="--png_dir $OUT_ROOT/$(basename $SC)_$TAG/test_png" \
        bash run_scenes.sh $G $TAG "$SC" || echo "!!! $(basename $SC) $TAG FAILED (gpu$G)"
    done
    echo "=== GATES [$(basename $SC)] ALL MEMBERS DONE (gpu$G) ==="
  done
}

( run_members 0 $D/HCM0421 $D/HCM0540 $D/HCM0674 ) &
( run_members 1 $D/HCM0539 $D/HCM0644 $D/chair ) &
wait
echo "=== GATES QUEUE DONE ==="
