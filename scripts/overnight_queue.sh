#!/usr/bin/env bash
# Sequential single-GPU overnight queue. EDIT the jobs() list each evening, then:
#   setsid nohup bash scripts/overnight_queue.sh > runs/overnight_$(date +%m%d)/queue.log 2>&1 < /dev/null & disown
#   ps -eo pid,sid,cmd | grep overnight      # VERIFY: SID differs from your shell's
# Rules (paid for in GPU-hours — see docs/runbooks/onsite_playbook.md):
#  - every job in its OWN fresh dir; .DONE markers; continue on failure
#  - always --ckpt_every so a crash leaves a renderable ckpt_latest.pt
#  - sum of expected wall-clocks <= 11h
set -u
cd "$(dirname "$0")/.."
[ -f configs/paths.sh ] && source configs/paths.sh || source configs/paths.example.sh
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate "$CONDA_ENV" || true
NIGHT="$RUNS_ROOT/overnight_$(date +%m%d)"; mkdir -p "$NIGHT"

job () {  # job <scene_dir> <recipe> <name> [extra...]
  local SCENE="$1"; local RECIPE="$2"; local NAME="$3"; shift 3
  local M="$NIGHT/$NAME"
  [ -f "$M/.DONE" ] && { echo "SKIP $NAME (done)"; return 0; }
  mkdir -p "$M"
  echo ">>> $NAME start $(date +%H:%M)"
  python original/train_gsplat.py --source "$SCENE/train" --images images \
    --out "$M" $(cat "configs/recipes/$RECIPE.args") --ckpt_every 2000 "$@" \
    > "$M/train.log" 2>&1 || { echo "!!! $NAME TRAIN FAILED $(date +%H:%M)"; return 1; }
  python original/render_gsplat.py --ckpt "$M/ckpt.pt" \
    --csv "$SCENE/test/test_poses.csv" --out "$M/test_render" --png_dir "$M/test_png" \
    > "$M/render.log" 2>&1 || { echo "!!! $NAME RENDER FAILED"; return 1; }
  touch "$M/.DONE"; echo "<<< $NAME done $(date +%H:%M)"
}

# ---------------- EDIT TONIGHT'S JOBS BELOW ----------------
# job "$DATA_ROOT/sceneA" tower_ut_production sceneA_ut42 --seed 42   || true
# job "$DATA_ROOT/sceneA" tower_ut_ema_member sceneA_ema7 --seed 7    || true
# -----------------------------------------------------------
echo "QUEUE COMPLETE $(date)"
