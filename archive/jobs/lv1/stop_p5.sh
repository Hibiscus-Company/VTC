#!/bin/bash
# Stop the p5 lam sweep once N images are scored (partial json is complete and self-consistent).
# Kills ONLY our own parent + its direct children.
N=${1:-10}
LOG=/home/bkai/.claude/jobs/1c9cf7e9/tmp/lv1/run_lamk8.log
while true; do
  c=$(grep -cE "^  [0-9]+/" "$LOG" 2>/dev/null || echo 0)
  pid=$(pgrep -x python -a | grep "p5_lam.py" | awk '{print $1}' | head -1)
  if [ -z "$pid" ]; then echo "p5 already exited (n=$c)"; exit 0; fi
  if [ "$c" -ge "$N" ]; then
    echo "reached n=$c, stopping pid $pid"
    pkill -P "$pid"; sleep 2; kill "$pid" 2>/dev/null; sleep 3; echo stopped; exit 0
  fi
  sleep 15
done
