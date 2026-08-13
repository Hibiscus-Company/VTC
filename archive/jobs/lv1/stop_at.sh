#!/bin/bash
# Wait until the p3_par run has scored N images, then stop it cleanly (partial json is complete).
# Kills ONLY our own parent + its direct children -- never touches production training.
N=${1:-15}
LOG=/home/bkai/.claude/jobs/1c9cf7e9/tmp/lv1/run_k8.log
while true; do
  c=$(grep -cE "^  [0-9]+/" "$LOG")
  pid=$(pgrep -x python -a | grep "p3_par.py" | awk '{print $1}' | head -1)
  if [ -z "$pid" ]; then echo "p3_par already exited (n=$c)"; exit 0; fi
  if [ "$c" -ge "$N" ]; then
    echo "reached n=$c, stopping pid $pid and its workers"
    pkill -P "$pid"; sleep 2; kill "$pid" 2>/dev/null
    sleep 3; echo "stopped"; exit 0
  fi
  sleep 15
done
