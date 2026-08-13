#!/bin/bash
# Wait for the last mip3d member, run the weight sweep, build+verify r25, THEN STOP ALL TRAINING.
# Per user: after r25 drops, no GPU training until 11:59 -- the window is for ideation + cheap
# validation only. This script enforces that automatically so nothing keeps burning GPU unattended.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
DATA=/mnt/d/avv/data/phase1/private_set2

echo "=== waiting for all 5 mip3d production members ==="
while [ ! -f /mnt/d/avv/mip3d_prod.DONE ]; do sleep 60; done
echo "=== all members landed $(date) ==="
for T in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  echo -n "  $T: "; ls /mnt/d/avv/r25_mip3d/$T/test_png 2>/dev/null | wc -l
done

conda activate fastgs2
echo ""
echo "=== mip3d ensemble-weight sweep (do this BEFORE building, not after) ==="
CUDA_VISIBLE_DEVICES=0 python /home/bkai/.claude/jobs/1c9cf7e9/tmp/mip_weight_sweep.py 2>&1 \
  | grep -avE "Warning|warn|Loading|Setting" || echo "(sweep failed, will ship uniform)"

echo ""
echo "=== STOPPING ALL TRAINING (user directive: ideation window until 11:59) ==="
pkill -f iters120k_auto.sh 2>/dev/null || true
pkill -f mip3d_prod.sh 2>/dev/null || true
pkill -f "train_gsplat.py" 2>/dev/null || true
sleep 5
echo "remaining trainers: $(ps -eo cmd | grep -c '[t]rain_gsplat.py')"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
touch /mnt/d/avv/TRAINING_HALTED
echo "=== TRAINING HALTED $(date) -- ideation window open ==="
