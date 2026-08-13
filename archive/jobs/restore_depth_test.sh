#!/bin/bash
# DECISIVE cheap test: does the restoration gain DECAY WITH ENSEMBLE DEPTH the way the encode gain
# did? The encode fix measured +0.259 at k=1, +0.115 at k=4 and went NEGATIVE at production depth.
# Restoration is the same class of intervention (it removes render noise), so if its gain shows the
# same decay, it is a proxy artifact and must NOT be shipped on proxy evidence.
# Costs ~30 min total, uses members already on disk. Waits for a free GPU (no stacking).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
G=${1:-0}
ES=/mnt/d/avv/evalsplit/HCM0181
R=/mnt/d/avv/restore
M1=/mnt/d/avv/tw_test/HCM0181_ema999/eval_png
M2=/mnt/d/avv/seedbank/HCM0181_ut_s101/eval_png
M3=/mnt/d/avv/seedbank/HCM0181_ut_s202/eval_png

while [ "$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 3000 ]; do sleep 60; done
conda activate fastgs2
mkdir -p $R
echo "=== building k=1 / k=2 / k=3 ensembles ==="
rm -rf $R/k1 $R/k2 && mkdir -p $R/k1/png
cp $M1/*.png $R/k1/png/
python ensemble_renders.py --dirs $M1 $M2 --out $R/k2/jpg --png_dir $R/k2/png 2>&1 | tail -1
# k=3 already exists at $R/HCM0181_ens/png

for K in 1 2 3; do
  case $K in
    1) D=$R/k1/png ;;
    2) D=$R/k2/png ;;
    3) D=$R/HCM0181_ens/png ;;
  esac
  echo ""
  echo "########## RESTORATION at ensemble depth k=$K ##########"
  CUDA_VISIBLE_DEVICES=$G python /home/bkai/.claude/jobs/1c9cf7e9/tmp/restore_proto.py \
    --render_dir $D --gt_dir $ES/eval_gt --n_fit 40 --iters 3000 --tag "k${K}" 2>&1 \
    | grep -aE "BASELINE|RESTORED|DELTA"
done
echo ""
echo "=== INTERPRETATION: if DELTA decays steeply with k (like the encode did: +0.259 -> +0.115),"
echo "=== restoration is a proxy artifact and must not ship on proxy evidence. If flat, it is real."
touch /mnt/d/avv/restore_depth.DONE
