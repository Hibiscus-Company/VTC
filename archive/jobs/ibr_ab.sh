#!/bin/bash
# IBR A/B on HCM0181 public GT (task #12, audit round-8 rank-1).
# Fires when GPU0 frees (privUT_gpu0 done ~05:20). Base ckpt = best single
# gsplatB10ut8M (75.5783). Three arms:
#   ibr_base    tau_geo .02  tau_pho .06  (audit-suggested defaults)
#   ibr_loose   tau_geo .04  tau_pho .10  (more photo coverage)
#   ibr_tight   tau_geo .01  tau_pho .04  (conservative; PSNR-protective)
# Each scored full-60 vs GT; compare to 75.5783 splat-only.
set -o pipefail
until grep -q "PRIVUT GPU0 DONE" /home/bkai/.claude/jobs/1c9cf7e9/tmp/privUT_gpu0.log 2>/dev/null; do sleep 120; done
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/public_set/HCM0181
CKPT=/mnt/d/avv/output/HCM0181_gsplatB10ut8M/ckpt.pt
source ~/miniconda3/etc/profile.d/conda.sh

run_ibr() {  # name, extra args...
  local NAME=$1; shift
  local OUT=/mnt/d/avv/ibr/HCM0181_$NAME
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/ibr_render.py \
    --ckpt $CKPT --source $S/train --csv $S/test/test_poses.csv \
    --out $OUT/renders --png_dir $OUT/renders_png "$@" \
    || { echo "=== IBR $NAME FAILED ==="; return 1; }
  conda activate fastgs2
  mkdir -p ~/densq/ibr_$NAME
  ln -sfn $OUT/renders ~/densq/ibr_$NAME/HCM0181
  CUDA_VISIBLE_DEVICES=0 python score_submission.py --sub ~/densq/ibr_$NAME --device cuda:0 2>&1 \
    | grep -E "HCM0181 |Score\(vgg\)" | sed "s/^/IBR $NAME /"
  echo "=== IBR $NAME SCORED ==="
}

run_ibr base
run_ibr loose --tau_geo 0.04 --tau_pho 0.10
run_ibr tight --tau_geo 0.01 --tau_pho 0.04
echo "IBR AB DONE"
