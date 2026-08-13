#!/bin/bash
# DECISIVE TEST for the restoration head, after adversarial verification found the input was
# UN-FIELDED while every production tower render IS fielded.
#
# The skeptic's own honest correction: the field is +1.15 on PSNR/SSIM but only +0.019 on LPIPS,
# and the restorer's gain is 96% LPIPS -- so the field does not simply re-explain the +0.315.
# What it DOES mean is that the restorer's near-zero SSIM cost (-0.0008) was measured on a
# MISREGISTERED image, where invented high-frequency detail cannot conflict with aligned true
# detail. On a registered render that cost should rise -- and in a metric where LPIPS and SSIM are
# near zero-sum, that is exactly what decides whether the gain is real.
#
# So: field-correct the SAME ensemble, retrain the SAME restorer, same split. If the gain survives
# on a registered render it is worth pursuing; if it collapses, restoration is dead and we stop.
# Also now: seeded RNGs (reproducible) + saved renders + per-image deltas (auditable).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
G=${1:-0}
ES=/mnt/d/avv/evalsplit/HCM0181
SRC=/mnt/d/avv/restore/HCM0181_ens/png
FLD=/mnt/d/avv/restore/HCM0181_ens_fielded

while [ "$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 3000 ]; do sleep 60; done
conda activate fastgs2

if [ ! -d "$FLD" ]; then
  echo "=== applying HCM0181 lens field to the restorer's input (production-matched) ==="
  # no --strict: fields/HCM0181.npy predates the .meta.json provenance convention. This is a local
  # experiment, never a submission, so the provenance gate is not the right guard here.
  python gsplat_track/apply_field.py --in_dir $SRC --field /mnt/d/avv/fields/HCM0181.npy \
    --out_dir $FLD || { echo "!!! FIELD APPLY FAILED"; exit 1; }
fi

echo ""
echo "=== A: baseline UN-fielded (what the original +0.3152 was measured against) ==="
CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $SRC --gt_dir $ES/eval_gt --tag unfielded_all60
echo "=== B: baseline FIELDED (what production actually ships) ==="
CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $FLD --gt_dir $ES/eval_gt --tag fielded_all60

for S in 0 1; do
  echo ""
  echo "########## RESTORATION on FIELDED input, seed $S ##########"
  CUDA_VISIBLE_DEVICES=$G python /home/bkai/.claude/jobs/1c9cf7e9/tmp/restore_proto.py \
    --render_dir $FLD --gt_dir $ES/eval_gt --n_fit 40 --iters 3000 --seed $S \
    --save_dir /mnt/d/avv/restore/fielded_restored_s$S --tag "fielded_s$S" 2>&1 \
    | grep -aE "BASELINE|RESTORED|DELTA|per-image|saved"
done

echo ""
echo "########## control: same seeds on the UN-fielded input (isolates the field's effect) ##########"
CUDA_VISIBLE_DEVICES=$G python /home/bkai/.claude/jobs/1c9cf7e9/tmp/restore_proto.py \
  --render_dir $SRC --gt_dir $ES/eval_gt --n_fit 40 --iters 3000 --seed 0 \
  --save_dir /mnt/d/avv/restore/unfielded_restored_s0 --tag "unfielded_s0" 2>&1 \
  | grep -aE "BASELINE|RESTORED|DELTA|per-image|saved"

echo ""
echo "=== READ: if the FIELDED delta collapses toward 0 (or the SSIM cost blows out), restoration"
echo "=== is an artifact of measuring on a misregistered image and we stop. Seed spread across"
echo "=== s0/s1 also gives the training-seed variance component that was previously unmeasured."
touch /mnt/d/avv/restore_fielded.DONE
