#!/bin/bash
# GPU1 work queue. Two jobs, in value order.
#
# JOB 1 -- the last open objection to restoration: DATA STARVATION. The fielded test killed the
# "misregistration artifact" objection (gain went UP to +0.43/+0.39 with 20/20 images improving),
# and the depth test killed the "decays with ensemble depth" objection. What remains is that the
# proxy model saw 180/240 photos while production sees 240. HCM0421's evalgen model saw 200/240
# (83% vs HCM0181's 75%) -- so if the restoration gain is comparable there, starvation is not what
# is driving it. Uses renders already on disk; ~15 min.
#
# JOB 2 -- bonsai seed 303, restarted clean (it was killed on GPU0 where it was contending mip3d).
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1

echo "########## JOB 1: restoration on HCM0421 (200/240 photos vs HCM0181's 180/240) ##########"
conda activate fastgs2
CUDA_VISIBLE_DEVICES=1 python scripts/eval_score.py \
  --render_dir /mnt/d/avv/evalgen/HCM0421/eval_png --gt_dir /mnt/d/avv/evalsplit/HCM0421/eval_gt \
  --tag HCM0421_base_all40
for S in 0 1; do
  CUDA_VISIBLE_DEVICES=1 python /home/bkai/.claude/jobs/1c9cf7e9/tmp/restore_proto.py \
    --render_dir /mnt/d/avv/evalgen/HCM0421/eval_png --gt_dir /mnt/d/avv/evalsplit/HCM0421/eval_gt \
    --n_fit 27 --iters 3000 --seed $S --tag "HCM0421_s$S" 2>&1 \
    | grep -aE "BASELINE|RESTORED|DELTA|per-image"
done
echo "(HCM0181 k=1 comparison point was +0.4072 at 75% data; this is k=1 at 83% data)"
touch /mnt/d/avv/restore_hcm0421.DONE

echo ""
echo "########## JOB 2: bonsai seed 303 (clean restart on a free GPU) ##########"
conda activate gsplat
DATA=/mnt/d/avv/data/phase1/private_set2/bonsai
M=/mnt/d/avv/r24_bonsai/aa303
CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py --source $DATA/train --images images \
  --seed 303 --iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 \
  --lpips_from 12000 --out $M 2>&1 | tail -2 || { echo "!!! bonsai s303 FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $DATA/test/test_poses.csv --out $M/test_render --png_dir $M/test_png 2>&1 | tail -1
rm -f $M/ckpt.pt
touch /mnt/d/avv/r24_bonsai/aa303.DONE
echo "=== GPU1 QUEUE DONE $(date) ==="
touch /mnt/d/avv/gpu1_queue.DONE
