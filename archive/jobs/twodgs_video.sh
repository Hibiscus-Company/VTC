#!/bin/bash
# 2DGS eval-split on the two pinhole video scenes (towers need UT which 2dgs lacks).
# $1 = scene, $2 = gpu. Scores solo + ensemble-add. Recipe per-scene.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONUNBUFFERED=1
SC=$1 G=$2; ES=/mnt/d/avv/evalsplit/$SC; M=/mnt/d/avv/twodgs/$SC
ITERS=30000; RSTOP=15000; LP=15000
[ "$SC" = "bonsai" ] && { ITERS=30000; RSTOP=15000; LP=12000; }
conda activate gsplat
echo "=== 2DGS $SC eval-split (gpu$G) $(date) ==="
CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_2dgs.py --source $ES/train_sub --images images \
  --out $M --iters $ITERS --refine_stop $RSTOP --lpips_from $LP \
  --csv $ES/eval_poses.csv --png_dir $M/eval_png 2>&1 | tail -4 || { echo "!!! 2DGS $SC FAIL"; exit 1; }
conda activate fastgs2
echo "=== $SC 2DGS scores ==="
CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag ${SC}_2dgs_solo
# ensemble-add vs existing eval members
if [ "$SC" = "chair" ]; then
  MEMBERS="/mnt/d/avv/tw_test/chair_ema099/eval_png /mnt/d/avv/tw_test/chair_ema999/eval_png /mnt/d/avv/tw_test/chair_capmax2M/eval_png /mnt/d/avv/tw_test/chair_aniso01/eval_png"
  REF="ref mean 71.0682"
else
  MEMBERS="/mnt/d/avv/tw_test/bonsai_ema099/eval_png"
  REF="bonsai baseline 71.8829"
fi
CUDA_VISIBLE_DEVICES=$G python scripts/combiner_sweep.py --tag ${SC}_2dgsadd --out_root /mnt/d/avv/twodgs \
  --dirs $MEMBERS $M/eval_png --gt_dir $ES/eval_gt 2>/dev/null | grep " mean "
echo "($REF)"
echo "=== 2DGS $SC DONE $(date) ==="
touch /mnt/d/avv/twodgs_${SC}.DONE
