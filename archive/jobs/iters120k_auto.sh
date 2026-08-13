#!/bin/bash
# 120k fidelity arm -- takes whichever GPU frees, without racing the mip3d pool.
# The pool claims a tower BEFORE waiting for its GPU, and all 5 towers are claimed once HCM0674 is
# taken, so the second worker to finish exits and its GPU is free for good. We wait for a GPU to be
# free for two consecutive checks 90s apart, which a pool worker about to grab it would fail.
#
# WHY RUN IT AT ALL, given the advisor's evidence that the tail is converged (exp26/exp31b: warm-start
# at ~30x terminal means-lr for 8-10k steps gave train +0.52dB but test +0.00):
# the ONE live component is the EXPLORATION phase (densify/relocate under noise), which is what the
# 30k->60k +0.32 actually bought -- the tail and noise-schedule sub-hypotheses both tested dead, and
# that rung of the ladder was never run. Advisor's prediction: proxy median +0.10..+0.15, 80% band
# [-0.10,+0.30], P(negative) 20-25%.
# THE RUN IS ALSO ITS OWN OVERFIT DETECTOR at zero extra cost: it is scored on HELD-OUT eval holes,
# so it measures WITHIN-scene transfer directly. Read: eval +>=0.1 => productionise; train up >=0.4dB
# with eval flat/down => overfit onset, kill the lever without spending a production GPU-minute.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
ES=/mnt/d/avv/evalsplit/HCM0181
M=/mnt/d/avv/iters/HCM0181_mip120k
mkdir -p /mnt/d/avv/iters

free_gpu() {
  for g in 0 1; do
    m=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "${m:-99999}" -lt 2000 ] && { echo $g; return 0; }
  done
  return 1
}
G=""
while [ -z "$G" ]; do
  g1=$(free_gpu) || { sleep 60; continue; }
  sleep 90
  g2=$(free_gpu) || { sleep 60; continue; }
  [ "$g1" = "$g2" ] && G=$g1 || sleep 60
done
echo "=== claimed gpu$G (free on two checks 90s apart) $(date) ==="
conda activate gsplat
echo "=== paired baselines, same seed 101 same scene: plain60k 74.9974 | mip3d60k 75.6570 ==="
CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
  --seed 101 --ut --out $M --mip3d 0.2 --mip3d_every 100 \
  --iters 120000 --cap_max 8000000 --refine_stop 100000 --noise_stop 100000 --lpips_from 100000 2>&1 \
  | grep -aE "^\[[0-9]+\] loss|Baked|Saved" > /mnt/d/avv/iters/curve120k.txt \
  || { echo "!!! 120k TRAIN FAIL"; exit 1; }
tail -3 /mnt/d/avv/iters/curve120k.txt
CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png --ut_render native 2>&1 | tail -1
conda activate fastgs2
CUDA_VISIBLE_DEVICES=$G python scripts/eval_score.py --render_dir $M/eval_png --gt_dir $ES/eval_gt \
  --tag HCM0181_mip120k
echo "=== VERDICT LINE: plain60k 74.9974 | mip3d60k 75.6570 | mip3d120k above ==="
echo "=== full loss curve saved to /mnt/d/avv/iters/curve120k.txt for the tail-slope diagnostic ==="
rm -f $M/ckpt.pt
touch /mnt/d/avv/iters120k.DONE
