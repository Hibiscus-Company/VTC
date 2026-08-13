#!/bin/bash
# THE GATE. One arm: lam=0.1, seed 42, SECOND replicate. Makes mean(sr01,sr01b) an A/A pair
# exactly like ctrl2=mean(K1,c42), so the k=2 delta is member QUALITY with the diversity
# confound removed -- and halves the run-noise on both sides.
# Noise floor for a single 1v1 comparison is 0.407; that is why one more run matters more than
# any new hyperparameter.
# GATE, set before looking: proceed to the 8-GPU-h production retrain ONLY if the
# diversity-matched, ENCODED k=2 delta is >= +0.12 with paired t >= 2.0. Below that the whole
# path is worth < +0.008 LB, inside the band of the last four losses.
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
ES=/mnt/d/avv/evalsplit/bonsai; M=/mnt/d/avv/r36_shape/sr01b; T=/home/bkai/.claude/jobs/1c9cf7e9/tmp
mkdir -p $M
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | head -1)" -gt 3000 ]; do sleep 90; done
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
conda activate gsplat
echo ">>> sr01b (gpu$G) START $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py --source $ES/train_sub --images images \
  --out $M --iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 \
  --lpips_from 12000 --seed 42 --scale_reg 0.1 2>&1 | grep -aE "Saved|Error" | tail -2
CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
  --csv $ES/eval_poses.csv --out $M/render --png_dir $M/eval_png 2>&1 | tail -1
conda activate fastgs2
python $T/census.py $M/ckpt.pt; python $T/bar.py $M/eval_png
rm -f $M/ckpt.pt
echo "<<< sr01b DONE $(date +%H:%M)"; touch /mnt/d/avv/r36_shape/sr01b.DONE
