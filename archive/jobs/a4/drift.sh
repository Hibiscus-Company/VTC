#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/bkai/.claude/jobs/1c9cf7e9/tmp/a4
python cpu_score.py --render_dir /mnt/d/avv/bonsai_sel/capD_5Mearly/eval_render --gt_dir /mnt/d/avv/evalsplit/bonsai/eval_gt --tag capD_5M_RESCORE 2>&1 | grep -E "^EVAL"
python cpu_score.py --render_dir /mnt/d/avv/bonsai_perc/pC_lpearly/eval_render --gt_dir /mnt/d/avv/evalsplit/bonsai/eval_gt --tag pC_lpearly_RESCORE 2>&1 | grep -E "^EVAL"
echo ALLDONE
