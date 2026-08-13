#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastgs2
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
GT=/mnt/d/avv/evalsplit/bonsai/eval_gt
S=/home/bkai/.claude/jobs/1c9cf7e9/tmp/a4/cpu_score.py
run() { python "$S" --render_dir "$2" --gt_dir "$GT" --tag "$1" 2>&1 | grep -E "^EVAL"; }
run capD_5Mearly  /mnt/d/avv/bonsai_sel/capD_5Mearly/eval_render
run capC_500k     /mnt/d/avv/bonsai_sel/capC_500k/eval_render
run capA_1M       /mnt/d/avv/bonsai_sel/capA_1M/eval_render
run capB_2M       /mnt/d/avv/bonsai_sel/capB_2M/eval_render
run pC_lpearly    /mnt/d/avv/bonsai_perc/pC_lpearly/eval_render
run pA_60k        /mnt/d/avv/bonsai_perc/pA_60k/eval_render
run pB_lpw2       /mnt/d/avv/bonsai_perc/pB_lpw2/eval_render
run pD_60k_lpw2   /mnt/d/avv/bonsai_perc/pD_60k_lpw2/eval_render
run K1_noUT_aa    /mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png
echo "=== RESCORE DONE ==="
