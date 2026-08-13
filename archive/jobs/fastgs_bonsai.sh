#!/bin/bash
# FastGS (Track A, the repo's ORIGINAL method) on bonsai -- never run on this scene.
# EXPERIMENTS.md:1530 gated FastGS to "6 tower/chair scenes"; bonsai was excluded.
# MECHANISM (EXPERIMENTS.md:153): FastGS champ median opacity .511, 0% dead splats, thin-structure
# splats 46% opaque; gsplat .097, 21.4% dead, 12% opaque. "FastGS prunes hard and lets survivors
# saturate." Our bonsai has ~70-80% of 5M splats below opacity 0.05 and its content IS thin
# structure. That is the regime where the two densifiers differ most.
# --mult 0.5 (indoor) and render MUST reuse the same value.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate fastgs2
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
export CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="12.0+PTX"
python -c "import diff_gaussian_rasterization_fastgs,simple_knn,fused_ssim;print('EXTENSIONS OK')" \
  || { echo "!!! extensions missing"; exit 1; }
ES=/mnt/d/avv/evalsplit/bonsai; OUT=/mnt/d/avv/r40_fastgs/bonsai; mkdir -p $OUT
while [ "$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 3000 ]; do sleep 45; done
echo ">>> FastGS bonsai START $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=0 python train.py -s /mnt/d/avv/ubs_data -i images -m $OUT \
  --densification_interval 500 --optimizer_type default --mult 0.5 --data_device cpu \
  --test_iterations 30000 --save_iterations 30000 --quiet 2>&1 | tail -15
CUDA_VISIBLE_DEVICES=0 python render_test_poses.py -m $OUT --csv $ES/eval_poses.csv \
  --out $OUT/eval_png --mult 0.5 --force_png 2>&1 | tail -3
python /home/bkai/.claude/jobs/1c9cf7e9/tmp/bar.py $OUT/eval_png
echo "<<< FastGS DONE $(date +%H:%M)  [lam0.01 control 71.799 | best single lam0.1 71.99 | 6-arm mean 72.26]"
touch /mnt/d/avv/r40_fastgs/bonsai.DONE
