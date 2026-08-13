export NVCC_PREPEND_FLAGS="" NVCC_APPEND_FLAGS=""
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gsplat
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
CSV=/mnt/d/avv/evalsplit/bonsai/eval_poses.csv
GT=/mnt/d/avv/evalsplit/bonsai/eval_gt
export CUDA_VISIBLE_DEVICES=1
echo ">>> control: score EXISTING capD render (expect 71.1563)"
python scripts/eval_score.py --render_dir /mnt/d/avv/bonsai_sel/capD_5Mearly/eval_render --gt_dir $GT --tag capD_EXISTING
echo ">>> render 8M eval_s42"
python gsplat_track/render_gsplat.py --ckpt /mnt/d/avv/r33_bonsai/eval_s42/ckpt.pt --csv $CSV --out /mnt/d/avv/r33_bonsai/eval_s42/eval_render
python scripts/eval_score.py --render_dir /mnt/d/avv/r33_bonsai/eval_s42/eval_render --gt_dir $GT --tag cap8M_s42
echo ">>> re-render 5M capD with identical flags"
python gsplat_track/render_gsplat.py --ckpt /mnt/d/avv/bonsai_sel/capD_5Mearly/ckpt.pt --csv $CSV --out /mnt/d/avv/bonsai_sel/capD_5Mearly/eval_render_v2
python scripts/eval_score.py --render_dir /mnt/d/avv/bonsai_sel/capD_5Mearly/eval_render_v2 --gt_dir $GT --tag capD_RERENDER
echo ">>> ALLDONE"
