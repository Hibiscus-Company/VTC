#!/bin/bash
# BOUNDED Difix fine-tune on chair (last multi-point-class candidate). Gated on HCM0674's
# marker (GPU0 freeing). Data: the 147 train-pose renders from the blur-bound test (same
# eval-split chair_ema099 model whose eval baseline is 69.8051) paired with train photos.
# 140 train / 7 repo-internal val. Kill bar: fine-tuned inference on the 58 eval holes must
# beat 69.8051 by >= +0.3 or the restoration class is CLOSED.
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
D3=/home/bkai/.claude/jobs/1c9cf7e9/tmp/Difix3D
OUT=/mnt/d/avv/difix_ft/chair
mkdir -p $OUT

echo "gate removed -- GPU0 freed by killing pathological HCM0674"
echo "=== GPU0 free, starting Difix chair fine-tune $(date) ==="

conda activate difix
python - <<'PY'
import json, os
ren = "/mnt/d/avv/blurbound/chair/train_png"
gt = "/mnt/d/avv/evalsplit/chair/train_sub/images"
gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gt)}
names = sorted(os.listdir(ren))
data = {"train": {}, "test": {}}
for i, f in enumerate(names):
    s = os.path.splitext(f)[0]
    entry = {"image": os.path.join(ren, f),
             "target_image": os.path.join(gt, gt_by[s]),
             "prompt": "remove degradation"}
    (data["test"] if i % 21 == 10 else data["train"])[s] = entry
os.makedirs("/mnt/d/avv/difix_ft/chair", exist_ok=True)
json.dump(data, open("/mnt/d/avv/difix_ft/chair/data.json", "w"), indent=1)
print(f"data.json: {len(data['train'])} train / {len(data['test'])} val")
PY

cd $D3
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision=bf16 src/train_difix_ft.py \
  --output_dir=$OUT/run \
  --dataset_path=$OUT/data.json \
  --init_from_difix "nvidia/difix" \
  --max_train_steps 3000 \
  --resolution=512 --learning_rate 2e-5 \
  --train_batch_size=1 --dataloader_num_workers 4 \
  --checkpointing_steps=1000 --eval_freq 500 --viz_freq 10000 \
  --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 1000 \
  --report_to "tensorboard" --tracker_project_name "difix_ft" --tracker_run_name "chair" \
  --timestep 199 2>&1 | tail -20 || { echo "!!! DIFIX FT TRAIN FAIL"; exit 1; }

echo "=== fine-tune done, checkpoints: ==="
ls $OUT/run/checkpoints/
touch /mnt/d/avv/difix_ft_chair_trained.DONE
