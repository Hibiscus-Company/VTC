#!/bin/bash
# H-A test setup: isolated conda env for Difix3D+ (nvidia/difix, single-step diffusion render
# fixer). Isolated so it can't poison fastgs2/gsplat. CPU/network only -- no GPU contention.
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q "^difix "; then
  conda create -y -n difix python=3.10 || exit 1
fi
conda activate difix
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121 || exit 1
pip install --quiet diffusers transformers accelerate safetensors pillow numpy peft || exit 1
# prefetch weights so the GPU run doesn't block on download
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("nvidia/difix")
print("weights at:", p)
PY
echo "=== DIFIX ENV READY $(date) ==="
touch /mnt/d/avv/difix_env.READY
