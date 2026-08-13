#!/bin/bash
# Track B env: clone fastgs2 (working torch for sm_120 Blackwell) + gsplat from source
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n gsplat --clone fastgs2 -y > /dev/null
conda activate gsplat
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
pip install ninja rich 2>&1 | tail -1
pip install git+https://github.com/nerfstudio-project/gsplat.git 2>&1 | tail -3
python -c "import gsplat; print('gsplat', gsplat.__version__)"
cd /mnt/d/avv && rm -rf gsplat_examples && git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git gsplat_examples 2>&1 | tail -1
echo "GSPLAT ENV DONE"
