#!/bin/bash
# Retry gsplat install with the flags that built the FastGS CUDA submodules
# on this machine (see memory: CUDA_HOME=$CONDA_PREFIX + --no-build-isolation;
# default pip build isolation hides torch from setup.py -> wheel build fails).
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export MAX_JOBS=8
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git 2>&1 | tail -20
python -c "import gsplat; print('gsplat', gsplat.__version__)" || { echo "GSPLAT IMPORT FAILED"; exit 1; }
cd /mnt/d/avv && rm -rf gsplat_examples && git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git gsplat_examples 2>&1 | tail -1
echo "GSPLAT ENV DONE"
