#!/bin/bash
# Attempt 4: identical recipe to fix2 but building on ext4 (~) instead of
# /mnt/d DrvFs, where nvcc I/O ran at ~11 min/object (6/48 in 80 min).
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export MAX_JOBS=8
rm -rf ~/gsplat_src
cp -r /mnt/d/avv/gsplat_src ~/gsplat_src
cd ~/gsplat_src
rm -rf build *.egg-info
pip install --no-build-isolation -v . > /home/bkai/.claude/jobs/1c9cf7e9/tmp/gsplat_build_full2.log 2>&1
if python -c "import gsplat; print('gsplat', gsplat.__version__)"; then
  echo "GSPLAT ENV DONE" >> /home/bkai/.claude/jobs/1c9cf7e9/tmp/gsplat_env.log
else
  echo "GSPLAT BUILD FAILED again, see gsplat_build_full2.log" >> /home/bkai/.claude/jobs/1c9cf7e9/tmp/gsplat_env.log
fi
