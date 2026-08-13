#!/bin/bash
# Third attempt: clone with submodules, build from local dir, FULL build log
# (previous tail -20 hid the actual nvcc/gcc error).
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export MAX_JOBS=4
cd /mnt/d/avv
rm -rf gsplat_src
git clone --recursive --depth 1 https://github.com/nerfstudio-project/gsplat.git gsplat_src
cd gsplat_src
pip install --no-build-isolation -v . > /home/bkai/.claude/jobs/1c9cf7e9/tmp/gsplat_build_full.log 2>&1
if python -c "import gsplat; print('gsplat', gsplat.__version__)"; then
  echo "GSPLAT ENV DONE" >> /home/bkai/.claude/jobs/1c9cf7e9/tmp/gsplat_env.log
else
  echo "GSPLAT BUILD FAILED again, see gsplat_build_full.log" >> /home/bkai/.claude/jobs/1c9cf7e9/tmp/gsplat_env.log
fi
