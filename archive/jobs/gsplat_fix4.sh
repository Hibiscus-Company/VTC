#!/bin/bash
# Attempt 5: Utils.cpp patched (cudaEventCreateWithFlags for CUDA 12.8).
# Incremental: reuse ~/gsplat_src/build (47/48 objects done) — only Utils.o
# rebuilds + link. Import test MUST run outside the source dir (cwd shadows
# the installed wheel and silently triggers a JIT build — that was fix3's
# mystery ninja).
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsplat
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export MAX_JOBS=8
cd ~/gsplat_src
pip install --no-build-isolation -v . > /home/bkai/.claude/jobs/1c9cf7e9/tmp/gsplat_build_full3.log 2>&1
cd ~
if python -c "import gsplat; print('gsplat', gsplat.__version__)"; then
  echo "GSPLAT ENV DONE" >> /home/bkai/.claude/jobs/1c9cf7e9/tmp/gsplat_env.log
else
  echo "GSPLAT BUILD FAILED again, see gsplat_build_full3.log" >> /home/bkai/.claude/jobs/1c9cf7e9/tmp/gsplat_env.log
fi
