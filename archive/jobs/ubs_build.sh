#!/bin/bash
# Build UBS in an ISOLATED env. Its submodule is also called `gsplat`, so installing into the
# working `gsplat` env would shadow the real one and break the production bonsai members that
# are training right now. /home has 714 GB, so cloning the env is free.
# Keep the toolchain we know works on sm_120: py3.10 + torch 2.7.1+cu128 (their README pins
# py3.8 + cu124, which cannot host sm_120).
set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n ubs --clone gsplat 2>&1 | tail -3
conda activate ubs
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
cd /home/bkai/.claude/jobs/1c9cf7e9/tmp/ubs_repo/submodules
TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=6 pip install . --no-build-isolation --no-deps 2>&1 | tail -25
echo "=== import check (must be the UBS fork, not our gsplat) ==="
cd /home/bkai/.claude/jobs/1c9cf7e9/tmp/ubs_repo
python - <<'PY'
import gsplat, inspect, os
print("gsplat at:", os.path.dirname(gsplat.__file__))
from gsplat.rendering import rasterization
sig = inspect.signature(rasterization)
ks = list(sig.parameters)
print("has betas:", "betas" in ks, "| has sb_params:", "sb_params" in ks, "| has camera_model:", "camera_model" in ks)
print("n params:", len(ks))
PY
echo "=== UBS BUILD DONE ==="
touch /mnt/d/avv/ubs_build.DONE
