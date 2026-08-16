#!/usr/bin/env bash
# Build the two CUDA extensions from vendored source for THIS machine's GPU.
#   bash env/build_extensions.sh 9.0     # H200 (sm_90)
#   bash env/build_extensions.sh 12.0    # RTX 5070 Ti (sm_120)
# Run inside the target conda/python env. Requires nvcc (CUDA >= 12.x).
set -e
ARCH=${1:?usage: build_extensions.sh <cuda-arch e.g. 9.0 or 12.0>}
cd "$(dirname "$0")"
export TORCH_CUDA_ARCH_LIST="${ARCH}+PTX"
[ -n "$CONDA_PREFIX" ] && export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
echo "arch=$TORCH_CUDA_ARCH_LIST CUDA_HOME=${CUDA_HOME:-system}"
# fused-ssim: OUR fork (shared-memory overflow fix, CCX=BX+10). Never replace with upstream.
pip install --no-build-isolation ./vendor/fused-ssim
# gsplat: from the vendored sdist (offline-safe)
SDIST=$(ls vendor/gsplat-*.tar.gz 2>/dev/null | head -1)
if [ -n "$SDIST" ]; then pip install --no-build-isolation "$SDIST";
else echo "no vendored gsplat sdist found — trying PyPI"; pip install --no-build-isolation gsplat==1.5.3; fi
python -c "import gsplat, fused_ssim; print('OK gsplat', gsplat.__version__)"
