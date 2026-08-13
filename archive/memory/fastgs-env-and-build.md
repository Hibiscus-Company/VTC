---
name: fastgs-env-and-build
description: Which conda env to use for FastGS on the RTX 5070 Ti machine and how the CUDA extensions were built
metadata: 
  node_type: memory
  type: project
  originSessionId: 60d62c42-eff2-4507-b2be-4de7dabfda94
---

Machine: WSL2, 2× RTX 5070 Ti (Blackwell sm_120, 16GB each, ~5GB used by Windows), driver CUDA 12.9, gcc 13.3.

- USE env `fastgs2`: python 3.10, torch 2.7.1+cu128, torchvision 0.22.1, numpy<2, cuda-toolkit 12.8.1 (conda, nvcc in env), libstdcxx-ng>=13.
- The original `fastgs` env (stock environment.yml: py3.7, torch 1.12.1+cu116) is unusable here: nvcc 11.6 can't target sm_120, CUDA submodules never installed, PIL broken. Kept untouched.
- CUDA extensions (diff-gaussian-rasterization_fastgs, simple-knn, fused-ssim) built with:
  `CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="12.0+PTX" pip install --no-build-isolation ./submodules/<name>`
- Required source fix for gcc13: added `#include <cstdint>` to submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/rasterizer_impl.h (uint32_t/std::uintptr_t undefined otherwise).
- Repo git status shows ~every file modified — that's CRLF line-ending churn from the Windows checkout, not real changes.
- FastGS `--mult` (compact-box tile culling, default 0.5, outdoor scenes use 0.7) affects rendering: render/eval MUST pass the same --mult used in training; cfg_args does not carry it to render.py.
[[nvs-competition-setup]]
