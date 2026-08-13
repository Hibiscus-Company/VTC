# Hibiscus Company — VAR 2026, Track 1 — environment specification

One environment runs the entire pipeline (train, render, ensemble, post-process, score).

## Hardware
- **GPU with >= 16 GB VRAM.** Peak usage is ~13.5 GB (cap_max 8,000,000 gaussians at 60k iters).
  Below 16 GB the training arms OOM.
- CUDA driver supporting **CUDA 12.8**. Developed on RTX 5070 Ti (sm_120, driver 576.88).

## Three things that will silently break the build if missed

### 1. The CUDA toolkit must be present AT RUNTIME, not just at install time
`gsplat` ships a **pure-Python wheel** (`gsplat-1.5.3-py3-none-any.whl`, 6.5 MB, no compiled
kernels). It JIT-compiles its CUDA extension **on first use**, into
`site-packages/gsplat/csrc.so` (~257 MB on our machine, built for sm_120 only).

`pip install gsplat` therefore **succeeds on an image with no nvcc**, and the failure only
appears minutes into the first training run. The image needs `nvcc` (CUDA toolkit >= 12.8),
not just the runtime libraries. Budget 10–25 min and several GB of RAM for that first compile.

### 2. gsplat must be built from source, with a patch
We do **not** use the PyPI package. We build from
`https://github.com/nerfstudio-project/gsplat.git` at commit `77ab983ffe43420b2131669cb35776b883ca4c3c`
(version string reports 1.5.3) **plus the one-hunk patch in `gsplat_cuda128.patch`**.

That patch fixes a genuine compile error under CUDA 12.8:
`cudaEventCreate(&event, flags)` is called with two arguments, but the CUDA API's
`cudaEventCreate` takes one — the two-argument form is `cudaEventCreateWithFlags`.
Without the patch the CUDA sources do not compile on 12.8.

```bash
git clone https://github.com/nerfstudio-project/gsplat.git && cd gsplat
git checkout 77ab983ffe43420b2131669cb35776b883ca4c3c
git apply /path/to/gsplat_cuda128.patch
pip install .
```

### 3. fused-ssim is a CUDAExtension built from our repo
`pip install ./submodules/fused-ssim` — needs nvcc and a torch already installed.
Note our scoring deliberately uses the repo's own SSIM (`utils.loss_utils.ssim`, zero-padded
conv), **not** `fused_ssim`; the two differ by up to 0.003 SSIM. Both must be importable.

## Build order (order matters)
```bash
conda create -n var2026 python=3.10.20 -y && conda activate var2026
conda install -c nvidia/label/cuda-12.8.1 cuda-toolkit -y     # FIRST — provides nvcc
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# gsplat from source + patch, as above
pip install ./submodules/fused-ssim
```

## Verifying the environment
Run `python verify_env.py`. It checks CUDA visibility, compute capability, that the gsplat
kernels actually cover the GPU, and that every module the pipeline imports is present.

## Network
First run downloads LPIPS VGG weights (`lpips`) and the torchvision VGG16 backbone.
On an air-gapped machine, pre-seed `~/.cache/torch`.
