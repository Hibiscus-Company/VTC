# Environment Setup

The pipeline needs ONE python env: python 3.10, torch (CUDA build), gsplat 1.5.3,
fused-ssim (our fork), and `requirements.txt`. Two machines matter:

## A. On-site H200 (sm_90) — assume pip may NOT work

1. **Check what exists first:** `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
   H200 server images usually ship torch. Any torch ≥ 2.4 with CUDA 12.x is fine.
2. **Wheels-first:** if `env/wheels/` is present in the bundle (built by
   `scripts/make_bundle.sh --with-wheels`), `pip install --no-index --find-links env/wheels -r env/requirements.txt`.
3. **CUDA extensions** (must be compiled for sm_90 — do this once, ~5 min):
   `bash env/build_extensions.sh 9.0`
   (fused-ssim from `vendor/fused-ssim` — our fork with the shared-memory fix;
   gsplat from `vendor/gsplat-1.5.3.tar.gz`.)
4. **Offline weights** (LPIPS scoring needs the VGG16 backbone):
   `mkdir -p ~/.cache/torch/hub/checkpoints && cp env/weights/torchhub/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/`
5. **Verify:** `bash scripts/smoke_test.sh`
6. If pip is completely dead and no wheels match: numpy/PIL/opencv/scipy are almost
   certainly preinstalled in any Jupyter image; the only hard requirements built from
   source are gsplat + fused-ssim (step 3 needs only nvcc + torch headers, no pip
   packages). lpips can run from a copied site-packages folder if needed
   (`weights/lpips/` has its linear heads).

## B. Local dev machine (RTX 5070 Ti, sm_120, WSL2)

- Existing working env: conda `gsplat` (torch 2.7.1+cu128). Recreate from scratch:
  ```bash
  conda create -n gsplat python=3.10 -y && conda activate gsplat
  pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  pip install -r env/requirements.txt
  bash env/build_extensions.sh 12.0
  ```
- sm_120 needs torch ≥ 2.7/cu128; every third-party 3DGS repo pinning torch ≤ 2.3 will
  not run on these cards without a torch bump.
- WSL note: `/mnt/c` and `/mnt/d` are slow and chronically full — keep datasets and
  runs on the ext4 filesystem.

## Weights staging (run on a connected machine BEFORE going on-site)

`bash env/fetch_weights.sh` → stages VGG16 (528 MB) + lpips heads into `env/weights/`
(gitignored; included by the bundle script).
