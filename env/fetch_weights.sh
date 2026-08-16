#!/usr/bin/env bash
# Stage the pretrained weights the pipeline needs OFFLINE into env/weights/
# (gitignored — goes into the upload bundle, not into git).
# LPIPS-vgg scoring needs: torchvision VGG16 backbone (528 MB) + lpips linear heads.
set -e
cd "$(dirname "$0")"; mkdir -p weights/torchhub weights/lpips
CACHE=~/.cache/torch/hub/checkpoints
if [ -f "$CACHE/vgg16-397923af.pth" ]; then cp -n "$CACHE/vgg16-397923af.pth" weights/torchhub/
else python -c "import torchvision; torchvision.models.vgg16(weights='IMAGENET1K_V1')" && cp "$CACHE/vgg16-397923af.pth" weights/torchhub/; fi
LP=$(python -c "import lpips, os; print(os.path.join(os.path.dirname(lpips.__file__),'weights/v0.1'))")
cp -n "$LP"/*.pth weights/lpips/
du -sh weights/*; echo "STAGED. On the offline machine: mkdir -p ~/.cache/torch/hub/checkpoints && cp weights/torchhub/* there; lpips heads ship inside the pip package (weights/lpips/ is the backup copy)."
