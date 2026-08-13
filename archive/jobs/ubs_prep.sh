#!/bin/bash
set -x
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ubs
pip install -q scikit-learn plyfile tqdm opencv-python tensorly tabulate 2>&1 | tail -3
python -c "import fused_ssim" 2>&1 | tail -1
cd /home/bkai/.claude/jobs/1c9cf7e9/tmp/ubs_repo
# --- patch: add VGG-LPIPS to the loss, matching OUR recipe (lambda 0.1 from iter 12000) ---
python - <<'PY'
p="train.py"; s=open(p).read()
if "UBS_LPIPS_PATCH" not in s:
    s=s.replace("""            if args.densify_from_iter < iteration < args.densify_until_iter:""",
"""            # UBS_LPIPS_PATCH: their stock recipe is L1+DSSIM only, ours trains with
            # VGG-LPIPS lambda=0.1 from iter 12000. Comparing stock-UBS to our LPIPS-trained
            # incumbent would be biased against UBS, so match the recipe before scoring.
            if iteration >= 12000:
                loss = loss + 0.1 * _ubs_lpips(image.unsqueeze(0)*2-1, gt_image.unsqueeze(0)*2-1).mean()
            if args.densify_from_iter < iteration < args.densify_until_iter:""")
    s=s.replace("from fused_ssim import fused_ssim","""from fused_ssim import fused_ssim
import lpips as _lp
_ubs_lpips = _lp.LPIPS(net='vgg').cuda().eval()
for _p in _ubs_lpips.parameters(): _p.requires_grad_(False)""",1)
    open(p,"w").write(s); print("PATCHED train.py with VGG-LPIPS")
else: print("already patched")
PY
grep -n "UBS_LPIPS_PATCH\|_ubs_lpips" train.py | head -4
python -c "import lpips; print('lpips ok')" 2>&1|tail -1 || pip install -q lpips
echo "=== loader test ==="
python - <<'PY'
import sys; sys.path.insert(0,"/home/bkai/.claude/jobs/1c9cf7e9/tmp/ubs_repo")
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
p="/mnt/d/avv/evalsplit/bonsai/train_sub/sparse/0"
try:
    ex=read_extrinsics_binary(f"{p}/images.bin"); ic=read_intrinsics_binary(f"{p}/cameras.bin")
    print("PARSED OK:",len(ex),"images,",len(ic),"cameras")
    c=list(ic.values())[0]; print("  cam:",c.model,c.width,c.height,c.params)
    n=[ex[k].name for k in list(ex)[:3]]; print("  names:",n)
except Exception as e: print("LOADER FAILED:",type(e).__name__,e)
PY
echo "=== PREP DONE ==="; touch /mnt/d/avv/ubs_prep.DONE
