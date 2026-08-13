#!/bin/bash
# P6 VALIDATION: are the FastGS gate members really garbage in the unsupervised ring?
#
# On HNI0131/HNI0265 (k1=-0.115) the same-K undistortion crops the outer FoV, so the
# gates were NEVER supervised on 11.9% of the frame -- yet they carry 0.4 of the
# ensemble weight there. No PUBLIC scene has negative k1 (0% masked), so this cannot
# be validated on public at all. TRAIN photos are the only legal ground truth.
#
# Render the gate (FastGS champ) and the UT member at TRAIN poses, then compare
# MSE INSIDE vs OUTSIDE the supervised mask against the train photos.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
S=~/data/phase1/private_set1/HNI0131
CSV=/mnt/d/avv/csv/HNI0131_train.csv
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1

# 1) FastGS gate at train poses (same warp/distort path the submission uses)
conda activate fastgs2
CUDA_VISIBLE_DEVICES=0 python render_test_poses.py \
  -m output/HNI0131_champ --csv $CSV \
  --out /mnt/d/avv/output/HNI0131_champA/train_renders \
  --force_png --distort auto --sparse $S/train/sparse/0 --mult 0.7 2>&1 | tail -2 \
  || { echo "=== GATE TRAIN RENDER FAILED ==="; exit 1; }
echo "=== GATE TRAIN RENDERED ==="

# 2) UT member at the same train poses (warp path -- the negative-k pair ships warp)
conda activate gsplat
CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_train.py \
  --ckpt /mnt/d/avv/output/HNI0131_gsplatB9ut/ckpt.pt \
  --source $S/train --images images \
  --out /mnt/d/avv/output/HNI0131_gsplatB9ut/train_renders_p6 \
  --stride 6 --ut_render warp 2>&1 | tail -1 \
  || { echo "=== UT TRAIN RENDER FAILED ==="; exit 1; }
echo "=== UT TRAIN RENDERED ==="

# 3) ring-region comparison against TRAIN photos
conda activate fastgs2
python - <<'PY'
import numpy as np, os
from PIL import Image
GT=os.path.expanduser("~/data/phase1/private_set1/HNI0131/train/images")
G="/mnt/d/avv/output/HNI0131_champA/train_renders"
U="/mnt/d/avv/output/HNI0131_gsplatB9ut/train_renders_p6"
m=np.load("/mnt/d/avv/masks/HNI0131.npy")          # 1 = gate supervised
inside, outside = m>0.5, m<0.5
gt={os.path.splitext(f)[0]:f for f in os.listdir(GT)}
def psnr(e): return 10*np.log10(1.0/max(float(e),1e-12))
acc={k:[0.0,0.0] for k in ("gate_in","gate_out","ut_in","ut_out")}
n=0
for f in sorted(os.listdir(U)):
    s=os.path.splitext(f)[0]
    if s not in gt or not os.path.exists(os.path.join(G,s+".png")): continue
    g=np.asarray(Image.open(os.path.join(GT,gt[s])).convert("RGB"),dtype=np.float32)/255
    a=np.asarray(Image.open(os.path.join(G,s+".png")).convert("RGB"),dtype=np.float32)/255
    b=np.asarray(Image.open(os.path.join(U,f)).convert("RGB"),dtype=np.float32)/255
    if a.shape!=g.shape or b.shape!=g.shape: continue
    ea=((a-g)**2).mean(axis=2); eb=((b-g)**2).mean(axis=2)
    acc["gate_in"][0]+=ea[inside].sum();  acc["gate_in"][1]+=inside.sum()
    acc["gate_out"][0]+=ea[outside].sum();acc["gate_out"][1]+=outside.sum()
    acc["ut_in"][0]+=eb[inside].sum();    acc["ut_in"][1]+=inside.sum()
    acc["ut_out"][0]+=eb[outside].sum();  acc["ut_out"][1]+=outside.sum()
    n+=1
print(f"\n===== P6: HNI0131 train views (n={n}), mask suppresses {100*outside.mean():.1f}% =====")
for k in ("gate_in","gate_out","ut_in","ut_out"):
    print(f"  {k:9s} PSNR {psnr(acc[k][0]/max(acc[k][1],1)):7.3f} dB")
gi,go = psnr(acc['gate_in'][0]/acc['gate_in'][1]), psnr(acc['gate_out'][0]/acc['gate_out'][1])
ui,uo = psnr(acc['ut_in'][0]/acc['ut_in'][1]),   psnr(acc['ut_out'][0]/acc['ut_out'][1])
print(f"\n  gate degrades {gi-go:+.2f} dB going into the unsupervised ring")
print(f"  UT   degrades {ui-uo:+.2f} dB over the same region")
print(f"  in the RING: UT beats gate by {uo-go:+.2f} dB")
print("\n  => if UT beats gate in the ring by a wide margin, giving the gates 0.4")
print("     weight there is pure loss and the mask is justified.")
PY
echo "P6 DONE"
