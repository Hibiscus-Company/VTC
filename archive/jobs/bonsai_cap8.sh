#!/bin/bash
# BONSAI CAPACITY LIFT -- the largest single lever found in the campaign.
#
# WHY: per-scene triage on TRAIN views (train photos = legal GT) shows bonsai is the one broken
# scene: LPIPS 0.2047 vs 0.075-0.091 for the other six, train score 74.70 vs a 7-scene mean of
# 80.36. Radial power spectrum render/GT: mid 0.42, high 0.19 -- the reconstruction never
# resolved the scene. Lifting bonsai to the 7-scene mean is worth ~+0.81 leaderboard points,
# roughly 20x anything else still open on freeze day.
#
# ROOT CAUSE: bonsai alone trains at 30k iters / 5M cap while chair and every tower get
# 60k / 8M. That came from the 17/07 eval sweep whose ladder was STILL CLIMBING at the top
# (0.5M 69.57 -> 1M 70.30 -> 2M 70.66 -> 5M 71.16) with the cap BINDING (the run saturated,
# "Saved 5000000 gaussians"). 8M was never tested, and every arm was fixed at 30k iters.
#
# COLLAPSE SAFETY: the 17/07 autopsy resolved the fog collapse as MCMC CHURN (noise injection
# running to 50k), NOT the cap -- median opacity 0.000, "noise never let it settle". Churn is
# orthogonal to capacity, so we raise ONLY cap_max and iters and leave the collapse-safe
# noise_stop 8000 / refine_stop 15000 exactly as capD set them. Every run is gated on opacity
# and on a render-sharpness check before it is allowed anywhere near a submission.
#
# ARM A (gpu1): full 248-frame production member -- the thing we would actually ship.
# ARM B (gpu0): identical config on the 17/07 eval split -> honest held-out score, so we learn
#               whether to ship ARM A while ARM A is still training.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
PRI=/mnt/d/avv/data/phase1/private_set2
ES=/mnt/d/avv/evalsplit/bonsai
OUT=/mnt/d/avv/r33_bonsai
mkdir -p $OUT

# capD churn (UNCHANGED, collapse-safe) + capacity raised to the tower/chair standard
COMMON="--mip3d 0.2 --mip3d_every 100 --cap_max 8000000 --refine_stop 15000 --noise_stop 8000"

opacity_gate() {   # $1 = ckpt ; fog collapse showed as median opacity 0.000
  conda activate fastgs2
  python - "$1" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu")
sp = c.get("splats", c)
o = sp["opacities"] if "opacities" in sp else sp["opacity"]
o = torch.sigmoid(o.float().flatten()) if o.min() < 0 else o.float().flatten()
med = o.median().item(); dead = (o < 0.005).float().mean().item()
print(f"GATE n={o.numel()} median_opacity={med:.4f} dead_frac={dead:.4f}")
sys.exit(0 if (med > 0.02 and dead < 0.60) else 1)
PY
}

armA() {   # production member, full data
  local M=$OUT/prod_s555
  mkdir -p $M
  conda activate gsplat
  echo ">>> ARM A (gpu1) bonsai prod 60k/8M START $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/train_gsplat.py \
    --source $PRI/bonsai/train --images images --out $M \
    --seed 555 $COMMON --iters 60000 --lpips_from 30000 2>&1 \
    | grep -aE "Baked|Saved|Error|error|iter" | tail -5
  [ -f $M/ckpt.pt ] || { echo "!!! ARM A NO CKPT"; return 1; }
  opacity_gate $M/ckpt.pt || { echo "!!! ARM A FAILED OPACITY GATE -- collapsed, discarding"; return 1; }
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=1 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $PRI/bonsai/test/test_poses.csv --out $M/test_render --png_dir $M/test_png 2>&1 | tail -1
  echo "$COMMON --iters 60000 --lpips_from 30000 --seed 555" > $M/train_args.txt
  echo "<<< ARM A DONE $(date +%H:%M) ($(ls $M/test_png 2>/dev/null | wc -l) pngs)"
  touch $OUT/armA.DONE
}

armB() {   # honest held-out validation on the 17/07 split that selected capD
  local M=$OUT/eval_s42
  mkdir -p $M
  conda activate gsplat
  echo ">>> ARM B (gpu0) bonsai eval 60k/8M START $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py \
    --source $ES/train_sub --images images --out $M \
    --seed 42 $COMMON --iters 60000 --lpips_from 30000 2>&1 \
    | grep -aE "Baked|Saved|Error|error" | tail -5
  [ -f $M/ckpt.pt ] || { echo "!!! ARM B NO CKPT"; return 1; }
  opacity_gate $M/ckpt.pt || { echo "!!! ARM B FAILED OPACITY GATE"; return 1; }
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $ES/eval_poses.csv --out $M/eval_render --png_dir $M/eval_png 2>&1 | tail -1
  conda activate fastgs2
  echo "--- ARM B HELD-OUT SCORE (compare vs capD 5M/30k = 71.156, LPIPS 0.2595) ---"
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_score.py \
    --render_dir $M/eval_png --gt_dir $ES/eval_gt --tag bonsai_cap8M_60k 2>&1 | tail -8
  rm -f $M/ckpt.pt      # 2GB, disk is at 18G
  touch $OUT/armB.DONE
}

armA > $OUT/armA.log 2>&1 &
sleep 45
armB > $OUT/armB.log 2>&1 &
wait
echo "=== BOTH ARMS FINISHED $(date +%H:%M) ==="
touch $OUT/ALL.DONE
