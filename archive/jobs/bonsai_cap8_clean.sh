#!/bin/bash
# BONSAI CAP 5M -> 8M, SINGLE VARIABLE. Replaces the confounded first attempt, which an
# adversarial verifier correctly killed. Recording why, because the correction is the point:
#
#   The first run changed FOUR things vs the shipped recipe (iters 30k->60k, lpips_from
#   12k->30k, cap 5M->8M, and added --mip3d). Three of those already have evidence:
#     * 60k iters was A/B-tested on these exact 28 eval holes on 17/07 and LOST twice --
#       pD (60k, lpips@30k) 71.0776 and pA (60k, lpips@40k) 70.7211 against the shipped
#       pC (30k, lpips@12k) 71.3605. My in-flight arm was the near-exact analog of pD.
#     * lpips_from 12k->30k reverts the +0.20 that pC won.
#     * the shipped 6 bonsai members carry NO mip3d, and bonsai's mip3d member landed SOFTER
#       than every peer (detail 0.00058 vs peer 0.00061-0.00078) -- the wrong direction for
#       the one scene whose entire deficit is high frequency.
#   Only the cap is untested, and it is the one the evidence points at: the 17/07 ladder rose
#   monotonically into it (0.5M 69.571 -> 1M 70.298 -> 2M 70.660 -> 5M 71.156), 5M->2.5M cost
#   -0.47, and both bonsai ckpts are byte-identical in size at 30k and 60k = the cap is BINDING.
#
# So: shipped recipe VERBATIM, cap_max 5000000 -> 8000000, nothing else touched.
#   shipped: --iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000
#   here   : --iters 30000 --cap_max 8000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000
# No --ut (bonsai trains antialiased), no --mip3d, capD churn untouched so the fog collapse
# stays fixed.
#
# GATE FIX: the previous opacity_gate called torch.load() without weights_only=False. On torch
# 2.7 that defaults to weights_only=True and these ckpts carry numpy objects, so it RAISED and
# exited 1 -- which the script read as a fog collapse. Both arms would have discarded themselves
# after ~4 GPU-hours and logged "collapsed, discarding", a false negative wearing the exact
# costume of the real failure. Fixed below, and the ckpt is now kept until AFTER the render.
#
# SHIP BAR (also corrected -- the old one was 0.857 scene-pts too lenient): the zip currently
# ships a 6-MEMBER bonsai ensemble scoring 72.013 raw on these 28 holes, NOT the 71.156 single
# model the old script compared against. A single model must beat 72.013 plus the documented
# single-seed noise floor of 0.30 => >= 72.31 to replace the ensemble outright. Below that it is
# only a candidate ENSEMBLE MEMBER, which bcomp.py evaluates separately.
set -o pipefail
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
PRI=/mnt/d/avv/data/phase1/private_set2
ES=/mnt/d/avv/evalsplit/bonsai
OUT=/mnt/d/avv/r33_bonsai
mkdir -p $OUT
ARGS="--iters 30000 --cap_max 8000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000"

opacity_gate() {
  conda activate fastgs2
  python - "$1" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)   # <-- the fix
sp = c.get("splats", c)
o = sp["opacities"] if "opacities" in sp else sp["opacity"]
o = o.float().flatten()
o = torch.sigmoid(o) if o.min() < 0 else o
med = float(o.median()); dead = float((o < 0.005).float().mean())
print(f"GATE n={o.numel()} median_opacity={med:.4f} dead_frac={dead:.4f}")
sys.exit(0 if (med > 0.02 and dead < 0.60) else 1)
PY
}

arm() {   # $1 gpu  $2 tag  $3 source  $4 seed  $5 csv  $6 pngdir
  local G=$1 TAG=$2 SRC=$3 SEED=$4 CSV=$5 PNG=$6 M=$OUT/$2
  mkdir -p $M
  conda activate gsplat
  echo ">>> $TAG (gpu$G) START $(date +%H:%M)  [$ARGS --seed $SEED]"
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/train_gsplat.py \
    --source $SRC --images images --out $M --seed $SEED $ARGS 2>&1 \
    | grep -aE "Baked|Saved|Error|error" | tail -4
  [ -f $M/ckpt.pt ] || { echo "!!! $TAG NO CKPT"; return 1; }
  if opacity_gate $M/ckpt.pt; then echo "    $TAG gate PASSED"
  else echo "!!! $TAG FAILED OPACITY GATE -- rendering anyway so the number is inspectable"; fi
  conda activate gsplat
  CUDA_VISIBLE_DEVICES=$G python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $CSV --out $M/render --png_dir $M/$PNG 2>&1 | tail -1
  echo "$ARGS --seed $SEED" > $M/train_args.txt
  echo "<<< $TAG DONE $(date +%H:%M)  ($(ls $M/$PNG 2>/dev/null | wc -l) pngs)"
  touch $OUT/${TAG}.DONE
}

# gpu1: production member on all 248 frames (the thing we would ship)
arm 1 prod_s555 $PRI/bonsai/train 555 $PRI/bonsai/test/test_poses.csv test_png > $OUT/prod.log 2>&1 &
sleep 40
# gpu0: honest held-out number on the SAME 28 holes the 72.013 baseline was measured on
arm 0 eval_s42 $ES/train_sub 42 $ES/eval_poses.csv eval_png > $OUT/eval.log 2>&1 &
wait
echo "=== BOTH ARMS FINISHED $(date +%H:%M) ==="
touch $OUT/ALL.DONE
