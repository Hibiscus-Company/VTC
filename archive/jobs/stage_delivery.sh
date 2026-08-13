#!/bin/bash
# Stage the post-verification delivery package on /home (707 GB free; /mnt/d has only 36 GB).
set -e
D=/home/bkai/delivery_r36
SRC=/mnt/d/avv
REPO=/mnt/c/Users/BKAI/an_plaza2/FastGS
mkdir -p $D/{environment,weights,renders,code}

echo "=== 1. code (working tree, no data / no outputs / no VCS) $(date +%H:%M) ==="
rsync -a --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
      --exclude 'output' --exclude 'viz' --exclude '*.log' --exclude 'assets' \
      $REPO/ $D/code/
du -sh $D/code

echo "=== 2. weights: the 12 surviving checkpoints $(date +%H:%M) ==="
cp $SRC/r14/bonsai_aa42/ckpt.pt          $D/weights/bonsai_r14_aa42.pt
cp $SRC/r14/bonsai_aa7/ckpt.pt           $D/weights/bonsai_r14_aa7.pt
cp $SRC/r14/bonsai_aa13/ckpt.pt          $D/weights/bonsai_r14_aa13.pt
cp $SRC/r14/chair_aa42/ckpt.pt           $D/weights/chair_r14_aa42.pt
cp $SRC/r14/chair_aa7/ckpt.pt            $D/weights/chair_r14_aa7.pt
cp $SRC/r14/chair_aa13/ckpt.pt           $D/weights/chair_r14_aa13.pt
cp $SRC/r17/chair_ema099_seed42/ckpt.pt  $D/weights/chair_r17_ema099_s42.pt
cp $SRC/r17/chair_ema099_seed7/ckpt.pt   $D/weights/chair_r17_ema099_s7.pt
cp $SRC/r17/chair_ema099_seed13/ckpt.pt  $D/weights/chair_r17_ema099_s13.pt
cp $SRC/r17/chair_depth_seed42/ckpt.pt   $D/weights/chair_r17_depth_s42.pt
cp $SRC/r28_members/HCM0644/ckpt.pt      $D/weights/HCM0644_r28.pt
cp $SRC/r28_members/HCM0674/ckpt.pt      $D/weights/HCM0674_r28.pt
ls $D/weights | wc -l | xargs echo "  checkpoints:"; du -sh $D/weights

echo "=== 3. renders: the ensemble inputs $(date +%H:%M) ==="
cpr(){ mkdir -p "$2"; cp -r "$1" "$2/" 2>/dev/null || true; }
for t in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  for m in $SRC/r2r9/models/${t}_*/test_png; do [ -d "$m" ] && cpr "$m" "$D/renders/$t/$(basename $(dirname $m))"; done
  for m in $SRC/r2r8/models/${t}_*/test_png; do [ -d "$m" ] && cpr "$m" "$D/renders/$t/$(basename $(dirname $m))"; done
  cpr $SRC/r22_seed101/$t/test_png  $D/renders/$t/r22_seed101
  cpr $SRC/r25_mip3d/$t/test_png    $D/renders/$t/r25_mip3d
  cpr $SRC/r28_members/$t/test_png  $D/renders/$t/r28_members
  # the r20/r21 base is only available already-averaged -- see WEIGHTS_MANIFEST.md section 3
  cpr $SRC/r20/tower_ens/$t/png_ens $D/renders/$t/r20_base_ALREADY_AVERAGED
  cpr $SRC/r21/tower_ens/$t/png_ens $D/renders/$t/r21_base_ALREADY_AVERAGED
done
for m in $SRC/r14/chair_aa42 $SRC/r14/chair_aa7 $SRC/r14/chair_aa13 \
         $SRC/r17/chair_ema099_seed42 $SRC/r17/chair_ema099_seed7 \
         $SRC/r17/chair_ema099_seed13 $SRC/r17/chair_depth_seed42 $SRC/r28_members/chair; do
  cpr $m/test_png $D/renders/chair/$(basename $m); done
for m in $SRC/r14/bonsai_aa42 $SRC/r14/bonsai_aa7 $SRC/r14/bonsai_aa13 \
         $SRC/r24_bonsai/aa101 $SRC/r24_bonsai/aa202 $SRC/r24_bonsai/aa303 \
         $SRC/r28_members/bonsai $SRC/r38_prod/s111 $SRC/r38_prod/s555 $SRC/r38_prod/s777 \
         $SRC/r38_prod/s222 $SRC/r38_prod/s333 $SRC/r38_prod/s999; do
  cpr $m/test_png $D/renders/bonsai/$(basename $m); done
du -sh $D/renders

echo "=== 4. the submission itself + the lens fields $(date +%H:%M) ==="
mkdir -p $D/submission $D/lens_fields
cp $SRC/submissions/sub_round36_bonsai13.zip $D/submission/
cp $SRC/fields_median_g1_g130/*.npy $D/lens_fields/ 2>/dev/null || true
cp $SRC/fields_chair/chair_g1_g130.npy $D/lens_fields/ 2>/dev/null || true
ls $D/lens_fields

echo "=== 5. integrity manifest $(date +%H:%M) ==="
( cd $D && find . -type f -printf '%s\t%p\n' | sort -k2 > MANIFEST.tsv )
( cd $D && sha256sum weights/*.pt submission/*.zip > SHA256SUMS.txt )
echo "  files: $(wc -l < $D/MANIFEST.tsv)"; du -sh $D
touch /home/bkai/delivery_r36/STAGED.DONE
echo "=== STAGED $(date +%H:%M) ==="
