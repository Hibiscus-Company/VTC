#!/bin/bash
# SSAA sweep on HCM0181 (exp01): render 5 variants, score each
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2
cd /mnt/c/Users/BKAI/an_plaza2/FastGS
export CUDA_VISIBLE_DEVICES=1
SCENE=HCM0181
CSV=~/data/phase1/public_set/$SCENE/test/test_poses.csv
SPARSE=~/data/phase1/public_set/$SCENE/train/sparse/0
OUT=~/ssaa
rm -rf $OUT && mkdir -p $OUT

run() {  # tag ss filter
  echo "=== render $1 (ss=$2 filter=$3) ==="
  python render_test_poses.py -m output/${SCENE}_g2 --csv $CSV \
    --out $OUT/$1/$SCENE --mult 0.7 --distort auto --sparse $SPARSE \
    --force_png --supersample $2 --ss_filter $3 2>&1 | tail -2
}
run ss1      1.0 lanczos
run ss1p5lz  1.5 lanczos
run ss2lz    2.0 lanczos
run ss2box   2.0 box
run ss3lz    3.0 lanczos

for tag in ss1 ss1p5lz ss2lz ss2box ss3lz; do
  echo "=== score $tag ==="
  python score_submission.py --sub $OUT/$tag --device cuda:0 2>&1 | grep -E "HCM0181|Score"
done
echo "DONE"
