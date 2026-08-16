# Pipeline Runbook — every stage as copy-paste commands

All tools live in `original/` (or your replica — substitute `an/`/`bach/`/`tu/`).
Conventions: `$SCENE` = scene folder with `train/images*`, `train/sparse/0`,
`test/test_poses.csv`; outputs under `runs/`.

## 0. Preflight (before ANY GPU time on a new scene)

```bash
python original/preflight.py --data_root <dataset root> --scenes <SCENE_NAME>
```
Validates layout, camera model (|k1| < 0.2), CSV/W/H consistency. Non-zero exit = fix
the data problem first.

## 1. Train

```bash
python original/train_gsplat.py \
  --source $SCENE/train --images images \
  --out runs/$NAME \
  $(cat configs/recipes/<recipe>.args) \
  --seed 42 --ckpt_every 2000
```
- Distorted cameras (k1 ≠ 0): the recipe must include `--ut`; train on the ORIGINAL
  distorted images (`--images images`, never `images_undist`) — "virgin pixels".
- Pinhole: no `--ut` (antialiased mode is the default rasterize path).
- `--ckpt_every` writes a rolling `ckpt_latest.pt` (atomic, render-ready) — mandatory
  for unattended runs.
- Recipes: `configs/recipes/*.args` (provenance in each file). Fork-ensemble members:
  same trunk, then vary schedule/scale_reg for the last 10–25% (config jitter, not just
  seed).

## 2. Render test poses

```bash
python original/render_gsplat.py \
  --ckpt runs/$NAME/ckpt.pt \
  --csv $SCENE/test/test_poses.csv \
  --out runs/$NAME/test_render --png_dir runs/$NAME/test_png
```
- Auto-detects UT checkpoints; renders native distorted. Large negative k1: add
  `--ut_render warp --distort auto --sparse $SCENE/train/sparse/0`.
- ALWAYS pass `--png_dir` — lossless PNG archives are what ensembling/scoring/zip
  building read, and they are the artifact that reproduces a submission.

## 3. Eval split (recipe selection without test GT)

```bash
python original/make_eval_split.py --scene $SCENE --out runs/split_$NAME   # isolated every-k
# train on the split's train_sub, render its eval_poses.csv, then:
python original/eval_score.py --render_dir <renders> --gt_dir <heldout GT> --tag $NAME
```
Rules: isolated evenly-spaced holdout only; a handful of candidates (eval-overfit is
real); an eval win needs ≥ +0.5 scene-pts to be worth a submission slot; structural
wins transfer ~1×, recipe nudges ~0.3×.

## 4. Lens-field correction (per scene, after members exist)

```bash
python original/render_train.py --ckpt runs/$NAME/ckpt.pt --source $SCENE/train --images images --out runs/$NAME/train_png
python original/fit_field.py  --render_dir runs/$NAME/train_png --gt_dir $SCENE/train/images --out runs/fields/$SCENE.npy
python original/apply_field.py --field runs/fields/$SCENE.npy --in_dir <png_ens> --out_dir <png_field> --strict
```
- Fit on TRAIN photos only (legal); the field is fit on the SAME render path used for
  test. Amplitude gain: Rounds 1–2 shipped ×1.30 (train fit undershoots ~30%) — treat
  as a prior, re-derive on the new rig before shipping. Guard: max |d| < 8 px.

## 5. Ensemble

```bash
python original/ensemble_renders.py --dirs runs/A/test_png runs/B/test_png runs/C/test_png \
  --out runs/ens/png_ens [--weights 1 1 1] [--names_from $SCENE/test/test_poses.csv]
```
- Pixel-mean in float32, round once (baked in). Members within ~0.15 of the best
  single, current generation only. Order of operations: ensemble → (energy restore,
  gated) → lens field → zip.

## 6. Build + verify the zip

```bash
python original/build_submission_zip.py --scene_dirs SCENE1=<png_dir1> SCENE2=<png_dir2> \
  --data_root <dataset root> --out runs/sub_rN.zip
python original/verify_zip.py --zip runs/sub_rN.zip --data_root <dataset root>
```
- Cap is budgeted in MiB (350 MiB = 367,001,600 B). verify_zip checks names, dims, CRC,
  size, single-generation JPEG. Green verify → notify the team → A HUMAN SUBMITS.

## 7. Score locally (when GT exists for calibration scenes)

```bash
python original/score_submission.py --sub runs/sub_rN.zip --gt_root <root> --device cuda
```
Composite = 100·[0.4(1−LPIPS_vgg) + 0.3·SSIM + 0.3·PSNR/50]. Uses the repo SSIM —
comparable to the calibrated Rounds 1–2 numbers.
