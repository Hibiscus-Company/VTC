---
name: 3dgut-breakthrough
description: 3DGUT distorted-space training is the new base method + LB progression/calibration for the NVS competition
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
---

3DGUT (gsplat Unscented-Transform, distorted-space) is the ADOPTED base method for the VAR2026 NVS competition (supersedes plain FastGS/gsplat as the strongest single model). Validated 2026-07-14.

- **What it is**: `gsplat_track/train_gsplat.py --ut` = classic rasterize_mode + with_ut + with_eval3d + radial_coeffs=[k1,0,0,0,0,0] (k1 from cameras.bin SIMPLE_RADIAL, per-scene). Train on the ORIGINAL distorted `train/images` (`--images images`, NOT images_undist) — "virgin pixels", no undistort INTER_CUBIC resample generation. Render native distorted via `render_gsplat.py` (auto-detects UT ckpt; `--ut_render native`). NO DistortionWarp on native path.
- **Why it wins**: pure LPIPS gain (virgin GT). Public 5/5 scenes mean +0.67 over FastGS champ; gain GROWS with scene sparsity (HCM0181 +0.07 dense … HCM0193 +1.04). A single UT model beats the old 3-member FastGS ensemble.
- **UT recipe ladder (HCM0181 single)**: 30k/cap5M 75.21 → cap8M 75.58 → 60k/cap8M **75.90**. cap8M + 60k = production recipe. Both capacity+schedule knobs additive.
- **Ensemble**: UT and FastGS are DIFFERENT families (UT carries LPIPS, gates carry PSNR/SSIM). Seed jitter works in UT family (MCMC data-order decorrelation) unlike FastGS (safe_state pins seeds). **Family-weighting beats uniform** (round-3 equal-weight null was intra-family only): w_UT≈0.5-0.6 optimal, +0.23 over uniform. `ensemble_renders.py --weights` added. Best local composition m6ut60 (3-4 gates + 2 UT, w_UT 0.6) = 77.04 on HCM0181.
- **negative-k gotcha**: HNI0131 & HNI0265 have k1≈−0.115 (14× |k|, opposite sign vs all other scenes +0.008..+0.014). UT-native forward distortion FOLDS (phantom corner gaussians). Render these two via `--ut_render warp` (needs the with_ut=ut,with_eval3d=ut fix in render_gsplat + `--distort auto --sparse`). Empirically native≈warp (~2-3/255) so gsplat likely culls the fold, but ship warp to be safe.
- **LB progression (private set)**: R1 single 74.348 → R2 mean2 76.166 → R3 mean3 76.4005 → **R5 (3 gates + UT w0.5) 77.2964**. Private runs ~+0.37 above HCM0181 public proxy for UT-heavy ensembles. Top1 = 85.94 (gap 8.6).
- **IBR (depth-guided photo reuse) REJECTED**: naive geometric warp of real train photos into test poses = −4 to −7 (monotone: more photo = worse). Diagnostic proved NO systematic offset (conventions correct) — failure is stochastic per-pixel depth-noise misregistration the low-pass gate can't catch. Flow-corrected retry (`ibr_render.py --flow_correct`, cv2 DIS) was the last shot. Takeaway: top1's 85+ is NOT naive photo reuse.
- Standing lever ranking to close the gap: per-pose local finetune (`perpose_finetune.py`) > UT recipe scaling on private members > idea-9 per-scene HPO.

[[nvs-competition-setup]] [[ensemble-strategy]] [[fastgs-env-and-build]]
