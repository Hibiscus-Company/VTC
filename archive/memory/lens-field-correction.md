---
name: lens-field-correction
description: Renders are misregistered from photos by a fixed sub-pixel 2D field; fitting it on TRAIN photos and warping renders is worth ~+0.9 dB PSNR (+0.55 score pts) with no retraining
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-07-28T02:08:07.123Z
---

Our 3DGS renders are misregistered against the real photos by a **small, fixed,
image-independent 2D displacement field** (mean ~0.2-0.3 px, up to ~2 px at the frame
edge). Correcting it is a post-render warp with NO retraining. Found 2026-07-14; the
single biggest lever in the project (cf. +0.10 pts for a 13-GPU-hour member retrain).

**CONFIRMED ON THE LEADERBOARD: R6 77.6629 -> R7 78.39740 (+0.7345).** PSNR +0.556 dB,
SSIM +1.31, LPIPS flat. Note **SSIM paid MORE than PSNR** (+0.392 vs +0.333 pts) both on
public and private — a sub-pixel misregistration fix is a STRUCTURAL correction, and SSIM
rewards it harder than MSE does. Public predicted +0.88, private delivered +0.73.

**Root cause:** COLMAP hands us `SIMPLE_RADIAL` = a single k1. We pass gsplat
`radial_coeffs = [k1,0,0,0,0,0]` and every higher-order radial + tangential + thin-prism
term is pinned to zero. gsplat supports the full OpenCV model but its distortion
coefficients **take no gradient** (probed: `.grad` is `None` after backward) — they're
CUDA-kernel constants. So they cannot be learned by backprop; the field must be fitted
and applied outside the renderer.

**Why it's legal (Rule 10):** fit the field ONLY on train renders vs train photos — never
test GT. One automatic per-scene field, applied programmatically. It's a lens/pipeline
calibration, not per-image editing. See [[nvs-competition-setup]].

**Why it transfers:** the field is a camera property, not an image artifact.
- Two disjoint halves of the test set fit the same radial curve to **0.08 px (r=0.9977)**.
- Held-out gain == in-sample gain (no overfitting).
- A field fit on TRAIN photos alone recovers **95%** of what an oracle test-fitted field
  would (train-vs-test field corr dx +0.94 / dy +0.88). The model DOES partially absorb
  the bias on train views (magnitude ratio 0.72) but the SHAPE is preserved.

## AMPLITUDE: ship the field at gain 1.30, NOT unity (r29, +0.1615 on the LB)

The train fit systematically **UNDERSHOOTS**. At train poses the model has already absorbed
part of the misregistration into its own geometry, so DIS flow sees only the UNABSORBED
remainder; at novel poses the absorbed warp does not cancel and the FULL displacement
appears. The "magnitude ratio 0.72" noted above IS this effect — 1/0.72 = 1.39 — and it sat
in this file unexploited from R7 to r28 while we shipped unity gain.

Confirmed two independent ways, then on the LB:
- SCORED (production harness, real test GT, full chain): 1.15 +0.0915 / **1.30 +0.1328** /
  1.45 +0.1231. Plateau 1.25-1.40; 1.60 costs -0.05, 1.80 -0.13.
- GEOMETRIC (no scores): best scalar alpha = <M,f>/<f,f> = 1.3185, R2 0.901 -> 0.957.
  Verifier got alpha* 1.34-1.43 and |M|/|f| 1.26-1.39 on all five public towers.
- **LB: r28 77.5029 -> r29 77.6644 (+0.1615)**, stacked with 2 other changes. Fingerprint
  matched the forecast's direction on all three submetrics and OVER-delivered on PSNR/SSIM.

**Why every train-side diagnostic missed it:** our leave-one-view-out protocol reports the
optimum as 1.00, because at train poses the absorbed warp is exactly what LOVO holds out.
The defect is structurally invisible to any train-fitted check. See [[production-harness]] —
this is the second time real test GT overturned a train-side conclusion.

**How to apply:** scale the saved .npy by the gain; nothing else changes. Guard: assert
post-scaling max|d| < 8.0 px (ours land at 1.086-1.470). gt_dir stays a TRAIN dir so
`apply_field.py --strict` still passes — only a SCALAR is calibrated on public test GT.

**How to apply:** `gsplat_track/render_train.py` (render a model at its own train poses)
→ `fit_field.py` (mean dense DIS flow, train render → train photo, saved .npy)
→ `apply_field.py` (warp renders, write lossless PNG) → `build_submission_zip.py`.
Use the full 2D field, NOT a radial polynomial: the measured curve is non-monotone
(+0.44 px bump at r=0.67, −1.5 px at the corner) and a 3-term odd polynomial captures
only half of it.

**Also settled by the same diagnostics:** the residual error decomposes as LENS +0.40 pts
/ POSE +0.14 / GEOM +0.74. **Idea 6 (pose refinement) is dead** — its oracle upper bound
is only +0.14 pts, so the HIGH-RISK gated idea isn't worth the rule risk. The remaining
big lever is GEOM (+0.74) → dense/MVS seeding. Per-image PSNR has no bad tail (std 1.25),
so "clean up the worst views" is also dead. See [[3dgut-breakthrough]].
