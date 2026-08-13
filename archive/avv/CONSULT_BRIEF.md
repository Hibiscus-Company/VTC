# Consult brief — VAR 2026 "BTS Digital Twin" (3DGS novel-view synthesis)

I'm stuck and want an outside view. Below is everything material, with the measurements.
Please attack the reasoning, not just the code.

---

## The task and the metric

Per-scene 3D Gaussian Splatting novel-view synthesis of telecom towers from drone photos.
Submit rendered JPEGs at given test poses; 8 private scenes; ≤350 MB zip.

    Score = 100 * [ 0.4*(1 - LPIPS_vgg) + 0.3*SSIM + 0.3*PSNR/50 ]

(LPIPS backbone = vgg, psnr_max = 50 — both calibrated from official leaderboard breakdowns.
Our local scorer reproduces our leaderboard score to 5 decimals: 78.39737 vs 78.39740.)

**Where we are: 78.3974.  Top-1: 86.1220 and still moving.  Gap: 7.7 points.**

Our submetrics: **PSNR 26.470 · SSIM 0.8816 · LPIPS 0.0983**

## Hard rules (from the organizer; we will not cross these)

No external data or imagery of these scenes. No inferring or looking at test GT. No manual
per-image editing. Pretrained VGG/AlexNet as a loss or metric is fine.

(Public-set test GT is provided and legal to use for diagnosis/scoring. Private test GT does
not exist for us.)

## Method (current)

3DGUT (gsplat, `with_ut=True, with_eval3d=True, rasterize_mode="classic"`) trained on the
ORIGINAL DISTORTED photos, 60k iters, MCMC, cap 8M gaussians, LPIPS in the tail.
Final submission = family-weighted pixel-mean ensemble of 5 members (3 FastGS + 2 3DGUT,
w_UT = 0.6) + a per-scene sub-pixel "lens field" correction (below).

## The one big win, for context on what "a real lever" looks like here

**All 13 scenes share ONE physical camera** (identical W/H, identical principal point, focal
within 0.3%) — yet COLMAP fit a DIFFERENT k1 to each, and on two scenes it landed at
k1 = −0.115, a 15× outlier of the OPPOSITE sign. A single-k1 model cannot express this lens,
so BA lands in a different local optimum per scene.

gsplat's distortion coefficients **take no gradient** (verified: `.grad is None`), so they
cannot be learned. So we measured the residual misregistration directly (dense optical flow,
render → photo), found it is a **fixed, image-independent 2D field** (two disjoint folds of the
test set agree to 0.08 px, r = 0.9977), fit that field on **TRAIN photos only** (Rule-10 clean),
and warped the renders.

**Result: +0.8836 on 5/5 public scenes; +0.7345 on the private leaderboard (77.66 → 78.40),
with NO retraining.** Notably SSIM paid more than PSNR (+0.39 vs +0.33 pts) — a sub-pixel
misregistration fix is structural.

## THE IMPASSE — what 86 requires, and six bounds saying we can't get there

86.12 requires **PSNR ≥ 30 dB even at NEAR-PERFECT LPIPS/SSIM**, and ≈33.7 dB at a
realistically-excellent LPIPS .05 / SSIM .93. We are at **26.47 dB**.

Every hypothesis we could construct is bounded:

| hypothesis | measurement | verdict |
|---|---|---|
| lens / intrinsics residual | +0.73 on the LB | **harvested** |
| global pose offset | oracle: **−0.02 dB** | nothing there |
| per-image pose error | oracle (homography vs test GT): **+0.14 pts**, and it should NOT transfer (test poses carry the same noise; a noise-blurred map is near-MMSE-optimal) | dead |
| photo reuse (paste nearest train photo) | **0 / 290 test views** beat our render. Paste mean 10.2 dB vs render 25.0 dB. Best near-duplicate (0.41°, baseline 0.029) pastes at 14.0 dB | dead |
| rolling shutter (drone = moving platform; gsplat supports it natively) | row-linear flow slope **0.32 px** over the full frame; alignment with drone velocity **cos = +0.089**; 37.5% aligned / 30.0% opposed / 32.5% orthogonal = **random** | not velocity-coupled — dead |
| per-view appearance / exposure | oracle affine colour correction: **+0.09 dB** | dead |
| **PERFECT per-image registration (CHEATING: dense flow fit against test GT)** | **+2.13 dB → 28.6 dB** | **still short of 31** |

**Even outright cheating with the test ground truth does not reach 31 dB.**

Other closed levers (all measured, all negative): supersampling/SSAA (monotone worse,
−0.88 to −6.73), bilateral-grid and affine per-view appearance (−7.6, −0.91), depth-guided IBR
(−4 to −7), per-test-pose finetune (no signal), SH-degree clamp (monotone worse: sh 0/1/2/3 →
67.8/69.1/72.5/75.9), MVS seeding (MCMC relocation erases init by 30k), JPEG/packaging
(q98 costs only −0.05 vs lossless; all-q100 is physically impossible under 350 MB).

## The context that makes me think the cave is sealed

**26.5 dB is SOTA-typical 3DGS for outdoor scenes.** Published 3DGS gets ~24–26 dB on
Mip-NeRF360 outdoor, ~23–25 on Tanks & Temples. **31–34 dB is above what any published NVS
method achieves on real photos at this view spacing** (our test views sit a median 1.2° in
view-direction but ~12% of scene-scale in BASELINE from the nearest train view — genuinely
novel views, as D12 proves).

Also on the record, unresolved: top-1 jumped **76.75 → 85.94 in ONE discrete step** (+9.2),
while our largest single controlled gain across ~35 experiments is +1.82. And the full-res
originals of these exact scenes **demonstrably exist** — `images.bin` stores keypoints at
5280×3956 while the delivered cameras are ÷4 (1320×989), i.e. the organizers COLMAPped at full
resolution and gave us quarter-res. We cannot see other teams' submetrics.

## The ONE thing still untested, and it has a mechanism

**Our loss has never optimized PSNR.** Standard 3DGS trains `0.8*L1 + 0.2*(1-SSIM)` —
**L1 optimizes the MEDIAN, not the mean.** The exact score-matched loss is

    minimize  0.4*LPIPS + 0.3*(1-SSIM) + 0.02606*ln(MSE)        [0.02606 = 0.06/ln10]

which is *exactly* the competition score (log-MSE is self-scaling: grad 0.026/MSE = 8.24 at
25 dB, 26.06 at 30 dB). Running now as a warm-start refit. I expect +0.2–0.5, not +7.

Also unmeasured: our **train-view** PSNR with the current model. The last measurement (older,
weaker model) was train 27.09 / test 24.47 — a 2.6 dB gap, and a train fit of 27 dB is LOW for
8M gaussians on 240 images. Standard 3DGS usually fits train to 30–35 dB. **If we are genuinely
underfitting the training data, that is the one number that would point somewhere new.**

---

## What I want from you

1. **Is my bound sound?** Specifically: perfect per-image registration (oracle dense flow
   against test GT) yields only 28.6 dB. Is there a class of error that oracle does NOT
   bound, and that could be worth several dB?
2. **Is 31–34 dB reachable at all on 240 drone photos of a tower with ~12%-of-scene-scale
   test baselines?** If yes, by what method class? If no, say so plainly.
3. **Is the low train PSNR (27 dB) the real signal I'm missing?** What limits 3DGS train fit
   to 27 dB with 8M gaussians, and is pushing it to 33 dB (a) possible and (b) likely to
   transfer to test?
4. **Anything in the six bounds above that is measured wrong, or a hypothesis nobody has
   named?**

Be blunt. If the honest answer is "you cannot get to 86 legitimately on this data," I would
rather hear it now and spend the remaining two weeks maximizing rank instead of chasing a
number that may not be reachable under the rules.
