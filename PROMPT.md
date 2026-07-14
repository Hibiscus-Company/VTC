# Research Brief — Novel-View Synthesis for a Drone/BTS "Digital Twin" Competition

You are a 3D reconstruction / neural rendering research assistant. Read this
brief in full, then produce three deliverables (see **Your Task** at the end):
1. **Related SOTA** that could raise our score on THIS dataset.
2. **Existing, ready-to-use methods / repos** we can integrate quickly.
3. **New, harder research directions** worth attempting (novelty allowed).

Ground every recommendation in the measured characteristics below — we have
already ruled out several "generic outdoor" priors with data. Do not repeat
advice that our measurements contradict.

---

## 1. The competition

- **Task**: VAR 2026 "Digital Twin for BTS (cell-tower) stations", Round 1.
  Given per-scene drone training images + COLMAP sparse reconstruction +
  camera intrinsics, **render RGB at a list of target test poses**
  (`test_poses.csv`). Pure novel-view synthesis.
- **Scored only on image quality** (no speed/size/VRAM constraints, no stated
  submission size limit).
- **Metric** (per test image vs held-out GT, averaged):

  `Score = 0.4·(1 − LPIPS) + 0.3·SSIM + 0.3·clamp(PSNR / psnr_max, 0, 1)`

  - LPIPS = VGG. Lower better.
  - **`psnr_max = 50.0` — CONFIRMED** by solving the official Round-1 breakdown
    (Score 0.7434810 = 0.4·(1−0.141134) + 0.3·0.836116 + 0.3·(24.84994/50)).
    **Consequence (now hard-verified): the PSNR term is only 20.1% of score and
    near-saturated — +1 dB ≈ +0.0060; a −0.01 LPIPS gain ≈ +0.0040 and LPIPS
    (weight 0.4) has by far the most headroom → LPIPS is the single
    highest-leverage metric.** Optimize perceptual quality, not raw PSNR.
- **Submission**: zip of `<scene>/<image_name>` where `image_name` is taken
  verbatim from `test_poses.csv` (JPEG, ~700 KB/img is fine).

## 2. Data (13 scenes)

- **5 public** (HAVE test GT → local scoring): hcm0031, hcm0034, HCM0181,
  HCM0193, HCM0204. **8 private** (no GT): HCM0249/0254/0276/1439,
  HNI0131/0265/0366/0437.
- Per scene: ~100–240 train imgs, 26–60 test poses. Images **1320×989**
  (downscaled ~1/4 from original drone frames). Single **SIMPLE_RADIAL**
  camera, principal point centred.
- **Test poses are INTERPOLATED within the drone flight** (verified: every test
  pose is byte-identical to an entry in the scene's COLMAP `images.bin`). This
  is the *easy* NVS regime (not extrapolation). Flight pattern (from viz):
  orbit + overhead grid around the tower.

## 3. Current pipeline & results

- **Backbone**: FastGS (CVPR 2026), a *speed-optimized* 3DGS framework
  (multi-view-consistent densification + pruning, "compact box" tile culling
  via `mult`). Repo is this directory. Runs on 2× RTX 5070 Ti (16 GB, sm_120).
- **Preprocessing**: images undistorted (SIMPLE_RADIAL → pinhole) before
  training; test renders warped back into the distorted geometry to match GT
  (`--distort auto`). This alone gave **+0.031 score** (hcm0034 0.7207→0.7513).
- **Config**: `densification_interval 100, grad_abs_thresh 0.0004,
  highfeature_lr 0.04, mult 0.7, 30k iters, --data_device cpu`.
- **Public scores (current submission, ≈74 overall)**:
  hcm0031 .7369 · hcm0034 .7513 · HCM0181 .7440 · HCM0193 .7413 · HCM0204 .7477
  (typical PSNR ~24–25, SSIM ~0.83–0.85, LPIPS ~0.13–0.15).
- **Private scores 74.34810** : PSNR = 24.84994; SSIM(*100) = 83.6116; LPIPS(*100) = 14.1134

## 4. Measured dataset characteristics (EDA) — use these, they override priors

| Property | Measured | Implication |
|---|---|---|
| **Exposure drift** (CV of per-image luminance across a flight) | **2.1–5.1% (LOW)** | Appearance/exposure embedding has little to model. |
| **Sky fraction** | **15–22%** of frame | Large infinite-depth region → floaters → hurts LPIPS. |
| **Camera scale spread** (far/near dist to scene centroid, p95/p5) | **2.3–4.1×** | Real aliasing/dilation regime → anti-aliasing (Mip) justified. |
| **COLMAP point density** (pts / bbox volume) | **0.01–0.27 (27× spread)** | Sparse scenes (HNI0265 0.01, HNI0131 0.04, HCM1439 0.05) may need more aggressive densification. |
| **Scene extent** | 7.9–10.6 (uniform) | One scale-config transfers; bounded-ish. |
| **Radial distortion k** | ~+0.008 typical; **−0.115** for HNI0131/0265 | Handled by undistort step. |
| Dynamic objects | not yet measured | To measure with a detector pass. |

## 5. Experiments already run (do not re-propose these)

- **Densification is the main quality lever, and it works through LPIPS.**
  Lowering `grad_abs_thresh` (FastGS's absolute-gradient densification
  threshold) 0.0004→0.0002 gives **+0.004–0.006 score, driven by LPIPS**
  (e.g. hcm0034 LPIPS 0.135→0.124, 1.6M→3.1M gaussians). PSNR barely moves.
  **The optimum is scene-dependent** (denser scenes saturate ~2.6M gaussians;
  sparser want more) → we will apply a density-adaptive rule.
- **FastGS already implements AbsGS-style densification** (backward pass
  accumulates `fabs(dL_dmean2D)` into extra channels that `grad_abs_thresh`
  thresholds). So AbsGS/Pixel-GS as separate add-ons are largely **redundant**.
- **`mult` (compact-box tile culling) is quality-neutral** at 0.7 vs 1.0
  (+0.0003…0.0012) — the speed trick isn't costing us quality.
- **Extending `densify_until_iter` past 15k CRASHES** (collapses to 0 gaussians)
  because it overlaps FastGS's post-15k multi-view prune schedule. Densifying
  longer requires refactoring the prune schedule first.
- **Appearance embedding is deprioritized** for two reasons: (a) drift is low
  (§4); (b) *structural NVS problem* — per-image appearance codes don't exist
  for novel test poses, so at test time only a mean/canonical code is usable →
  ~zero upside, real downside if test exposure ≠ mean.

## 6. Current planned direction (for you to critique & extend)

1. **Free win now**: density-adaptive `grad_abs_thresh` (0.0002 default; 0.0001
   for density<0.1 scenes) → expected ~0.744.
2. **Mip-Splatting** as the first real add-on (justified by 2.3–4.1× scale
   spread; targets LPIPS/SSIM; 3DGS-family so composes with FastGS).
3. **Sky handling**: segment sky (SegFormer/FastSAM) → mask its training loss
   to kill floaters. **Caveat we already identified**: sky pixels ARE scored at
   test time, so masking alone leaves sky unmodeled → must pair with a
   sky-sphere/env-map or background-colour match, and validate on public GT.
4. Skip appearance embedding; measure-then-decide on dynamic-object masking.

## 7. Hardware / practical constraints

- 2× RTX 5070 Ti 16 GB (Blackwell sm_120), CUDA 12.8, torch 2.7.1. ~10 min per
  30k-iter FastGS train per scene. CUDA extensions compile locally.
- Anything proposed should ideally **compose with the FastGS explicit-Gaussian
  rasterizer** (we have working CUDA forward/backward we can extend), or be a
  **loss/data/post-process add-on**. Full representation swaps (e.g. Scaffold-GS
  anchors+MLP, pure NeRF) are high-cost — flag them as such.
- We can only locally validate on the 5 public scenes → **generalization
  matters; avoid per-public-scene overfitting**. Prefer methods/params that are
  either single-robust or driven by a measurable scene feature.

---

## Your Task

Produce three sections, each item with: *what it is · why it fits THIS data
(cite the §4/§5 evidence) · expected effect on LPIPS/SSIM/PSNR · integration
cost against FastGS · a concrete first experiment we can run on the 5 public
scenes.* Rank within each section by expected score-per-effort.

1. **Related SOTA** (2024–2026 GS/NVS): anti-aliasing (Mip-Splatting and
   successors), better densification, floater/geometry regularization,
   aerial/large-scale GS, view-dependent appearance — but only what the data
   supports. Explicitly say if any is redundant with FastGS (§5).
2. **Existing ready-to-use methods/repos**: name repos, license, how they graft
   onto a FastGS-style CUDA rasterizer, and known photometric-vs-geometry
   trade-offs (e.g. does 2DGS lose NVS PSNR?).
3. **New / harder research directions**: novel ideas specific to this setting —
   interpolated drone poses + BTS thin metal structures + 15–22% sky +
   LPIPS-dominated scoring. Higher risk, potentially higher payoff. Be concrete
   about the hypothesis and how to falsify it on the public set.

Be skeptical and quantitative. If our current plan (§6) is wrong or misordered,
say so with reasoning.
