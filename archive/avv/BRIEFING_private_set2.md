# private_set2 — cold-start briefing for a fresh session

Purpose: everything a new Claude needs to start finding ideas on this competition without
re-deriving four weeks of work or repeating measurements that already have answers.
Written 2026-07-29. Incumbent best submission: **r32 = 77.6907**.

---

## 1. The task and the metric

VAR 2026 "BTS Digital Twin": per-scene 3D Gaussian Splatting novel-view synthesis. For each
scene you are given training images with COLMAP poses, and a CSV of *test* camera poses. You
render one image per test pose and submit them all in a zip. The organizers score against
held-out test photos you never see.

```
score_scene = 100 * [ 0.4*(1 - LPIPS_vgg) + 0.3*SSIM + 0.3*PSNR/50 ]
final       = unweighted mean over the 7 scenes
```

The submetric arithmetic you will use constantly:

```
dScore = -40*dLPIPS + 30*dSSIM + 0.6*dPSNR(dB)          # per scene, /7 for the final
```

**LPIPS dominates.** A 0.01 LPIPS improvement on one scene is worth +0.057 final; the same
relative effort on PSNR is worth a fraction of that. Verified: the grader is a plain unweighted
mean of each submetric over 7 scenes (reconstructing r31's 77.6804 from its published
PSNR/SSIM/LPIPS closes to 4 dp), so residual algebra across scenes is legitimate.

Submission: **386 JPEGs**, `<scene>/<image_name>` exactly matching each `test_poses.csv`,
zip ≤ **350 MiB = 367,001,600 bytes** (MiB, not MB — this cost a round once).

---

## 2. The seven scenes

Root: `/mnt/d/avv/data/phase1/private_set2/<scene>/`
with `train/images/`, `train/sparse/0/` (COLMAP binary), `test/test_poses.csv`.

| scene | train | test | resolution | camera | character |
|---|---|---|---|---|---|
| HCM0421 | 240 | 60 | 1320x989 | SIMPLE_RADIAL k1≈+0.009 | drone orbit, telecom tower |
| HCM0539 | 240 | 60 | 1320x989 | SIMPLE_RADIAL | ditto |
| HCM0540 | 240 | 60 | 1320x989 | SIMPLE_RADIAL | ditto |
| HCM0644 | 240 | 60 | 1320x989 | SIMPLE_RADIAL | ditto, near-nadir heavy |
| HCM0674 | 240 | 60 | 1320x989 | SIMPLE_RADIAL | ditto |
| chair | 205 | 58 | 720x1280 **portrait** | SIMPLE_PINHOLE | indoor phone video, office chair; DoF + motion blur |
| bonsai | 248 | 28 | 1920x1080 | SIMPLE_PINHOLE | indoor video, plant on **glossy black glass** table (mirror reflections), motion blur, 63-level auto-exposure drift |

Scale note: organizers COLMAPped at full res and delivered downscaled — towers 1/4, chair 1/1.5
(COLMAP ran at 1080x1920), bonsai 1/1. `cameras.bin` is already rescaled; `images.bin` `xys` are
NOT (they are at original resolution). Nothing in the shipped pipeline consumes `xys`.

**`images.bin` contains the test frames' poses and ~2.3k SIFT keypoints each, plus 11–98 extra
undelivered frames per tower.** This is a compliance gray zone. It was quantified on 16/07 at
**+0.048 points on top of the shipping path** and REJECTED. Leave it closed.

---

## 3. Rule 10 (inviolable)

No external data or imagery of these scenes. No looking at or inferring from private test GT.
No manual per-image editing. Pretrained VGG/AlexNet as a loss or metric is fine.

**Legal ground-truth surfaces** (use these; there are no others):
- `/mnt/d/avv/data/phase1/public_set/{HCM0181,HCM0193,HCM0204,hcm0031,hcm0034}` — different
  scenes, but they ship **test GT images**, so you may score on them freely.
- Each scene's own **train photos** — legal GT. Train→test transfers at slope ~0.80 with a
  near-constant ~2 dB gap, so train-view scoring ranks scenes correctly.
- Eval-splits at `/mnt/d/avv/evalsplit/{bonsai,bonsai2,chair,HCM0181,HCM0421}` — isolated-hole
  train/eval splits. **See the warning in §6 before trusting these.**

---

## 4. What we currently ship

Per scene: **ensemble mean of k seed-diverse models → energy restoration → lens displacement
field → single-generation JPEG q100 / 4:2:0 / progressive / optimize.**

- **Base models**: gsplat MCMC + 3DGUT ("--ut", distorted-space unscented transform), 60k iters,
  cap_max 8M, refine_stop 50k, noise_stop 50k. **Except bonsai**: 30k iters, cap 5M,
  refine_stop 15k, noise_stop 8k, no `--ut`, no mip3d.
- **Ensemble**: towers k=8 (4:1:1 weighting over an r22 6-member mean, a mip3d member, and an
  r28 member), chair 8 members, bonsai 7.
- **Energy restoration** (`energy_restore.py`): rescales the finest Laplacian band of the mean by
  `r = sqrt(1 + (k/(k-1)) * E(L0_i - L0_mean)/E(L0_mean))`, clamped at 4;
  `out = mean + lam*(r-1)*L0(mean)`. Shipped lambdas: towers 1.0, chair 0.40, bonsai 0.25.
- **Lens field**: a fitted sub-pixel displacement field, fit on TRAIN photos vs TRAIN renders,
  gaussian-smoothed sigma=1, applied at **gain 1.30** (the train fit undershoots ~30%),
  resampled lanczos4. Worth **+0.7345** on the leaderboard historically — the single biggest
  post-processing win. Shipped on all 5 towers and chair; **dead on bonsai** (LOVO n=40, every
  variant ≤ no-field).
- **Encode**: q100/ss2 everywhere except HCM0421 at q99 (byte budget). q99-vs-q100 measured on
  290 real GT pairs: |effect| ≤ 0.001. Lossless PNG is worth only +0.0196 and costs 2.6x the cap.

Key scripts live in `/home/bkai/.claude/jobs/1c9cf7e9/tmp/` (build_r*.sh, energy_restore.py,
lapfuse.py, lens/) and `/mnt/c/Users/BKAI/an_plaza2/FastGS/gsplat_track/` (train_gsplat.py,
render_gsplat.py, apply_field.py, ensemble_renders.py).

---

## 5. Graded history — every number is a real submission

| round | score | what changed |
|---|---|---|
| r28 | 77.5029 | energy restoration introduced, all 7 scenes (+0.1005 over r27) |
| r29 | 77.6644 | 7 new members + field gain 1.30 + video lam=0 |
| r30 | 77.6614 | +2 ensemble members (k=8→10) — **LOST** |
| r30c | 77.6595 | JPEG q98/4:4:4 — **LOST** |
| r31 | 77.6804 | gauss-1 field at gain 1.30 + chair field |
| **r32** | **77.6907** | **chair lam 0→0.40, bonsai 0→0.25 — BEST, INCUMBENT** |
| r33 | 77.6758 | tower "deadband fix" + HCM0421 q100 — **LOST −0.0149** |
| r33a | 77.6867 | bonsai lam 0.25→0 — **LOST −0.0041** |
| r34 | 77.6855 | chair lam→0.60, bonsai→0.50 — **LOST −0.0052** |

r32 submetrics: PSNR 26.640357, SSIM 87.3191, LPIPS 11.2230.

**r32 is a bracketed local optimum.** Four consecutive rounds probing four different directions
all lost. The video-lambda curve is measured on the leaderboard on both sides of r32.

---

## 6. The two rules that decide whether a measurement means anything

### The transfer rule
**PER-IMAGE** operators (act on one image; value independent of the ensemble pool) transfer to
the leaderboard at ~1x — lens field, field smoothing, resampling kernel: predicted +0.0127,
delivered +0.0160.
**POOL-DEPENDENT** operators (value scales with how much ensemble members disagree) do NOT —
predicted +0.100, delivered −0.0049. The public harness pool disagrees roughly 2x more than our
shipped private pool, so anything that pays off under disagreement over-reads there. **Divide a
pool-dependent harness number by ~20 before believing it.**

### Proxy splits can be sign-INVERTED
`chair_eval` and `bonsai_eval` are built from **different models** than the production members.
For the pool-dependent energy operator they were **wrong in sign, twice**:
- chair_eval said lam>0 monotonically harmful. Leaderboard: chair lam=0.40 is worth **+0.0063**.
- bonsai_eval said the same. Leaderboard: bonsai lam=0.25 is worth **+0.0041**.

So: for pool-dependent things, only shipped-pool / leaderboard evidence counts. For
*reconstruction quality* (same model class, held-out views) the eval splits are reasonable.

**Also: `scripts/eval_score.py` CHANGED since 17/07 — identical bytes score 0.95 lower today.
Any eval-split number recorded before 17/07 is incomparable to anything you measure now.**

---

## 7. Closed axes — do not re-propose without naming a defect in the measurement

**Post-processing / per-image**
- Energy restoration / band boost / sharpening / unsharp / grain / texture injection — the
  3-point parabola through r27/r32/r33 puts the optimum at 1.13–1.27x the shipped amplitude with
  **≤ +0.006 left on the entire axis**. Adding high-frequency energy trades PSNR for LPIPS and is
  **net negative** at these weights (r34: +0.015 dB PSNR bought +0.021pp LPIPS = −0.0052).
- JPEG quality/chroma (q99 vs q100 ≤0.001 on n=290; q98/4:4:4 lost on the LB; lossless +0.0196).
- Per-image exposure / white balance — oracle too small (+0.109 dB chair, +0.192 dB bonsai).
- Temporal exposure interpolation on the video scenes — explains only 14–17% of the residual.
- Chroma anything — LPIPS is **87.5% luma, 5.7% chroma**; a *perfect* chroma oracle is +0.4575
  and every real exploit measured 0/5 negative.
- Field amplitude / kernel / estimator / downsampling / sigma — all swept and closed.
- Residual registration, 3D-lift field oracle, band-limited warp, flat-region attenuation.

**Ensemble**
- Depth k=8→10 LOST (−0.0030); diagnosed as member QUALITY, not depth.
- Diversity at fixed k is at noise (t=0.62).

**Bonsai reconstruction** (the scene with the most apparent headroom)
- 8M cap: scores **−0.7101** vs 5M on 28 held-out holes. Closed on a score.
- 60k iters: A/B'd on its own eval holes — both 60k arms lost (71.08, 70.72 vs 71.36 at 30k).
- Depth priors: scored **−0.78 to −1.48**. Closed.
- Per-frame sharpness loss weighting: arithmetically a no-op.
- Lens field on bonsai: every variant ≤ no-field (LOVO n=40).
- Post-hoc global blur kernel (H-B bound): flat.

---

## 8. Where the headroom is — and four corrections to earlier beliefs

Per-scene levels on the private test scale (two independent derivations agree to 0.124):
**5 towers ≈ 78.6, chair ≈ 78.8–80.2, bonsai ≈ 70.0–72.0.** Six of seven scenes are within ~0.1
of what this method's 7–8 member ensemble delivers. **Bonsai is the only scene with real
headroom** — roughly +0.94 final if fully repaired.

But be careful, because these plausible-sounding diagnoses of bonsai are **wrong**:

1. ~~"bonsai is SfM-starved (54k points vs the towers' 154–219k)"~~ — **FALSE.** Per-*view*
   density is the **highest in private_set2**: 32,354 points in view, NN spacing 3–10 px, vs
   towers 17–19 px. Track length 7.73 vs 5.68–6.25, reproj 0.912 px vs 1.17.
2. ~~"the render/GT power spectrum shows bonsai is under-resolved"~~ — the ratio is **dominated
   by member recipe** (UT 0.72–0.75 vs non-UT 0.51–0.58) and is **uncorrelated with score**
   (r = −0.195). It is not a per-scene headroom detector.
3. ~~"8M collapsed into dead splats"~~ — 8M actually has **more** live gaussians than 5M
   (2.123M vs 1.795M). It still scores worse. The capacity axis is closed on the score, not on
   the opacity histogram.
4. Render error **ANTI-correlates with SfM density** (r = −0.775) and tracks **GT texture**
   (r = +0.778). Denser init aims at pixels that are already correct.

What IS solid about bonsai: its renders are blurrier than **100%** of its own training photos
(render median VoL 46.4 vs GT 125.1; render p95 below GT p25). The deficit is uniform across
frames, not caused by the blurry ones.

**Where LPIPS actually lives** (5 towers, n=100, bands sum to 102% so the decomposition is sound):
60% of it is below 0.8 px, 86% below 1.6 px, and everything coarser than 3 px is ≤5.8%.
Per-VGG-layer it is nearly flat, so scale — not layer — is the discriminating coordinate. This is
why every coarse operator has failed and why the sub-pixel lens field was the one big win.

**The one lead with positive evidence:** members are individually harmful, and the ballast is
predictable without GT. On the public pool, removing `m31b_nolpips` gains **+0.0372** (t=7.34),
and **corr(LOO_delta, deviation-from-pool-mean) = −0.901** — the harmful member is the one
*closest* to the mean, not the farthest. That predictor needs no ground truth, so it can be run
on the private members directly. Pool-dependent, so price it accordingly.

Also open/unfixed: the public harness pool contains a **byte-identical duplicate member**
(de-duplicating is +0.0171 there, and every published k-curve number is contaminated), and
`energy_restore` is invoked with `--k 8` while given 7 member dirs when HCM0421's true member
count is 9 — the r-map is mis-scaled 2–4%.

---

## 9. Environment

- Conda: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastgs2`
  (scoring/post-processing) or `conda activate gsplat` (training).
- Repo `/mnt/c/Users/BKAI/an_plaza2/FastGS`; scratch `/home/bkai/.claude/jobs/1c9cf7e9/tmp`.
- Lab notebook `/mnt/d/avv/EXPERIMENTS.md` (~4450 lines). **Grep it before proposing anything.**
- Submissions and provenance sidecars: `/mnt/d/avv/submissions/`.
- GPUs: 2 x 16 GB. Measured costs: bonsai 30k/5M ≈ 72 min; 30k/8M ≈ 5.2 h (superlinear — 12M
  likely will not fit).
- **Disk is the binding constraint: ~19 GB free on /mnt/d.** An 8M checkpoint is 1.8 GB. Delete
  after rendering. `chair_eval` (14 G) and `bonsai_eval` (8.2 G) are retired but not yet cleared.
- Long jobs must be launched `setsid nohup ... &` — plain background jobs get reaped.
- `bonsai_eval/K2_clip_full` is a **collapsed run** (PSNR 12.346) sitting in the arm pool. Exclude
  it or it poisons any ensemble/correlation computed over those arms.

## 10. House rules

- **Never auto-submit.** Build the zip, verify it, report — the user submits.
- Give upfront time estimates for anything slow.
- Verify a claim before shipping it: four of the last five submissions lost points because a
  plausible-sounding candidate was shipped without an adversarial check.
- Reproduce the shipped chain **byte-exactly** before changing one variable in it. This is
  achievable: r31/r32 towers = r29 `png_er` → `apply_field(fields_median_g1_g130)` → JPEG
  (q99 HCM0421 / q100 rest), and that reproduces the shipped bytes 8/8.
