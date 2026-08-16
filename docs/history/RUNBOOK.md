# RUNBOOK — NVS competition pipeline

Everything needed to take a **new dataset** to a **verified submission**, plus the traps
that cost us real points. Written 2026-07-15, ahead of the 16/07 dataset change: all
trained models, fitted fields and masks are scene-specific and die with the data. This
file and `scripts/` are what survive.

Decision history and every negative result: `/mnt/d/avv/EXPERIMENTS.md`.

---

## 0. The metric you are optimizing

```
Score = 100 * [ 0.4*(1 - LPIPS_vgg) + 0.3*SSIM + 0.3*PSNR/50 ]
```
- LPIPS backbone is **vgg** (pip `lpips`), `psnr_max = 50` — both calibrated from official
  leaderboard breakdowns, not guessed.
- Score is the mean over scenes; each scene is the mean over its images.
- **PSNR is averaged per-image in dB**, so the mean is dragged by the WORST views.
- Exchange rates: `Δscore(pts) = 0.6*ΔPSNR(dB) + 30*ΔSSIM − 40*ΔLPIPS`.
- Local scorer: `score_submission.py` reads `--gt_root/<scene>/test/images`. Local runs
  ~0.6 pts BELOW the leaderboard — rank by local, don't predict absolute.

---

## 1. One command

```bash
scripts/run_dataset.sh \
    --data_root ~/data/phaseN/private \
    --out      /mnt/d/avv/phaseN \
    --zip      /mnt/d/avv/submissions/phaseN_r1.zip \
    --tier 2 --gpus 0,1
```

Tiers (GPU time vs quality):
| tier | members | ~time/scene | notes |
|---|---|---|---|
| 1 | 1 UT model | ~3h | fastest usable |
| 2 | 2 UT seeds | ~6h | **default** — seed jitter works in the UT family |
| 3 | 2 UT + FastGS gates | ~9h | best; gates add ~+0.5 but are a separate codebase |

Then verify (run automatically, but run it again before you upload):
```bash
python scripts/verify_zip.py --zip <zip> --data_root <data_root>
```

---

## 2. The method, in the order it must happen

1. **Train 3DGUT (UT) on the ORIGINAL DISTORTED photos.** `--ut` sets
   `rasterize_mode="classic", with_ut=True, with_eval3d=True, radial_coeffs=[[k1,0,...]]`.
   This trains on "virgin pixels" — it removes the undistort resample generation entirely.
   Recipe: `--iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000
   --lpips_from 50000`. **LPIPS belongs in the TAIL.** (Running it from step 0 made an
   8k-step refit take 8.5 hours — see exp31.)
2. **Render test poses** — native, or **warp** if k1 < 0 (§3).
3. **Render TRAIN poses** (`render_train.py`) through the SAME path.
4. **Fit the lens field** on (train renders, TRAIN photos). Never test GT.
5. **Build the FoV mask** (`fov_mask.py`) from intrinsics.
6. **Ensemble** — family-weighted, per-pixel masked.
7. **Apply the field** to the ensemble PNGs.
8. **Build + verify the zip.**

---

## 3. TRAPS — every one of these cost us points or hours

### 3.1 Negative k1 is a DEGENERATE COLMAP FIT, not a lens
All scenes share one camera (same W/H, same principal point, focal within 0.3%) — yet
COLMAP fits a different k1 per scene, and on some it lands at **k1 ≈ −0.115**, a 15×
outlier of the OPPOSITE sign. That is not a lens; it is a bad optimum.

- **UT-native forward distortion FOLDS** at negative k1 (r_u = 1.704) → phantom corner
  gaussians. Those scenes MUST render through `--ut_render warp`.
- **The field must be FIT on the same path it is APPLIED to.** Native vs warp differ by
  mean 2.4/255 with 27% of pixels off by >2 — far above the 0.2–2 px scale the field
  operates on. Fitting on native and shipping warp silently mis-aims the correction.
- `run_dataset.sh` auto-detects the sign and routes both. **Do not hand-edit k1** — the
  given test poses live in the frame COLMAP fit under that k1; changing it desynchronizes
  poses and points.

### 3.2 The lens field (biggest single lever: R6 77.66 → R7 78.40, +0.73, NO retraining)
Our renders are misregistered from the photos by a fixed, image-independent sub-pixel 2D
field (mean ~0.2–0.3 px, up to ~2 px at the edge; up to **5.7 px** on negative-k1 scenes).
Cause: COLMAP gives SIMPLE_RADIAL (one k1) and **gsplat's distortion coefficients take no
gradient** (verified: `.grad` is `None`) — every higher-order radial/tangential/thin-prism
term is pinned to zero, and the gaussians contort to absorb it.

- **Fit on TRAIN photos only.** `fit_field.py` hard-asserts the GT dir is not `/test/`.
- Use the **full 2D field**, not a radial polynomial: the real curve is non-monotone and a
  3-term odd polynomial captures only half of it (+0.42 dB of +0.79 dB).
- It transfers: fit on train recovers **95%** of an oracle test-fitted field.
- **It buys SSIM more than PSNR** (+0.39 vs +0.33 pts) — it's a structural fix.
- Sanity-check any new scene with a 2-fold CV on its own train views (`d10_privcv.py`
  pattern): fold correlation should be >0.9 and held-out gain positive. A big field is not
  automatically wrong — the negative-k1 scenes had the LARGEST fields (5.7 px) and the
  MOST reproducible ones (r=0.97).

### 3.3 FastGS gates are UNSUPERVISED in the outer ring on negative-k1 scenes
Same-K undistortion crops the FoV: **11.9% of the frame** has no training signal for the
gate members there (0.00% on normal scenes). Measured on train GT: in that ring the gates
collapse to **17.8 dB** while the UT member holds **21.1 dB**. Mask them out
(`--masks`); the UT members carry the ring alone. The mask is all-ones on normal scenes,
so it is safe to pass everywhere.

### 3.4 JPEG: exactly ONE generation
Encode ONCE from the lossless PNG archive. Re-encoding an already-encoded JPEG costs
−0.14..−0.26. `apply_field.py` refuses non-PNG input for this reason. The zip builder walks
a quality ladder (q100→q95) until it fits 350MB; progressive JPEG is pixel-identical and
saves ~5%, letting small scenes ride q100.
**Note: sharper images cost more bytes** — the field and the mask both ADD detail and push
the ladder down a step. That is a real (small) tax against the gain.

### 3.5 Never double-apply the field
Warping an already-warped dir roughly doubles the displacement — it would *increase* error
by about what it removed, silently. `apply_field.py` writes a `field_applied.json` stamp
and refuses to run on a stamped dir.

### 3.6 Ensembling is the reliable lever, and weights matter
- Pixel **mean** (not median: −0.08). Accumulate float32, round once at the end.
- Average in **sRGB** (linear-RGB averaging: −0.012).
- **Family weighting is free points**: UT and FastGS are different families (UT carries
  LPIPS, gates carry PSNR/SSIM). w_UT ≈ 0.5–0.6 beats uniform by ~+0.23. The optimum is
  broad — but **re-sweep it whenever member quality changes**, or you ship stale weights.
- Only ensemble members within ~0.15 pts of the best; a weak member dilutes.
- **Seed jitter works in the UT family** (MCMC data-order decorrelation) but NOT in FastGS
  (`safe_state` pins the seeds).

### 3.7 Ops
- Launch long jobs `setsid nohup bash job.sh > log 2>&1 < /dev/null &`, **one per call** —
  batching several in one shell silently drops some.
- Set `PYTHONUNBUFFERED=1` or a job's progress is invisible for hours (stdout is
  block-buffered at 4KB; bash `echo` markers still appear immediately).
- Killing a launcher does NOT kill its child loop — kill the process group, or a zombie
  will respawn training and hold the GPU for hours (this cost us 12 GPU-hours once).
- `/usr/bin/time` cannot exec an env-var prefix: write
  `CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f ... python ...`, not `time ... CUDA_... python`.

---

## 4. Dead ends — do NOT re-run these

| idea | verdict |
|---|---|
| Supersampling / SSAA at render | Monotonically worse (−0.88 to −6.73). Any imposed low-pass moves away from GT. |
| Per-view appearance (affine, bilateral grid) | −0.91 and −7.6. Absorbs real signal. Oracle colour correction bounds the whole family at **+0.09 dB**. |
| Depth-guided IBR (warp train photos into test poses) | −4 to −7. 11.8° parallax, 9.4 dB oracle paste. |
| Per-test-pose local finetune | >8 min/pose (64h), colour-only probe = noise. |
| Pose refinement (per-image) | Oracle upper bound only **+0.14 pts**, and it does NOT transfer: test poses carry the same noise, so a noise-blurred map is near-MMSE-optimal. |
| Test-pose cleanup / fixing worst views | No bad tail exists (per-image PSNR std 1.25). |
| SH-degree clamp at render | Monotonically harmful (sh 0/1/2/3 → 67.8/69.1/72.5/75.9). |
| MVS dense seeding | MCMC relocation erases init advantages by 30k. |

## 5. What's left worth trying

- **Metric-exact loss** (`--metric_loss`): `0.3*(1−SSIM) + 0.02606*ln(MSE)` — exactly the
  score. Untested (exp31 killed on cost). Re-run it CHEAPLY: no LPIPS, or LPIPS in the tail
  only. Log **train-view PSNR** alongside test for both arms.
- **Refit k2 and RETRAIN.** The field is post-hoc; it does not stop the gaussians contorting
  to absorb the bad lens during training. Fixing the camera model at the source could bite
  into the GEOM residual (the largest remaining, ~+0.74 pts by oracle).
- **Train-view PSNR as standing telemetry** — the single most informative free number.
