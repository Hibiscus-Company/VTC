# README2 — Consolidated Working Memory

> Distilled 2026-08-15 from the 24 persistent cross-session memory notes behind this project.
> Companions: `README.md` (upstream FastGS), `archive/INDEX.md` (full working record, 2,889 files),
> `REPRODUCE_r36.md` (rebuild of the graded submission).

---

## 1. Project & current status

Fork of **FastGS** (CVPR 2026) carrying Hibiscus's **VAR 2026 "BTS Digital Twin"** NVS competition
work. The production method is **not** the upstream FastGS trainer: it is the `gsplat_track/`
pipeline (gsplat MCMC + 3DGUT distorted-space rendering), with FastGS as the fork base.

- **Rounds 1–2 (finished):** final graded submission **r36 = 77.7230** on private_set2
  (PSNR 26.655 / SSIM 87.347 / LPIPS 11.186). LB top-1 finished 82.17. **r37** (forecast +0.11)
  was built + verified but never submitted. Campaign paused 2026-08-01; full record archived into
  `archive/` on 2026-08-13. Post-verification package **delivered** to the organiser (Drive);
  local copies intentionally cleaned; only `~/r36_core.zip` (351 MiB) survives locally.
- **Round 3 (current):** organiser announced 2026-08-14 — expansion to **city-scale Large-Scene
  NVS from drone imagery**, emphasis on large scenes, many viewpoints, **spatial consistency**.
  No dataset/metric/dates yet. We are in the research-ahead prep phase; a refereed idea slate and
  day-1 data-drop protocol exist (§7).

## 2. Competition setup & metric facts

- **Score** = `100·[0.4·(1−LPIPS_vgg) + 0.3·SSIM + 0.3·clamp(PSNR/50,0,1)]`, mean over scenes.
  Leverage per scene: −0.01 LPIPS = +0.4, +0.01 SSIM = +0.3, +1 dB PSNR = +0.6 (÷N scenes for the
  final). **LPIPS dominates; PSNR is near-saturated** (psnr_max = 50.0, solved exactly from R1).
- **Submission**: JPEG images (only frames named in each scene's `test/test_poses.csv`) in a zip.
  **Cap = 350 MiB = 367,001,600 bytes** (empirically confirmed: a 357-MB-decimal zip was accepted).
  Ship `quality=100, subsampling=2`, single-generation encode from PNG archives, never re-encode
  an existing JPEG. Build: `build_submission_zip.py`; gate: `scripts/verify_zip.py`.
- **Pose convention verified**: `test_poses.csv` rows are byte-identical to COLMAP `images.bin`
  world-to-camera entries. Organiser poses measured correct to **0.018 px** — trust their geometry;
  never re-solve test poses.
- Round-2 data: 5 towers (SIMPLE_RADIAL, k1≈+0.009, 240/60) + 2 indoor videos (`bonsai` 248/28,
  `chair` 205/58, PINHOLE). Test frames are interleaved holdouts of the capture.

## 3. What won (proven levers, in order of measured impact)

1. **Render ensembling** — pixel-mean of independent models' PNG renders. +1.82 on the LB in one
   step (amplified 3.6× from local). Rules: float32 accumulate, round once; sRGB; members within
   ~0.15 of the best single; **config-jitter decorrelates, same-config reruns do not** (seeded
   RNG); member QUALITY beats diversity (diversity at fixed k measured at noise, t=0.62); never
   add stale-generation members (r30's k=8→10 loss).
2. **3DGUT** (`train_gsplat.py --ut`) — train on original distorted images ("virgin pixels"),
   render native distorted. ~+0.9 LB; the gain was earned on cameras with **small** k1, so the
   default is UT-on unless pure pinhole. Video scenes (k1=0): drop `--ut`, use antialiased
   (+0.49/scene). Negative-k scenes render via `--ut_render warp`.
3. **Lens-field correction** — renders are misregistered from photos by a fixed sub-pixel field.
   Fit on TRAIN photos (`render_train.py → fit_field.py → apply_field.py`), ship at **gain 1.30**
   (train fit undershoots ~30% — the model absorbs part of the bias at train poses; LB-proven
   +0.7345 then +0.1615). Per-image operator class → transfers ~1×.
4. **MCMC schedule/capacity tuning** — churn window (refine/noise stop) and training length are
   the strongest recipe dials: bonsai collapse fix +3.6 (cause was churn on glossy glass, NOT
   capacity), long45k +0.71, cap 5M→8M +0.36 on rich towers but **−1.7 on sparse-coverage bonsai**
   (sign flips by scene → tune per scene).
5. **Eval-split recipe selection** (LB-validated): hold out **isolated, evenly-spaced** frames
   (never contiguous arcs — those overshoot test difficulty; isolated-every-k matched real test
   GT: gap 0.121 vs 0.116). Name-free router; recipe nudges transfer ~0.3×, need ≥+0.5 scene-pts
   on eval to earn a submission slot.

## 4. Calibrations & transfer rules (the measurement machinery)

- **Transfer taxonomy** (learned by losing rounds): per-image operators ~1×; model-recipe changes
  ~0.85× (the diversity-matched **encoded k=2 gate** predicts LB scene gain at 0.84×); ensemble
  mixsweeps inflate ~2.2×; **pool-dependent operators ~1/20** — the public harness pool is 2× as
  diverse as ours (4.67 vs 2.20/255), so anything whose value depends on ensemble over-smoothing
  over-reads there. Cross-validate on ≥3 public towers before shipping.
- **Production harness** (`public_set` scenes have real test GT): submetric-predictive of the LB
  (within 15%). It overturns train-side conclusions — any protocol holding out train views cannot
  see a defect the model absorbed at train poses (this is how the 1.30 field gain hid for weeks).
  Score through the full shipped chain (operator → field → JPEG), never raw PNG.
- **uint8 deadband**: sub-LSB corrections die to intermediate rounding (49–79% of energy-restore's
  correction was destroyed). And fixing delivery is harmful where the operator's sign is negative
  (energy restore is monotonically harmful on bonsai). Measure delivered amplitude in LSB; sweep
  operator sign per scene family.
- Local scorer: `score_submission.py` (pip lpips vgg, psnr_max=50), bias local−LB ≈ −0.6.
  `scripts/eval_score.py` drifted after 17/07 (−0.95 on identical bytes) — pre/post numbers are
  incomparable.

## 5. Kill list (measured dead — do not retry)

- **JPEG q98/4:4:4** — shipped and LOST (−0.0049); its harness gain was ensemble-heterogeneity
  artifact substitution we don't have. Keep q100/ss2.
- **Depth priors as training loss** — lost to churn/length tuning even in sparse-coverage
  conditions; a pure dial between view populations with global geometry damage. (Caveat: that
  verdict is for monocular priors on multi-view-rich outdoor scenes.)
- **Appearance modeling on photometrically stable rigs** — killed 4 ways (bilagrid −7.6, affine
  −0.9, PPISP marginal); fit-to-GT oracle caps the whole class at **+0.09 dB**.
- **Pose refinement** — in-training dead 3×; test-time BA oracle +0.23 dB, measured −0.02
  (organiser poses are near-perfect).
- **Sharpening / deconvolution / radial frequency re-weighting** — dead by proof (render HF
  amplitude already 0.98–1.00 of GT); the 0.4-weighted LPIPS term vetoes global frequency moves.
- **SSAA / super-resolution at test time** — monotone negative (−0.9…−6.7).
- **IBR photo reuse** — −4 to −7, stochastic depth-noise misregistration.
- **Per-tile member selection** — honest fit −0.69 dB below the plain mean; winner agreement at
  chance. **Sky dome** — neutral on towers. **eps2d sharpening on video** — GT is genuinely blurry.
- **The "PSNR ceiling impossibility proof" is RETRACTED** — stop citing it (the gap was real but
  no longer provable; top-1 submetrics were never visible).
- **bonsai confounds**: both the capacity and blur diagnoses were wrong; the measured driver is
  **sparse view coverage in the first third of the capture** (score ~ log d_mean5, R²=0.68).
  Lesson: partial every per-frame correlation against frame index before believing it.

## 6. Post-mortem — where the gap lived (168 arms mined)

93 measured-negative / 43 positive / 6 ops-failures / 26 abandoned. The decisive numbers:

- Our LPIPS deficit was **86% sub-pixel** (<1.6 px) — why every coarse operator failed and the
  sub-pixel lens field was the one big operator win.
- A **cheating dense-flow registration oracle caps the entire render-correction class at 78.79**;
  top-1 finished 82.17. ~3.4 pts lay outside every operator: **the gap is base reconstruction
  quality, spread across all scenes** (perfect video repair buys only 1.49 of 4.94).
- **The frontier died operationally while the interior died scientifically**: the arms that could
  have raised base reconstruction (cap16M, 2DGS, UBS-6D, Difix-FT, 120k) all ended as ops
  failures, never measurements. Ops tax ≈ 100 GPU-h (lower bound) on a 2-GPU campaign.
- The two largest LB jumps ever were structural moves made early (ensembling, 3DGUT); the last
  four graded rounds bought +0.032 total.
- Still-open per the record: MVS/dense seeding (GEOM band +0.74), coverage-weighted loss
  (pre-registered, never run), capacity >8M (never measured), pose-keyed per-frame selection
  (+0.12 ungated), field refit on ensemble mean (+0.03–0.05).

## 7. Round 3 — strategy, slate, day-1 protocol

**Strategy verdict** (6-lane web sweep, 2026-08-14; briefing artifact "City-Scale Flight Plan"):
**stay on gsplat** — the only known-good sm_120 build, now shipping Grendel-style distributed
training and 3DGUT — and extend `train_gsplat.py` with VastGaussian-style independent cells
(10–12 GB/cell is the only sub-16 GB datapoint in the field) plus gated appearance machinery.
Fallback: CityGaussianV2 (torch≥2.7 bump + 3 CUDA-fork recompiles). Probes: CityGS-X, Momentum-GS.
Capacity valve: CLM (GS-Scale is Intel-only; our CPU is AMD). Every rival repo pins torch≤2.3
(pre-Blackwell). Benchmarks that fit disk: Mill-19 (~21 GB) + MatrixCity small-city aerial
(31.3 GB); Rubble bar ≈ 26.9–27.9 PSNR at the 1600-px protocol. No published city-scale pipeline
uses 3DGUT — still our differentiator.

**Refereed idea slate** (3-skeptic adversarial pass, 2026-08-15):

- *Build now:* ① cell-partition platform + 2-GPU orchestration + eval harness (isolated-every-k
  holdout; seam metric in merge acceptance tests) ② **shared-trunk fork ensemble** as default
  production shape (fork last ~10–25% with config-jitter; SoccerNet-2026 winner recipe) ③ harness
  arsenal port — code day-0, but every carried constant (0.84×, metric, cap) is a **prior to
  recalibrate** on first round-3 probes ④ capacity/churn tuning **per cell-cluster**, not per
  cell (per-cell sweeps = 100+ GPU-nights). VRAM reality: Windows eats ~5 GB of GPU0 → effectively
  an 11 GB + 16 GB pair.
- *Day-1 data-drop protocol* (scripted before drop): rig forensics is the decision node — camera
  model/k1 (→ 3DGUT default ON unless pure pinhole), exposure-vs-capture-time (→ appearance gate,
  super-LSB threshold, nearest-pose fallback), sky fraction in test frusta (→ sky policy only if
  >10–15%), pose reprojection sanity check (→ trust organiser geometry; GLOMAP only as
  contingency), test-CSV structure (→ holdout design). Lens-field refit runs unconditionally in
  production round 1 (re-verify the ×1.30 gain on the new rig).
- *Killed by the panel:* selective depth regularization (refuted ×2 → replaced by post-hoc
  view-consistency floater culling); coverage-weighted loss (→ per-image view-class up-weighting,
  gated); HF energy restoration on a low-disagreement fork pool (near-refuted; 3 strict rules if
  ever used); sky + appearance machinery → threshold-gated contingencies.

## 8. Environment & hardware

- **Machine**: WSL2, 2× RTX 5070 Ti (Blackwell **sm_120**, 16 GB, ~5 GB of GPU0 taken by Windows),
  AMD Threadripper 9960X, 47 GB RAM visible to WSL (raisable via `.wslconfig`), driver CUDA 12.9.
- **Env**: conda `fastgs2` (python 3.10, torch 2.7.1+cu128, cuda-toolkit 12.8.1). The stock
  `fastgs` env is unusable on this GPU. CUDA extensions built with
  `CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="12.0+PTX" pip install --no-build-isolation`.
  gcc13 needs `#include <cstdint>` in the FastGS rasterizer header. gsplat needs the CUDA-12.8
  patch kept in `archive/env/`.
- **FastGS crash fixes** (all local, sanitizer-clean): negative tile-count in `processTiles`
  (the root cause), fused-ssim shared-memory overflow, SnugBox write-budget hardening, backward
  prefetch guards. Debug recipe: `--debug` dump → replay → compute-sanitizer.
- FastGS `--mult` must match between train and render (cfg_args doesn't carry it).
- **Disk**: D: and C: are at 100%; ext4 `/` (629 GB free) is the only viable home for new data.
  35 checkpoints (~19 GB) + submission zips (6.5 GB) still sit on the full D: — protective copy
  to ext4 proposed, awaiting go.

## 9. Operational rules (paid for in GPU-hours)

- **Detach every long job**: `setsid nohup bash queue.sh > log 2>&1 < /dev/null & disown`, verify
  its SID. A process exit once killed a 10.5-h run with no checkpoint (~12.5 GPU-h lost).
- **Never delete checkpoints behind a submission** — `rm -f ckpt.pt` destroyed 50/62 weights
  before the organiser asked for them. Archive to `WEIGHTS/` first. (Render PNG archives are the
  load-bearing reproduction artifact — bonsai rebuilt 28/28 byte-identical from them.)
- Never `pgrep`/`pkill` a pattern that matches your own launcher (self-kill, happened twice).
  Never co-schedule test jobs with production trainers (19.3 GPU-h contention loss). Never edit a
  running bash script. Separate `local` statements in bash (`local A=$1 B=$A` silently empties).
  No `set -u` in env-switching scripts. Set a waiter on every queue.
- Byte-verify render provenance before rebuilding anything shipped.

## 10. Working-practice directives (user)

- **ETAs upfront** for any long task — duration + wall-clock, derived from measured past runs.
- **Family-branch ensembling**: a candidate within ~0.15 of baseline that is method-decorrelated
  ADDs as an ensemble member (judge on the 2-member ensemble score, not the solo score).
- **Zip + notify every submission round; never auto-submit.** Bundle confirmed gains into one
  combined zip per round (single-slot swaps are unmeasurable on the LB). PushNotification when a
  verified zip is ready.
- Periodic read-only **sub-agent audits**: (1) code-error audit, (2) fresh-agent game-changer
  audit. CONFIRMED vs HYPOTHESIS labels, file:line, most-severe first.
- Keep both GPUs at maximum utilization (claim-based job pool, not rigid lanes).

## 11. Side study — surgical 3DGS (template, not the contest)

`gs-surgery-formula-guide.md.pdf` (repo root) is an 11-page formula tour of endoscopic 3DGS,
studied as a *template* for problem→mechanism thinking. Measured while grounding it:
`/mnt/d/avv/data/SCARED-E2E` = 4 sequences, ~1,810 rectified 1280×1024 stereo pairs, 26%
disparity holes, 4.14 mm baseline, no distortion; `/mnt/d/avv/gsplat_src` contains NVIDIA's
G-SHARP surgical scaffold with a SCARED loader stub. Flagship idea if ever pursued:
**pose-conditioned illumination** (light rigidly attached to the endoscope → appearance is a
function of pose, generalizes to novel views by construction). Three CPU-cheap probes
pre-registered (staticness, illumination variance, depth-alignment).

## 12. Pointers

| What | Where |
|---|---|
| Full experiment ledger (397 KB) | `archive/avv/EXPERIMENTS.md` |
| Idea ledger (85 closed / 25 untried) | `archive/avv/IDEA_LEDGER.md` |
| Production builds as-run | `archive/jobs/build_r22.sh … build_r37.sh` |
| Graded-submission rebuild | `REPRODUCE_r36.md` (§7 is fresh-build, not as-run) |
| Surviving weights manifest | `archive/env/WEIGHTS_MANIFEST.md` |
| Round-3 briefing artifact | claude.ai/code/artifact/c65b74d5-4c23-40db-812b-c9cf5dbbde77 |
| Off-repo data/renders | `/mnt/d/avv/` (submissions, r38/r45 member PNGs, phase1 data) |
| Memory source notes (24) | `~/.claude/projects/-mnt-c-Users-BKAI-an-plaza2-FastGS/memory/` |
