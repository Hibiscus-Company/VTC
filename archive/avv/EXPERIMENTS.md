# VAR 2026 Bài 1 — Experiment Log

**Objective:** maximize `Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*clamp(PSNR/50, 0, 1)` (×100), mean over scenes. Beat leaderboard top 76.75.

**Local scorer:** `FastGS/score_submission.py` — LPIPS backbone **vgg** (pip `lpips`), **psnr_max = 50.0** (both calibrated/derived from official leaderboard breakdowns: official LPIPS 0.1411→0.1331 matches vgg family, not alex ~0.10; psnr_max=50 solved exactly from round-1 breakdown). Known constant bias local−leaderboard ≈ **−0.6 pt** (public g2: local 73.88 vs LB ~74.5) — ranking by local vgg score is trustworthy; absolute LB prediction ≈ local + 0.6.

**GT for local scoring:** `~/data/phase1/public_set/{hcm0031,hcm0034,HCM0181,HCM0193,HCM0204}/test/images` (fast ext4 copy; slow master at `/mnt/d/avv/data/phase1`).

**Baseline config (exp00 = round-1 g2, LB private 74.769 / public ~74.5):** FastGS 30k iters, train on `images_undist` (undistorted), render with `--distort auto` (warp back to SIMPLE_RADIAL), `densification_interval 100`, `highfeature_lr 0.04`, `grad_abs_thresh 0.0002`, `mult 0.7`, `--data_device cpu`. Models in `FastGS/output/<scene>_g2`.

**E: is READ-ONLY from WSL** → durable store is `/mnt/d/avv/submissions/` (D:), this log at `/mnt/d/avv/EXPERIMENTS.md`.

## Prior negative results (before this log; don't retry blindly)

| change | result (Δ local score, old vendored-vgg scorer) | note |
|---|---|---|
| test-time supersample 2× + **box** downsample | **−0.02** | re-testing per plan with Lanczos + more factors, new scorer |
| edge-aware TV loss λ=0.1 | −0.001 | over-smooths |
| densify_until_iter 15k→20k (naive) | −0.025, gaussians DROP 2.3M→1.1M | FastGS prune/opacity-reset schedule tuned around 15k; naive extension backfires. >15k also collided with post-15k prune (crash, fixed by schedule-untie in train.py) |
| opacity-sparsity entropy λ=0.01 / 0.0005 | −0.048 / −0.034 | FastGS already prunes opacity<0.005 every 100 it |
| grad_abs_thresh 0.0004→**0.0002 (g2)** | **+0.004 ACCEPTED** | capacity ↑ = the one lever that worked |
| grad_abs_thresh 0.0001 (g1) | +0.0015 on dense scenes, regresses sparse | risky, not adopted |

## Runs

| exp | change (vs exp00) | scenes | PSNR | SSIM | LPIPS(vgg) | Score(vgg)×100 | Δ vs exp00 | wall-clock | verdict |
|---|---|---|---|---|---|---|---|---|---|
| exp00 | baseline g2 (round-1 submission) | 5 public | 24.8118 | 0.8388 | 0.1543 | **73.8802** | — | (10 min/scene train) | BASELINE. LB public ~74.5 / private 74.769. zip: `pub_exp00_baseline_g2.zip` |
| exp01a | re-render lossless PNG (vs JPEG q100 in zip) | HCM0181 | 24.30 | .8469 | .1499 | 73.99 | +0.01 | 2 min | neutral — compression format is a non-lever |
| exp01b | SSAA 1.5× Lanczos | HCM0181 | 23.46 | .8325 | .1486 | 73.11 | **−0.88** | 3 min | REJECT |
| exp01c | SSAA 2× Lanczos | HCM0181 | 22.45 | .8029 | .1666 | 70.89 | **−3.10** | 4 min | REJECT |
| exp01d | SSAA 2× box | HCM0181 | 22.83 | .8142 | .1629 | 71.61 | **−2.38** | 4 min | REJECT |
| exp01e | SSAA 3× Lanczos | HCM0181 | 21.26 | .7614 | .2086 | 67.26 | **−6.73** | 5 min | REJECT |

| exp02 | schedule scale ×4/3 (40k iters, densify→20k, LR→40k, prune window scaled) | HCM0181 | 23.92 | .8265 | .1804 | 71.93 | **−2.06** | 15 min | REJECT — gaussians 2.33M→1.24M: longer densify keeps opacity-resets to 18k + 6 (vs 4) aggressive final prunes → capacity HALVED. Densify-longer branch closed (2nd independent failure) |
| exp03 | grad_abs_thresh 0.0002→0.00015 | HCM0181 | 24.30 | .8458 | .1445 | **74.17** | **+0.18** | 13 min | PROMISING — gaussians 2.86M (+23%), LPIPS −.0054, SSIM/PSNR flat. Confirming on full set |
| exp03-full | grad_abs 0.00015 (**"g15"**) | 5 public | 24.8396 | .8426 | .1542 | **74.0160** | **+0.136** | ~65 min | **ACCEPTED — new base config.** ALL 5 scenes positive (+0.06..+0.19), incl. sparse hcm0031 — no g1-style regression. All 3 metrics up. Models: `output/<scene>_e03ga15` |
| exp04 | + VGG-LPIPS loss λ=0.1 from iter 25k (`--lambda_lpips`) | HCM0181 | 24.23 | .8431 | .1266 | **74.77** | **+0.78** | ~22 min | **BIG WIN.** LPIPS −.023 for −.004 SSIM / −.07dB. alex-LPIPS also −.007 → genuine perceptual gain, not backbone gaming. Official metric confirmed vgg-family → directly aligned |
| exp06 | STACK: g15 + lpips-ft λ0.1@25k | 5 public | 24.7936 | .8406 | .1368 | **74.6229** | **+0.74** | ~100 min | **ACCEPTED — NEW BEST.** All 5 scenes positive (+0.6..+0.9 pts each); LPIPS −.0175, SSIM +.002, PSNR −.02 (noise). Stack g15(+0.14)+lpips(+0.6) ≈ additive. LB projection ≈75.2. zip: `pub_exp06_g15_lpips01.zip`. Models `output/<scene>_e06stack`. NOTE: scenes 4-5 trained with lpips-clamp parity fix, 1-3 without (negligible) |

## Code audit (subagent, 2026-07-11) — pipeline verified CLEAN + 2 new levers

**CONFIRMED clean (numeric):** principal point exactly W/2,H/2 all 5 scenes, 0px render↔GT offset (ndc2Pix convention closes the loop); DistortionWarp round-trip 1.4e-9px, corner shift 5.05px as expected; undistort→train intrinsics consistent to 0.006px; no silent resize at -r -1; repo SSIM ≈ Wang reference (+0.003 pt); mult 0.7 truncates footprint at α≈0.02 but consistently in train+densify+render (A/B was neutral). Rasterizer has stock `cov += 0.3f` dilation, NO AA compensation, `pipe.antialiasing` dead → matches over-smoothing diagnosis, principled fix = MIP_PLAN.md.

**Fixes applied:** `gaussian_model.py:512` padded_importance now on cuda (perf, was CPU multinomial + sync every 100 it); lpips-ft input now clamped [0,1] (parity with scored artifact).

**New tunable gates (audit hypotheses H1/H2, defaults = stock):** `--metric_gate` (was hardcoded `importance_score > 5` — a 5px cable gaussian maxes ~5 hits/view → floor→ permanently BLOCKED from densifying; large blurry gaussians pass trivially); `--final_prune_thresh` (was hardcoded 0.9 — min-max-normalized score ALWAYS prunes top slice post-15k with no re-densification; on this data the persistent-high-error regions ARE the cables).

| exp | change (on g15+lpips base) | scenes | PSNR | SSIM | LPIPS(vgg) | Score(vgg)×100 | Δ | wall | verdict |
|---|---|---|---|---|---|---|---|---|---|
| exp08 | metric_gate 5→2 | HCM0181 | 24.3150 | .8469 | .1224 | **75.1010** | **+0.23** (vs e06stack .7487) | ~25 min | **WIN — audit H1 confirmed.** Gaussians 2.86M→4.10M (+43%), ALL 3 metrics up. Cumulative on HCM0181: 73.99→75.10 (+1.11). → full-set validation |
| exp09 | final_prune_thresh 0.9→0.98 | HCM0181 | 24.2672 | .8443 | .1251 | 74.8855 | +0.02 (vs e06stack) | ~25 min | NEUTRAL — audit H2 rejected. Gaussians unchanged 2.86M; keep stock 0.9 |
| exp10 | metric_gate 1 | HCM0181 | 24.3347 | .8475 | .1220 | **75.1459** | +0.28 (vs e06stack) | ~28 min | best gate. Gaussians 4.84M. Dose-response monotone: gate 5→3→2→1 = 74.87→75.04→75.10→75.15 |
| exp11 | metric_gate 3 | HCM0181 | 24.3033 | .8461 | .1231 | 75.0410 | +0.17 | ~26 min | between, as expected |
| exp12 | CHAMPION full-set: g15 + λ0.1@25k + gate1 | 5 public | 24.8173 | .8429 | .1345 | **74.7969** | **+0.17** (vs e06) / **+0.92** (vs exp00) | ~2h | **ACCEPTED — NEW BEST.** 4 scenes up (+0.16..+0.33), hcm0031 flat (−0.04 noise; sparsest scene). zip `pub_exp12_champ_gate1.zip`. Submission uses best-per-scene: hcm0031←e06stack, rest←champ |

**Cumulative session progress (local vgg mean): 73.88 → 74.80 (+0.92). LB projection ≈ 75.4 (bias +0.6). Top = 76.75, gap ≈ 1.35.**

**SPEC UPDATE (user, 2026-07-12): submission = private_set1 ONLY (8 scenes), only CSV-listed images, .JPG, ≤350MB.** The 13-scene zip below is superseded (public scenes = local validation only).

**JPEG-encoding A/B on public proxy (q100 baseline 74.7969):** q95ss0 −0.255 (NOT negligible — user was right to ask) | q96ss0+opt −0.140 | **q98ss2+opt −0.052 ← chosen** (chroma subsampling ≈ free on this imagery; quantization level is what LPIPS-vgg notices) | q99ss2 doesn't fit (404MB).
→ **FINAL ARTIFACT: `sub_round1_champ_private_q98ss2.zip` — 343.9MB, 434 files** (q98ss2, three largest scenes auto-degraded to q97ss2 to fit; expected cost vs lossless ≈ −0.05..−0.08). `sub_round1_champ_private_q95.zip` (323.8MB) kept as backup only.

**SUBMISSION SHIPPED (2026-07-12 05:12): `/mnt/d/avv/submissions/sub_round1_champ.zip`** — 13 scenes / 724 files / 1.22GB, counts+names+sizes CSV-validated. Public-5 = best-per-scene (HCM0181←e10gate1, hcm0031←e06stack, rest←e12champ); private-8 = champion config (g15+λ0.1@25k+gate1), models `output/<scene>_champ`. Expected ≈75.4 LB (public proxy 74.80 local +0.6 bias).

## Audit round 2 (2026-07-12) — key outputs

- **MIP_PLAN.md reviewed**: implement (A) 2D opacity-compensation ONLY (~90-140 LoC); 5 corrections appended to the plan (cov.y off-by-2×, comp≥0.005 guard, fork-specific t≤0 tile guard against the old crash family, dL_do must scale by comp, h=0.1 variant). **(B) 3D filter REJECTED** — 2 math errors in spec + wrong layer + net-negative expected on interpolated poses.
- **H3 traced**: opacity ceiling 0.8 clamped every 100 it during densify → thin structures ≥20% transparent during layout formation → stacked/fat splats. Now `--opacity_ceiling`.
- **Gate semantics**: gate g = ≥10·(g+1) total hits; `--metric_gate -1` disables gate entirely, 0 lines. Curve 5→1 monotone & unsaturated → test 0 and −1.
- **Importance denominator bias**: fixed ÷10 under-counts peripheral gaussians by (visible/10) — ~5 LoC fix + gate re-sweep, queued behind cheap levers.
- appearance_affine code verified correct; added per-channel mean-recentering (global exposure must stay in the model).
- LPIPS window @20k: already tested (exp07c, tie) — dropped from shortlist.

| exp | change (on champion) | scenes | PSNR | SSIM | LPIPS(vgg) | Score(vgg)×100 | Δ | wall | verdict |
|---|---|---|---|---|---|---|---|---|---|
| exp13 | metric_gate 0 | HCM0181 | 24.3155 | .8478 | .1219 | 75.1458 | ±0.00 vs gate1 | ~30 min | gate curve **SATURATED at gate1** (5.93M vs 4.84M gaussians, identical score). Keep gate1. e13 = 4th ensemble member |
| exp14 | metric_gate −1 (off) | HCM0181 | | | | | | 7.7h+ (!!) | **PATHOLOGICAL — REJECT.** Gate removal → unbounded densification: VRAM pinned 15.4/16GB, 15× normal wall clock, still unfinished. Zero info value (gate curve saturated at gate1). Kill blocked by permission classifier — awaiting user `kill -TERM -- -1951`; queue14.sh staged to resume e15→e16→e17→memB |
| exp15 | opacity_ceiling 0.95 | HCM0181 | 24.3058 | .8470 | .1226 | 75.0930 | −0.05 vs gate1 | ~40 min | NEUTRAL (noise) — audit H3 ceiling lever dead, keep stock 0.8. Within 0.15-band → eligible spare ensemble member (config jitter) |
| exp16 | appearance_affine λ_app 0.1 | HCM0181 | 23.4321 | .8377 | .1238 | 74.2404 | **−0.91** | ~40 min | **REJECT — PSNR −0.90dB.** Affine absorbed real signal, not just exposure (recentering insufficient at λ_app 0.1). Lever closed; not worth a λ sweep at current deadline economics |
| exp17 | importance_vis_norm (per-gaussian visible-view denominator, audit item 5) | HCM0181 | 24.3164 | .8474 | .1220 | 75.1318 | −0.01 vs gate1 | ~40 min | NEUTRAL-TIE (8M cap never hit). Lever dead on HCM0181 as predicted (tower-centric); per round-4 caveat a null here ≠ dead on hcm0031, but not worth a slot at current deadline economics. Champion config unchanged |

| exp18 | **RENDER ENSEMBLE (test-time, zero training): pixel-mean of 3 models' renders** | 5 public | 25.1905 | .8522 | .1309 | **75.4452** | **+0.65** | ~10 min CPU+GPU0 | **HUGE WIN — every scene +0.48..+0.83.** Members: HCM0181 = gate1+gate2+gate3; others = g15+stack+champ. Unlike SSAA (pixel smoothing, failed), model-averaging cancels reconstruction noise and keeps GT frequency response: PSNR +0.37dB, SSIM +.009, LPIPS −.0036 simultaneously. Stacks with everything. zip `pub_exp18_ensemble3.zip`. **Private-8 needs 2 more models/scene (~9h GPU) to ensemble the real submission** |

Code added while GPU busy: `--importance_vis_norm` (fast_utils.py; hit-counts ÷ per-gaussian visible views instead of ÷10), `--opacity_ceiling`, appearance recentering. Mip (A) implementation deferred to a dedicated session — corrected spec ready in MIP_PLAN.md; nothing to build/test until GPU frees, and exp13-17 results may reorder priorities.
| exp07a | λ_lpips 0.2@25k | HCM0181 | 24.2343 | .8429 | .1230 | 74.9065 | +0.04 (vs e06stack) | ~25 min | tie (noise). λ curve FLAT 0.1–0.2 |
| exp07b | λ_lpips 0.05@25k | HCM0181 | 24.2523 | .8446 | .1279 | 74.7712 | −0.10 | ~25 min | worse — λ too weak |
| exp07c | λ_lpips 0.1@20k (earlier start) | HCM0181 | 24.2117 | .8436 | .1237 | 74.8895 | +0.02 | ~28 min | tie — earlier start buys nothing. **Keep λ0.1@25k** |

**exp01 conclusion:** SSAA monotonically worse with factor, all filters. At 1.5× LPIPS is flat but PSNR −0.84dB / SSIM −0.014: GT has a specific sharp/aliased frequency response at native res and the model is fit to it — ANY extra low-pass moves away from GT. Test-time resampling family CLOSED (deltas vs exp01a ss1-PNG baseline 73.99).

Per-scene exp00 (Score vgg): HCM0181 .7398 | HCM0193 .7361 | HCM0204 .7419 | hcm0031 .7309 | hcm0034 .7453

## AUDIT ROUND 3 (2026-07-13) — ensemble numerics CONFIRMED on HCM0181 (6 members available)

| variant | Score×100 | verdict |
|---|---|---|
| best single (e13/e10 gate1) | 75.15 | baseline |
| mean3 (gate1+2+3 = exp18) | 75.871 | |
| **mean4 (+gate0)** | **75.980** | **best — 4th member +0.11; adopt** |
| mean5 (+e06stack, −0.28 single) | 75.977 | flat — weak member adds nothing |
| mean6 (+e03ga15, −0.97 single) | 75.892 | **NEGATIVE −0.09 — quality rule** |
| median3 / median5 | 75.789 / 75.940 | rejected (mean dominates) |
| linear-RGB mean3 | 75.860 | rejected −0.012 (average in sRGB) |
| weighted mean3 | 75.871 | identical to 4 d.p. — worthless |

Rules adopted: (1) members must be within ~0.15 pts of best single; (2) config-jitter (gate 0/1/2/3) generates decorrelated members — same-config reruns DON'T decorrelate (safe_state seeds all RNGs to 0); (3) float32 accumulate + single +0.5 round (uint8 // N costs −0.009); (4) PNG-source averaging ~+0.005 free; (5) 5th member ~+0.05 by 1/N fit. Private projection: mean2 ≈ +0.5, mean3 ≈ +0.65, mean4 ≈ +0.75 (exp18 realized ~85% of HCM0181 delta).
- Member gates: A=gate1 (champ, trained), B=gate5 (stock, queue13 training), C→gate2, D→gate0.
- Committed to repo: `ensemble_renders.py` (float mean, sRGB, PNG-source, dual-write out), `build_submission_zip.py` (single-generation encode from PNG archives, quality ladder to 350MB, CSV/CRC self-check) — closes audit provenance flag on the ephemeral q98ss2 builder.
- q98ss2 zip re-verified post-hoc: CRC OK, 434/434 exact CSV name match (case incl.), sampling=2 all scenes. NOTE: it was built from q100 JPEGs (pre-dual-write) = generation-2; superseded by ensemble zip built from PNGs.
- importance_vis_norm: numerator/denominator CONFIRMED consistent (same render pass, int32, clamp(1) correct). Caveat for e17 read: vis_norm+gate1 is looser at visibility margin → if e17 regresses, retry gate 2–3 before discarding; cleanest read would be hcm0031.
- Appearance recentering: CONFIRMED Adam-safe (in-place under no_grad post-step; tensor identity preserved). Watch [APP] |gain|max in e16 — >0.05 means concentrated exposure drift.
- gsplat port top pitfalls (ranked): parser scene-normalization transform must be applied to CSV test poses (and inverted w2c→c2w); double-undistortion of images_undist (feed PINHOLE cameras.bin or k=0); data_factor default 4 → set 1; cx as-is (no −0.5); antialiased mode must match train/test; cap_max default 1M → 5M. Smoke tests: (a) re-render training image >20dB, (b) assert parsed K/size, (c) 1k-iter mini-train vs FastGS render of same pose.

## AUDIT ROUND 4 (2026-07-13) — submission tools review + e14 postmortem

- **ensemble_renders.py: CONFIRMED ship-ready** (rounding = measured-optimal round-half-up; JPEG encode byte-parity with render path; naming divergence impossible with --names_from). Fixed per review: PNG-over-JPEG preference was inverted when a dir holds both (glob order); added palette-PNG (quantized source) warning.
- **build_submission_zip.py: CONFIRMED ship-ready** (single-generation guarantee holds on every ladder path; termination bounded; arcname CSV-exact). Hardened per review: image-dimension assert vs CSV w/h (a stale wrong-res png_dir previously passed all checks → would score 0 organizer-side), final zip-size assert (blob budget ignored ~150B/entry zip overhead), per-scene duplicate-name assert, makedirs guard.
- **Cap guard added** (audit sketch, applied): `--max_gaussians` (default 0 = off) in densify_and_prune_fastgs — zeroes metric_mask at cap so growth stops but all prune paths keep running. Deterministic; VRAM-probe variant rejected (nondeterministic under fragmentation). Suggested value for this box: 8M. run_scenes.sh `timeout 2h` + break-on-124 deferred — script is mid-execution by live queue; apply when queue dead.
- **e15 blow-up risk LOW** (hit counter is not opacity-weighted; higher opacity terminates rays earlier → same-or-fewer counted hits; expect ~4.5-5M). **e17 MODERATE**: at gate1 a single-view gaussian needs only 2 flagged hits → peripheral churn possible. Tripwire: if N crosses ~7M before iter 10k, kill and rerun as vis_norm+gate2. HCM0181 is a weak testbed for e17 (tower-centric); null result there ≠ dead lever, hcm0031 would show it.
- e14 postmortem confirmed: gate −1 removes the only volume regulator ahead of grad_abs 0.00015; budgeted multinomial prune removes ≤~half of candidates per round → compounding growth. Zero info lost by killing.

## TRACK B PROGRESS (2026-07-13, cycle 5)

- **gsplat 1.5.3 BUILT** after 5 attempts. Root causes for the record: pip build isolation hides torch (need `--no-build-isolation` + `CUDA_HOME=$CONDA_PREFIX`); building on /mnt/d DrvFs = ~11 min/object (48 objects) — always build on ext4; gsplat main uses `cudaEventCreate(&ev, flags)` (CUDA ≥12.9 only) → patched to `cudaEventCreateWithFlags` in cuda/csrc/Utils.cpp (both source copies); NEVER run `import gsplat` with cwd inside the source tree (shadows the wheel, silently starts a full JIT rebuild).
- **NEW CODE `gsplat_track/train_gsplat.py` + `render_gsplat.py`**: MCMC trainer + CSV renderer per audit round-3 pitfall list — no gsplat Parser (raw COLMAP world, no normalization/double-undistort/data_factor), cap_max 5M, antialiased train+render, lpips-ft λ0.1@25k port, renderer pads via principal-point shift then reuses DistortionWarp + dual-write.
- **Smoke (a) geom_check PASSED: 0.81px median reprojection (HCM0181)** — pose/K conventions confirmed. Discovery: competition sparse models store images.bin xys at ORIGINAL 5280×3956 coords while cameras.bin is organizer-rescaled to 1320×989 (÷4). Harmless (nothing in either pipeline consumes xys — audit verifying), but any future xys consumer must scale.
- Smoke (c) 1k-iter mini-train PENDING GPU: GPU1 wedged by e14 (kill still awaiting user), GPU0 under standing "light use only, no training" rule (note: omnivoice has been dead since the reboot — rule may be revisitable).
- Audit round 5 dispatched: line review of the adapter vs gsplat 1.5.3 API (MCMC opacity-space contract, step_post_backward lr semantics, shapes), xys-consumer sweep.

## AUDIT ROUND 5 (2026-07-13) — Track B adapter review vs gsplat 1.5.3 source

- **2 real bugs found & FIXED pre-smoke:** (1) BLOCKING: load_scene kept all images.bin entries (388 poses incl. test+dropped; only train files exist) → FileNotFoundError in-loop + scene_scale inflation. Now filtered by file existence before scene_scale (240/371 kept on HCM0181). (2) SILENT: step_post_backward was called between backward() and opt.step() — MCMC relocation rebuilds params with grad=None, so every refine event (~245×) silently no-op'd the optimizer step. Moved after opt.step per simple_trainer reference.
- Quality deviations fixed: init_opa 0.1→0.5 (MCMC reference; relocation samples ∝ opacity), loss now on UNCLAMPED render (clamp zeroes grads at saturated sky), lpips term stays clamped (scoring parity).
- **CONFIRMED correct:** opacity contract (raw logits in params, sigmoid to rasterization — MCMC sigmoids/logits internally), means-lr contract (decayed lr both to optimizer and step_post_backward noise), all rasterization arg shapes, pixel convention (gsplat (i+0.5) centers == ndc2Pix == GT; cx=660.0 as-is), fused_ssim, RGB2SH, lrs, reg formulas, scene_scale recipe, renderer pad-by-principal-point.
- xys 4× quirk CONFIRMED harmless: nothing in FastGS consumes images.bin xys.
- Later opportunities logged: 1.5.3 rasterization supports radial_coeffs natively (could replace DistortionWarp entirely — A/B after warp path validates); noise_injection_stop_iter=25k for a noise-free lpips-ft phase; --seed jitter for Track B ensemble members.
- **USER: "maxxing use GPU0"** — GPU0 fully open. queue15 launched there: Track B 1k smoke → ladder e15/e16/e17 (e17 with --max_gaussians 8M guard) → private member B. GPU1 still wedged by e14; when it dies, GPU1 → Track B full 30k run.

## TRACK B SMOKE VALIDATED + FULL RUN LAUNCHED (2026-07-13)

- **Smoke (c) PASSED**: 1k-iter gsplat MCMC on HCM0181 → 273k gaussians, loss .343→.117, SSIM .25→.66; full render path (60 CSV poses, distortion warp-back) scored **PSNR 19.78 / SSIM .600 / LPIPS-vgg .469** vs GT = "aligned but undertrained" signature (convention bug would be <12dB). Adapter validated end-to-end after round-5 fixes + scipy float64→float32 scales fix.
- User killed wedged queue13/e14 (`kill -TERM -- -1951`) → GPU1 free; e14gateoff partial dir removed. Planned run_scenes.sh GPU1 collision guard became moot (never installed).
- **RUNNING — trackb_full (GPU1): exp19 = gsplat MCMC 30k, cap 5M, antialiased, lpips λ0.1@25k, HCM0181.** A/B vs FastGS champion single 75.1459. This is the decisive Track A vs Track B datapoint for the 80-target.
- RUNNING — queue15 (GPU0): e15ceil95 (peak phase) → e16app → e17visnorm(+8M cap) → private member B.

| exp19 | **TRACK B first full run**: gsplat MCMC 30k cap5M antialiased lpips@25k (gsplatB1) | HCM0181 | 24.2208 | .8388 | .1298 | **74.5072** | **−0.64 vs champ** | ~1.2h GPU1 | First untuned run lands within 0.64 of the 6-experiment-tuned FastGS champion — method competitive, not yet ahead. Cap 5M hit at 10k. Train ssim .930 vs test .839. Known defect: MCMC noise injected through the lpips phase |
| exp20 | Track B + noise_injection_stop_iter=25k (gsplatB2) | HCM0181 | | | | | | ~1.2h | RUNNING GPU1 — one variable: lpips phase polishes a noise-free model |

## AUDIT ROUND 6 + exp20 (2026-07-13) — root cause of Track B gap ISOLATED

- **exp20 (noise_stop 25k): 74.5067 — EXACT tie with exp19 (74.5072). Noise-through-lpips was a non-factor.**
- **Audit round 6 CONFIRMED root cause on ckpt stats: OPACITY STRUCTURE.** gsplatB1 vs FastGS champ: median opacity .097 vs .511; 21.4% dead splats (<0.005) vs 0%; opacity-mass 1.21M vs 2.61M-equiv; thin-structure splats (maxscale<.01) are 12% opaque vs 46%. Scale distributions near-identical → NOT an allocation/absgrad issue. Mechanism: constant opacity_reg pull + Adam scale-invariance drains weak-gradient (thin-structure) splats; MCMC keeps a dead reservoir at the relocation boundary. FastGS instead prunes hard and lets survivors saturate.
- **exp21 (gsplatB3) RUNNING GPU1: opacity_reg 0.01→0.002** (audit rank-1, expected +0.2..0.4). Rank-2 ready: warm-start `--init_ply` implemented in train_gsplat.py (FastGS champ ply → MCMC+antialiased finetune; f_rest channel-major transpose per audit). Rank-3 cap 8M only after reg fix.
- **Native radial_coeffs render lever CLOSED**: requires with_ut=True, which gsplat 1.5.3 hard-rejects with rasterize_mode="antialiased" (rendering.py:170) — classic-mode render of antialiased-trained model = dim mismatch. DistortionWarp stays.
- exp20 semantics verified by audit (noise off for steps ≥25000, lpips active >25000 — clean split; boundary step harmless).

| exp21 | Track B opacity_reg 0.01→0.002 (gsplatB3) | HCM0181 | 24.1333 | .8356 | .1316 | 74.2841 | **−0.22 vs B1/B2** | ~1.2h | **REJECT — OVERFIT.** Train SSIM .9343 (best of any B run) but ALL test metrics down. Audit rank-1 prediction inverted: less opacity reg = more train capacity, worse generalization. Opacity-structure diagnosis stands; reg-tuning is the wrong route to it. Keep opacity_reg 0.01 |
| exp22 | Track B WARM-START from FastGS champ ply, 15k finetune (5k lpips tail, cap 5.5M, refine/noise→10k) (gsplatB4warm) | HCM0181 | 24.2174 | .8423 | .1244 | **74.8205** | **+0.31 vs B1/B2, −0.33 vs champ** | ~0.7h | **BEST TRACK B.** Warm-start imports FastGS opacity structure and MCMC finetune keeps most of it — confirms round-6 opacity diagnosis by construction. Still behind champ: MCMC relocation partially re-drains opacity. As ensemble member: exceeds the 0.15-band rule (−0.33) but is the only different-renderer-family member available — borderline, A/B in ensemble before adopting |
| exp23 | idea 1c: per-image 3x4 affine + trajectory-interpolated test params (gsplatB5affine) | HCM0181 | 24.0304 | .8388 | .1369 | 74.1063 | **−0.40 vs B1/B2** | ~1.2h | **REJECT.** Same failure as FastGS e16 (−0.91): per-image affine absorbs real signal on this data even with L2-to-identity reg (λ 0.1) and trajectory-interpolated test params. Idea 1c dead in BOTH tracks — appearance variation on these scenes is not low-dim affine |
| exp24 | idea 1b: bilateral grid 16x16x8, identity at test (gsplatB6bilagrid) | HCM0181 | 17.3790 | .7586 | .1575 | 66.8850 | **−7.6 (!!)** | ~1.2h | **CATASTROPHIC REJECT.** PSNR 17.4dB: per-view grids absorbed large low-freq signal (exposure/tone) into per-view params; rendering test views WITHOUT a grid (identity) exposes the mismatch. Would need grid-interpolation at test (like 1c) — but 1c already proved interpolated per-view color transforms lose on this data. Idea 1b closed |
| exp25 | idea 1a: PPISP, controller activation 0.8×iters=24k (gsplatB7ppisp) | HCM0181 | | | | | | killed @~8k | **KILLED — audit round 7 CRITICAL: from activation, controller distillation does rgb.detach() → splats get ZERO photometric gradient for 24k-30k; the whole λ_lpips phase (25k+) would train only the controller while opacity/scale reg drained the splats unopposed.** Restarted as exp25b (~15 min wall lost) |
| exp25b | idea 1a CORRECTED: --ppisp_activation 29/30 (activation 29k) + lpips_from 24000 + noise_stop 24000 → live-radiance LPIPS 24k-29k, then 1k distillation with reg zeroed (new guard in train_gsplat.py) (gsplatB7ppisp2) | HCM0181 | 24.1893 | .8416 | .1265 | 74.6992 | **+0.19 vs exp20 base** | ~1.3h | **IDEA 1a: modest genuine positive on Track B** (74.51→74.70; alex-LPIPS also better than base). But below exp22 warm (74.82) and champ (75.15) — PPISP doesn't change the Track B ranking, and porting to FastGS is a new integration for <0.2 expected. Idea 1 CLOSED: 1b/1c dead, 1a positive-but-not-adopted. Audit confirmed integration clean (scheduler alignment exact, frame_idx=None true controller path, pre-warp application sound). WARNING on record: never render --ppisp from a ckpt killed before activation (controller would be random-init) |
| exp26 | audit item-3 pick: warm-start PURE finetune — refine_stop 0 (no relocation), noise 0, opacity_reg 0, scale_reg 0, 10k iters, lpips_from 3000, means_lr ×0.3 (gsplatB8pure) | HCM0181 | 24.2349 | .8439 | .1219 | **74.9809** | **+0.16 vs exp22; −0.165 vs champ** | ~0.5h | **BEST TRACK B** — audit prediction (+0.15..0.35) hit at the low end. LPIPS .1219 matches champ (.1220); gap is PSNR/SSIM. Confirms drain/relocation diagnosis. As ensemble member: different-renderer family AND now within ~0.17 of best — strong 5th-member candidate (round-7 A/B says additive-only). Next lever if iterated: 15k pure @ lpips_from 5000, or opacity-lr↓ |

## AUDIT ROUND 7 (2026-07-13) — PPISP detach catch + ensemble A/B ran + 3DGUT pre-flight

- **PPISP distillation detach (CONFIRMED, critical)**: see exp25 row. Fix applied to train_gsplat.py: `--ppisp_activation` flag (default 29/30) + opacity/scale reg zeroed from activation step.
- **Ensemble A/B with B4warm (audit RAN it, full-60 PSNR/SSIM + 10-img LPIPS subset): mean5B (mean4 + B4warm as 5th) ≈ +0.05 over mean4; swapping B4warm IN for gate0 = −0.03.** Different-renderer decorrelation beats the 0.15-band rule only ADDITIVELY. Action: B4warm goes in as member 5, never as a replacement. Re-run A/B if exp26 closes the gap (add-value should grow).
- **exp24 bilagrid postmortem CONFIRMED integration-inherent, not API misuse**: usage matches upstream exactly (identity init, TV 10, normalized xy). Structural defect: nothing anchors grids to identity after init; upstream masks this because gsplat eval applies GT-fitted color_correct — unusable in competition. Revivable only with identity-anchor reg; per-view-appearance family now 3 strikes — no more slots.
- **3DGUT pre-flight (idea 2) — half-day effort, inside budget**: classic + with_ut + with_eval3d + radial_coeffs [k,0,0,0,0,0] + camera_model pinhole, train+render identical; GT = train/images (original distorted 1320×989 — VIRGIN pixels, removes the undistort INTER_CUBIC resample generation every current model is fit to — the real upside); K unchanged, no pad/warp; MCMC ops confirmed pure-3D (compatible). EV caution: classic mode surrenders antialiased AA. Smokes: distorted-GT train-pose check, UT-vs-warp A/B on same ckpt, 1k-iter wall-clock (expect 1.5-2.5× slower).
- **--radial flag risk (CONFIRMED)**: render_gsplat --radial passes radial_coeffs WITHOUT with_ut — unverified whether plain projection consumes or silently ignores them. Already marked CLOSED LEVER; never use for submissions until the UT-vs-warp smoke proves distortion appears in output.

| exp27 | **idea 2 (3DGUT)**: distorted-space train on ORIGINAL train/images (virgin pixels), classic + with_ut + with_eval3d + radial k1; native distorted render, no warp (gsplatB9ut) | HCM0181 | 24.2540 | .8435 | **.1161** | **75.2143** | **+0.07 vs champ — NEW BEST SINGLE** | ~1.4h (UT overhead mild) | **IDEA 2 WINS on first untuned run.** Entirely LPIPS: vgg .1161 vs champ .1220 (alex .0721 vs .0816 — genuine perceptual, not backbone artifact); PSNR/SSIM ≈ par (−0.06dB/−.004). Virgin-pixels effect confirmed: every other model fits once-resampled undistorted GT. UT smokes passed (k1 +0.009004 consumed: corner nat-vs-warp divergence 2.2-4.0/255 < center; PSNR-vs-GT 20.6dB @3k; 3k train 48.9s). 1st attempt died on script bug (`/usr/bin/time` can't exec env-prefix). NEXT: 4-scene public validation + ensemble A/B (pubUT queue) |

| exp28 | UT cap_max 5M→8M (gsplatB10ut8M) | HCM0181 | 24.3392 | .8480 | **.1116** | **75.5783** | **+0.36 vs exp27 — NEW BEST SINGLE** | ~1.6h | Audit rank-3 knob hit its band's top. All metrics up (alex .0676). exp27 was genuinely growth-limited. **cap 8M = new UT default.** exp29 (60k iters @ cap8M, schedule scaled 50k) RUNNING GPU1 |
| exp29 | UT cap8M + 60k iters (refine/noise/lpips_from 50k) (gsplatB11ut60k) | HCM0181 | 24.4646 | .8526 | **.1089** | **75.8989** | **+0.32 vs exp28 — NEW BEST SINGLE** | ~3.3h | Schedule knob also delivers; both audit rank-3 sub-knobs positive and additive so far (75.21→75.58→75.90 in 24h; single now −0.05 vs old FastGS m4 ensemble). **60k/8M = production UT recipe** — private members (trained at 30k/5M exp27 cfg) are 1 recipe-generation behind; retrain ≈13h across both GPUs, schedule after IBR/ppft results land. exp30 (seed-7 member @30k/8M, composition A/B only) next on GPU1 |

## ENSEMBLE A/B ROUND 3 (2026-07-14 ~05:15, HCM0181) — 2nd UT seed is the next big member

| variant | PSNR | SSIM | LPIPS(vgg) | Score×100 |
|---|---|---|---|---|
| m4 (4 gates, ref) | 24.69 | .8574 | .1146 | 75.9492 |
| m5utA = gates + UT8M(s42) | 24.90 | .8632 | .1090 | 76.4756 |
| utpair = UT8M(s42)+UT8M(s7) | 24.69 | .8575 | .1045 | 76.3574 |
| utpair60 = UT60k(s42)+UT8M(s7) | 24.78 | .8603 | **.1025** | 76.5738 |
| m6ut2 = gates + 2×UT(30k) | 25.01 | .8661 | .1062 | 76.7411 |
| **m6ut60 = gates + UT60k + UT8M(s7)** | **25.04** | **.8669** | .1055 | **76.8100** |

- exp30 seed-7 single 75.6112 ≈ seed-42 75.5783 (recipe-equivalent, decorrelated). **Seed jitter works in the UT family** (MCMC data-order divergence) — unlike FastGS (safe_state pins all seeds).
- Adopted: **m6ut60 composition** (4 gates + best-recipe UT + second-seed UT) = 76.81 local, +0.86 over yesterday's ceiling. 2-model utpair60 alone = 76.57 with the best LPIPS anywhere (.1025) — the UT family carries LPIPS; gates carry PSNR/SSIM.
- Production gap: private scenes have ONE UT member at the old 30k/5M recipe. Round-6 needs per scene: +1 UT@60k/8M (s42) +1 UT@30k/8M (s7) ≈ 5h/scene both GPUs ≈ 20h total. Schedule vs IBR outcome (~07:00).

## *** THE PSNR CEILING PROOF (2026-07-14 13:45) — WE WERE CLIMBING THE WRONG HILL ***

**User (via ChatGPT) proved the binding constraint. Verified exactly:**
- At our PSNR 25.04 (m6ut60), even with **PERFECT LPIPS=0 AND PERFECT SSIM=1.0**: score = 0.4 + 0.3 + 0.3(25.04/50) = **85.02 < top1 85.94**.
- **Therefore top1 MUST have higher PSNR than us.** Inverting: 85.94 at plausible (LPIPS .04, SSIM .94) requires **PSNR ≈ 32 dB** — we are at 25.0, i.e. **+5 to +8 dB behind**.
- Our measured PSNR levers: UT cap5M→8M **+0.085 dB**; UT 30k→60k **+0.125 dB**; ensemble m4→m6ut60 **+0.350 dB**. We harvest +0.1 dB/lever and need +6. **No incremental combination closes this.** LPIPS (weight 0.4) has only 4.2 points left in it total; the gap is 8.6.
- Corollary: every optimization this project has made was LPIPS-first. That was rational for 74→77, and is a dead end for 77→85.

### THREE DIAGNOSTICS (D1/D2/D3), all run — they corner the answer

| D | question | result | verdict |
|---|---|---|---|
| **D1** | is the gap exposure/WB? | **ORACLE** color-correct (3×4 affine fitted TO GT, per test image — the max any appearance method could ever achieve): **+0.09 dB** (UT single 24.277→24.374; ensemble 24.846→24.939) | **APPEARANCE IS DEAD.** Worth 0.1 dB, not 6. Confirms idea-1 closure from a completely different direction. Post-net colour compensation also capped at ~0.1 dB |
| **D2** | registration/generalization vs underfit? | same UT model: **TRAIN views 27.09 dB** vs TEST 24.47 dB → gap **2.6 dB** | Generalization gap is real but SMALL. **The alarming part: in-sample fit is only 27 dB.** Even with PERFECT generalization we'd hit 27, still 4-6 dB short. **Our reconstruction is underfit for PSNR** |
| **D3** | is photo-reuse geometrically possible? | nearest train pose to each test pose: **11.8° mean view-angle** difference, 5.9% of scene radius. **Oracle "paste nearest train photo" = 9.4 dB** (vs our render 24.5) | **PHOTO-REUSE IS DEAD.** Huge parallax — the poses are NOT near-duplicates. Explains IBR's death mechanistically: sub-pixel reprojection at 11.8° parallax needs near-perfect depth |

### ROOT CAUSE FOUND: our loss never optimized PSNR

Stock loss = 0.8·L1 + 0.2·DSSIM (+LPIPS tail). **L1 is a median estimator; PSNR needs the mean.** There has never been an MSE term anywhere in this project.

**Exact metric-matched loss derived from the score formula** (idea 3, now TOP priority):
S = 0.4(1−LPIPS) + 0.3·SSIM + 0.3·PSNR/50, and PSNR = −10·log₁₀(MSE)
⇒ **maximize S ≡ minimize  0.4·LPIPS + 0.3·(1−SSIM) + 0.02606·ln(MSE)**   [0.06/ln10]
The log-MSE term is **self-scaling** (∂/∂mse = 0.026/mse → grad 8.2 at 25 dB, 26.1 at 30 dB — it fights harder the better you get). Implemented as `train_gsplat.py --metric_loss` (+ `--init_ckpt` for UT warm-starts).

**exp31 RUNNING (GPU1, promoted ahead of the UT60k retrain):** warm-start exp29 (60k/8M UT, 75.8989 / PSNR 24.4646), 8k-step pure refit (no densify/noise, means-lr ×0.3), two arms:
- A `ut60kMetric`: exact loss, LPIPS weight 0.4 from step 0
- B `ut60kMetricNoLp`: metric loss with LPIPS OFF — isolates how much PSNR the log-MSE term alone can buy, and what LPIPS *costs* in PSNR

## STRATEGY PIVOT: CONSOLIDATION (2026-07-14 13:20, audit round 10)

- **Per-pose finetune (task 13) CLOSED — no signal.** Color-only probe (6 poses stride-10, 200 steps, no LPIPS, 140s wall): tuned 74.2839 vs base 74.1914 = **+0.09, inside the ±0.2-0.3 noise band** for n=6; metrics disagree internally (PSNR +0.22/SSIM +.006 but LPIPS-vgg WORSE .1263 vs .1210 = geometry-free color tuning trades perceptual for pixel fit). Full version separately compute-infeasible (>8 min/pose → 64h for 480 imgs). Per-segment redesign SKIPPED per audit gate (probe < +0.15).
- **IBR (task 12) CLOSED** — flow-corrected arm implemented but not run (deprioritized).
- **AUDIT ROUND 10 HONEST VERDICT (agreed): 85.94 is a different method class; our realistic ceiling is ~79-80.** Both theories that could explain top1 are now empirically dead (naive photo-IBR: misregistration −4..−7; per-pose specialization: no signal + compute-bound). Recommendation adopted: **consolidate, don't moonshot.**
- **CONSOLIDATION LADDER (both GPUs, running):**
  1. **Retrain all 8 private UT members at 60k/8M** (shipped R5 members are old 30k/5M — measured +0.68/model on public). GPU0: HCM0249/0254/0276/1439 (started 13:21). GPU1: chains after seed-7 members → HNI0131/0265/0366/0437. ~22 GPU-h ≈ 1.5 days. Negative-k pair (HNI0131/0265) auto-routed to `--ut_render warp`.
  2. Family-weight sweep on private (free, CPU).
  3. Seed-13 UT members (more decorrelated variance, ~+0.1-0.2).
  Projected consolidated LB: **~78-79**.
- Only sanctioned moonshot per audit: ONE bounded, user-approved probe of learned frame interpolation (RIFE/FILM-class) — requires user approval (pretrained net beyond VGG/Alex, Rule 10). NOT started; audit's own honest odds of reaching 85 with it: low (bounded by the same registration enemy that killed IBR).

## ZOMBIE memD + per-pose finetune compute reality (2026-07-14 ~12:45)

- **ZOMBIE FOUND**: a memD (gate0) `train.py` survived the 23:35 kill for **12 HOURS** on GPU0. Root cause: killing queueCD.sh (parent) with SIGTERM did NOT propagate to the child run_scenes.sh `for`-loop, which kept advancing scenes and respawning train.py. Killed the loop (run_scenes.sh PID) + its trainer. LESSON: kill the run_scenes.sh loop PID, not just queueCD.sh, or use process-group kill (`kill -- -PGID`). It held ~14GB → co-resided with ppft → caused the ppft VRAM-thrash misdiagnosis.
- **Per-pose finetune (task 13) compute-infeasible as designed**: even ALONE on clean GPU0 at 100% util, 400 steps (8M UT gaussians + LPIPS backward) = >8 min/pose → 480 test images = 64h. Out for production. Running a cheap effect-size PROBE: `perpose_finetune.py --color_only` (freeze means/quats/scales → only sh0/shN/opacities; ~5× faster, no floater risk — audit's safe variant) + no-LPIPS + 200 steps + 6 poses, to decide if the DIRECTION merits an efficient redesign before dropping.
- Reusable op: `/usr/bin/time` cannot exec an env-var-prefixed command — put `CUDA_VISIBLE_DEVICES=N` BEFORE `/usr/bin/time`, not after (2nd time this bit; also in trackb7).

## LEADERBOARD PROGRESSION (private set, actual)

| round | composition | private LB | Δ | key lever |
|---|---|---|---|---|
| R1 | single champ (g15+λ0.1+gate1) | 74.348 | — | baseline |
| R2 | mean2: gate1 + gate5 | 76.166 | +1.82 | ensemble + PNG-gen fix |
| R3 | mean3: + gate2 | 76.4005 | +0.23 | 3rd gate member |
| **R5** | **3 gates + 3DGUT(UT), family-weighted UT=0.5** | **77.2964** | **+0.90** | **3DGUT member + family weighting** |

- **R5 = 77.296 CONFIRMS the two big bets on the real LB**: 3DGUT as a member (single UT beats a 3-FastGS ensemble) + family-weighting (UT gets 0.5 of 4-way weight). +0.90 over R3 for adding ONE UT member at weight 0.5.
- Recalibrated LB↔proxy: R5 HCM0181 proxy 76.93 → LB 77.30 (+0.37); private continues to run slightly ABOVE public proxy for UT-heavy ensembles. Round-6 (3 gates + 2 UT, w_UT 0.6, HCM0181 proxy 77.04) projects **~77.6-77.8 LB**.
- Gap to top1 85.94 now 8.6. Standing lever ranking: per-pose finetune (task 13, smoke running) > UT recipe scaling on private (retrain members at 60k/8M ≈ +0.68/model) > flow-IBR (task 12, last shot) > idea 9 per-scene HPO.

## IBR A/B (2026-07-14 ~10:30) — NEGATIVE as implemented; one flow-correction shot left

| arm | coverage | PSNR | SSIM | LPIPS(vgg) | Score | Δ vs splat 75.5783 |
|---|---|---|---|---|---|---|
| loose (τ .04/.10) | 81.5% | 22.55 | .7468 | .1932 | 68.2064 | **−7.37** |
| base (τ .02/.06) | 76.0% | 23.03 | .7679 | .1784 | 69.7186 | **−5.86** |
| tight (τ .01/.04) | 67.6% | 23.40 | .7892 | .1642 | 71.1490 | **−4.43** |

- **Monotone dose-response: photo fraction ∝ damage.** ALL metrics degrade incl. LPIPS — misregistered real texture loses to soft rendered texture everywhere.
- **Diagnostic (12-img shift sweep + per-image): NO systematic offset (peak exactly 0.00 both axes — conventions correct, audit trace validated). Failure = STOCHASTIC per-pixel misregistration: splat ED depth not accurate enough for sub-pixel photo reprojection at 1320px.** The low-pass photometric gate is structurally blind to this (misalignment lives at high frequency).
- Read-across to the 85.94 theory: naive geometric photo-reuse does NOT explain top1; if they use photos, they align them (flow/learned) or optimize per-pose.
- **Arm 4 queued (last shot, task #12): --flow_correct** — cv2 DIS flow snaps each warped photo onto the splat render pre-gating (splat = alignment target; classical flow, Rule-10 clean), flow>3px rejected. If negative → IBR CLOSED, pivot fully to per-pose finetune + UT scaling.

## AUDIT ROUND 9 + FAMILY-WEIGHT SWEEP (2026-07-14 ~09:00)

- **Both game-changer scripts CONFIRMED sound by line review** (ibr_render.py: backprojection algebra correct, half-pixel conventions cancel end-to-end by identity trace, ED usage correct, sky/thin-edge pixels safely rejected to splat fallback; perpose_finetune.py: legal, no test-info leak, no state bleed). Pre-registered PSNR-lowering channels for reading the IBR A/B: double-resample softening (future win: IBR directly in distorted space, one resample), passed-gate exposure residual ≤ tau_pho. ppft caveats applied: --stride 5 added (first-12 rows = biased segment); baseline/ckpt confound checked clean (both exp28). Negative-k scenes: expect LOW IBR coverage, read separately.
- Progressive JPEG CONFIRMED zero-decode-risk (pure entropy reorganization, libjpeg-family universal). Ladder termination intact.
- **FAMILY-WEIGHTED ENSEMBLE = free +0.23**: audit predicted two-family weighting breaks the round-3 equal-weight null; sweep confirmed. 2-UT composition (4 gates + UT60k + UTs7): w_UT 1/3→76.81, 0.4→76.91, 0.5→77.01, **0.6→77.0447 PEAK**, 0.7→77.02. Broad optimum, PSNR also up. **New local ceiling 77.04** (yesterday 75.98). 1-UT composition (round-5 zip analog): w_UT 0.4→76.88, **0.5→76.93 peak**, 0.6→76.89. ensemble_renders.py gained --weights (uniform default, backward compatible).
- **Round-5 zip updated to family weights** (gates 1/6 each + UT 0.5). Round-6 target composition: 3 gates + 2 UTs, w_UT=0.6 (0.4/3 per gate, 0.3 per UT), pending privUTs7 completion (~19:00).
- IBR production hardening pre-registered (audit list): coverage floor 0.5 w/ per-scene composition switch, public-proxy PSNR guard ≥−0.15dB, uncovered-pixel identity assert, neighbor sanity, gain-saturation telemetry, negative-k pair defaults to non-IBR composition, IBR ships as ensemble MEMBER first.

## PACKAGING UPGRADE (2026-07-14 00:30) — q100 side quest

- **All-q100 is impossible under 350MB** (measured on real ensemble PNGs: best q100 variant = 4:2:0 progressive = 438MB). Information-theoretic wall, not an encoder setting.
- **Adopted: progressive encoding + ladder extended to start at q100** (build_submission_zip.py defaults now [100..95] + progressive=True). Progressive = pixel-identical at same q, −5% bytes (343.7→327.0MB @q98ss2) → headroom lets SMALL scenes ride the top rungs (competition averages per-scene, so few-image scenes are cheap score-per-MB).
- Dry-run verified on round-3 ABC PNGs: **347.7MB, HCM1439@q100, HNI0265+HNI0437@q99, five big scenes @q98-prog** — strictly ≥ old all-q98 quality at equal size. Applies automatically to tonight's round-5 zip.

## MEMD KILLED, PIPELINE REWIRED (2026-07-13 23:35)

- **memD (gate0) killed after scene 1**: HCM0249_memD took **3h09m** (vs ~25 min for memB/C) — gate0 grows 5.7M gaussians on private scenes (ply 1.42GB vs memC 1.01GB/memB 723MB). 8 scenes ≈ 24h → would block privUT_gpu0 + round-5 zip until tomorrow evening for a member worth ~+0.11 in pure-FastGS mean4 and less next to the UT member (same-space redundancy: B8pure added +0.05, m6 diluted). HCM0249_memD itself completed and is kept (usable spare); HCM0254_memD partial removed.
- Round-4 ABCD zip CANCELLED (ens4 waiter killed — memD will never complete). Round-3 ABC zip remains the shipped fallback.
- **Round-5 composition now A(gate1)+B(gate5)+C(gate2)+UT per scene → `sub_round5_ensABCUT_private.zip`**, on the ORIGINAL schedule (~05:45): privUT_gpu0 launched immediately on freed GPU0 (~05:20 done), GPU1 privUT→utfix→exp28 unchanged.
- Sub-pixel alignment sweep (user gap hypothesis): render-vs-GT shift sweep on HCM0181 UT renders peaks at −0.125px/+0.03dB = **no systematic misalignment** (aligned ≲0.1px; ÷4-rescale convention self-consistent). The flat-PSNR gap is (a) loss never had an MSE term (idea 3 pending), (b) reconstruction-vs-photo-reuse ceiling (IBR/per-pose finetune, tasks 12-13).

## AUDIT ROUND 8 (2026-07-13 ~23:15) — negative-k fold fault + game-changer ranking

**Mission A faults:**
- **CONFIRMED: HNI0131 & HNI0265 have k1 ≈ −0.115** (different lens; 14× |k|, opposite sign vs all 5 public validation scenes +0.008..+0.014, corner displacement ~110px). Negative k folds the forward distortion at r_u=1.704 → verified numerically on the finished HNI0131 UT ckpt: 13k-17k gaussians op>0.5 project in-frame via the folded branch → corner phantoms in UT-NATIVE renders (corner divergence up to 0.29 mean-abs vs FastGS member). Model itself is healthy (train views self-clean); the native render path is the exposure. Side discovery: on these two scenes the same-K undistortion CROPS the outer FoV ring → FastGS members A-D were never supervised in the test corner ring there (long-standing quiet weakness; UT member fixes exactly this).
- **CONFIRMED + FIXED: render_gsplat warp-fallback bug** — `--ut_render warp` dropped with_ut/with_eval3d (response-model mismatch vs training). Fixed to `with_ut=ut, with_eval3d=ut` (pinhole UT = valid, fold-free). Also: warp re-render must pass `--distort auto --sparse` or it silently emits undistorted pinhole.
- **REMEDIATION ARMED (utfix.sh)**: after privUT_gpu1 finishes, re-render HNI0131+HNI0265 via fixed warp path and swap into canonical dirs before ens5 fires; native kept as *_native; corner-diff stats logged. If warp render fails, native stays (mean-of-5 dilutes phantoms to 1/5 weight — degraded, not ruined).
- Rest of overnight chain audited SOUND: per-scene k1 (fresh process/scene), no pad leak in native path, no PPISP/app leakage, mode mismatch impossible, ens5 waits correct (hard-fails to round-4 zip if a member is missing), zip ladder sane, --ut cold-start SH ramp correct.

**Mission B — game-changer ranking (top1 85.94 implies LPIPS .03-.05 + PSNR ~30 = photo-reuse territory):**
1. **Depth-guided IBR — highest EV, Rule-10-legal, THE move**: per test pose render UT depth (render_mode RGB+ED) → backproject → sample the 2 nearest REAL train photos (distorted model, conventions verified) → blend with depth-consistency occlusion mask + angle weights + per-neighbor scalar exposure gain + LOW-FREQ residual gate vs splat render (gate on low-passed diff so photo high-freq detail survives); splat fills rejects. Expected LPIPS .115→.05-.07 on 85-95% of pixels ≈ **+2.5-4 pts**; ~150 LoC + half day; A/B on public GT with existing HCM0181 UT ckpt (~1 GPU-h). Bonus: solves negative-k corners with real distorted pixels.
2. **Per-test-pose local finetune** (audit steel-man for what top1 does): 300-500 steps on K≈4-6 nearest train frames per test pose, render, discard. Legal, seam-free, ~8-16 GPU-h/full set, +1.5-3; COMPOSES with IBR.
3. **UT capacity/schedule (boring reliable +0.2-0.5)**: cap_max 8M first (cap currently hit @9-10k — genuinely growth-limited; ~5.6GB params+Adam), then 45-60k iters.
4. Idea 9 pseudo-test HPO (+0.2-0.4, ~40 GPU-h — after 1-3). 5. RIFE/FILM (gated, expected ≤ IBR; only if IBR seam-limited). 6. MVS seeding (+0.1-0.2 HYPOTHESIS, skip unless idle). 7. Training supersampling: SKIP (VRAM infeasible + native-frequency argument).
**Composite path to ~80-82 by 30/07: cap8M/60k UT retrains → IBR prototype → productionize on best members + ensemble; per-pose finetune second wave. 85 needs both #1 and #2 near upper bounds.**

## IDEA 2 PUBLIC VALIDATION (2026-07-13 evening) — 3DGUT ADOPTED, 4/4 positive (HCM0204 pending)

| scene | champ single | UT single | Δ | notes |
|---|---|---|---|---|
| HCM0181 | 75.1459 | 75.2143 | +0.07 | exp27; densest testbed, smallest gain |
| hcm0031 | 73.6877 | 74.6170 | **+0.93** | sparsest scene; UT single ≥ 3-member ensemble (74.556) |
| hcm0034 | 75.4453 | 76.1116 | **+0.67** | UT single > ensemble3 (75.991) |
| HCM0193 | 74.6693 | 75.7081 | **+1.04** | biggest gain; LPIPS .1262 vs .1409 |
| HCM0204 | 75.0361 | 75.6814 | **+0.65** | validation COMPLETE |

- **FINAL: 5/5 positive, mean +0.67. Full-set UT single = 75.4665 vs champ 74.7969 — a single 3DGUT model beats the exp18 3-member ensemble (75.4452).** Every metric positive on every scene, LPIPS driving (−.006..−.015). The virgin-pixels effect grows with scene sparsity — exactly where the resample softening hurt most. **3DGUT is the new base method.** Follow-ups queued behind private members: UT+lpips-tune ladder, UT warm-start, UT ensemble members (seed jitter now meaningful — different data order ≠ same model).
- **LB result (user): sub_round2_ens2AB scored 76.166 private (round-1 single was 74.348 → mean2 realized +1.82 = 3.6× the public-derived +0.5 projection). Ensembling amplifies on private.** Current top1 = 85.94 — 9.8 pts; honest assessment: pipeline tops out ~78; step-change needs idea 9 (per-scene HPO) / idea 5 (MVS) territory.
- OVERNIGHT ARMED: privUT_gpu0 (HCM0249/0254/0276/1439, fires on MEMBER D DONE ~23:00), privUT_gpu1 (HNI0131/0265/0366/0437, fires on PUBUT QUEUE DONE ~23:50), ens5.sh → `sub_round5_ens5m5ut_private.zip` (~06:00, m5ut composition A+B+C+D+UT, +0.46 local over m4).

## ENSEMBLE A/B ROUND 2 (2026-07-13 ~18:20, HCM0181, full-60 GPU scoring) — B9ut is the member that matters

| variant | PSNR | SSIM | LPIPS(vgg) | Score×100 | verdict |
|---|---|---|---|---|---|
| m4 = gate1+2+3+0 | 24.6875 | .8574 | .1146 | 75.9492 | baseline (round-3 75.980 reproduced; −0.03 = JPEG-source members) |
| **m5ut = m4 + B9ut** | 24.8827 | .8625 | **.1099** | **76.4060** | **WINNER +0.457 — all three metrics improve.** 3DGUT member is decorrelated in GT-space, not just renderer family |
| m5pure = m4 + B8pure | 24.7073 | .8578 | .1141 | 75.9954 | +0.05 — same marginal add as B4warm (round 7); same-space members are near-redundant |
| m6utpure = m5ut + B8pure | 24.8697 | .8621 | .1103 | 76.3721 | −0.03 vs m5ut — B8pure dilutes; DO NOT include |

- **Adopted composition: 4 FastGS gates + per-scene B9ut (m5ut).** HCM0181 ceiling now 76.41 local (was 75.98). Private projection at 85% realization ≈ +0.39 over mean4 path.
- Consequence: private-8 needs UT models (~1.4h/scene ≈ 11h GPU; both GPUs free ~23:00-23:50 → overnight split ≈ done ~05:30). GATED on the 4-scene public validation finishing favorably (user rule).

## SUBMISSION ROUND 2 SHIPPED-READY (2026-07-13 16:12) — first ensemble private zip

- **`/mnt/d/avv/submissions/sub_round2_ens2AB_private.zip` BUILT + VERIFIED PASS**: CRC OK, 434/434 files exact CSV name match (8 scenes: 60/60/60/26/60/52/60/56), dims 1320×989 all, single-generation JPEG q98ss2 from PNG-source float-mean, 345.4MB ≤ 350. Members: A = champ gate1 (re-rendered with PNG dual-write to <scene>_champA), B = gate5 memB (all 8 trained on GPU0, ~25 min/scene). Round-3 private projection for mean2: ≈ +0.5 over single-champ (sub_round1 LB 74.348 → expect ≈ 74.85 LB).
- Per user standing rule (2026-07-13): every confirmed increase → new zip in D:/avv/submissions. mean3 (+memC gate2) and mean4 (+memD gate0) rebuilds follow automatically-in-workflow once queueCD lands (~19:45 / ~23:00).
- Ops note: session restart killed the queueCD/trackb7 waiters (empty logs); relaunched 16:25 with grep --line-buffered fix in queueCD (block-buffered grep was hiding run_scenes output from the monitor).

## IDEA 1 STATUS (2026-07-13) — 2 of 3 variants REJECTED, PPISP pending

- **1c affine: dead in both tracks** (FastGS e16 −0.91, Track B exp23 −0.40). Interpolating test params from trajectory neighbors did NOT rescue it — the transform absorbs scene signal, not just exposure.
- **1b bilagrid: catastrophic** (−7.6) as configured (identity at test). Only rescue would be test-time grid interpolation, which 1c's result predicts also loses. Closed.
- **1a PPISP (exp25)**: only surviving variant; controller inference at test avoids the identity-at-test mismatch that killed 1b. Result decides idea 1.
- Read-across: appearance modeling on this dataset hurts unless the test-time path predicts appearance from CONTENT (PPISP controller) rather than assuming identity or interpolating per-view params. Drone captures appear exposure-consistent enough that most "appearance" degrees of freedom just leak reconstruction error.
- Meanwhile GPU0 queue15: **private member B (gate5) training started** (HCM1439 first). e15 spare 75.0930, e16 REJECT, e17 75.1318 neutral-tie → FastGS ladder CLOSED, champion config final for members.

---

# 2026-07-14 — THE LENS FIELD: biggest single lever found (+0.55 pts, no retraining)

## Why we went looking

R6 landed 77.6629 (PSNR 25.9147 / SSIM .86854 / LPIPS .098553). The ceiling proof
says PSNR is the ONLY metric that can close the gap to 85.94 (perfect LPIPS+SSIM at
our PSNR = 85.55 < 85.94). Our submetrics have a strange shape: **LPIPS .0986 is
genuinely good, SSIM .869 is decent, PSNR 25.9 is bad.** Good-LPIPS/bad-PSNR is the
signature of a small fraction of pixels carrying enormous squared error — PSNR is
dominated by worst-case regions, LPIPS/SSIM largely forgive them. So: find where the
MSE actually lives.

## D4 — where the MSE lives (HCM0181, best public ensemble wut60)

| probe | result | verdict |
|---|---|---|
| per-image PSNR spread | mean 25.09, median 25.03, **std 1.25**, min 22.12, max 28.28 | **NO bad tail.** Lifting the worst 10 images to the median = only +0.30 dB |
| pixel concentration | **top 1% of px = 35.6% of SE; top 5% = 66.5%** | error is savagely concentrated → EDGES, not blur |
| by GT luminance | dark .74x / mid 1.12x / lite 1.13x / bright 1.06x | **sky is innocent** (no far-field catastrophe) |
| border 10% vs center | **1.40x vs 0.78x** | border carries 1.8x the center's error density |

**Kills idea 8** (test-pose-aware cleanup): there are no catastrophic views to clean.

## D5 — is it geometry, photometry, or irreducible? (oracle bounds on PUBLIC GT)

| oracle | dB | verdict |
|---|---|---|
| global integer shift | **−0.02** | poses have NO systematic offset |
| **dense local flow (≤6px)** | **+2.13 dB (+1.28 pts)** | **the content is RIGHT but LOCALLY DISPLACED** |
| radial gain/offset (vignetting) | +0.11 | photometry dead (3rd independent confirmation, cf. D1's +0.09) |

Radial SE profile is a **U**: center ring **2.33x**, mid-annulus 0.59x, corner **2.67x**.
Center spike = the tower itself (thin lattice/antennas). Monotone rise from r=0.4→1.0 =
classic under-parameterized lens.

## D6 — decompose the 2.13 dB (cumulative)

| source | marginal | pts | fix |
|---|---|---|---|
| **LENS** (shared radial warp) | +0.669 dB | **+0.40** | train-only, legal, cheap |
| **POSE** (per-image homography) | +0.233 dB | +0.14 | **idea 6 — NOT WORTH IT** |
| **GEOM** (residual dense flow) | +1.231 dB | +0.74 | hard; needs better 3D (idea 5 / MVS) |

**IDEA 6 IS DEAD — no need to ask the user.** The gated HIGH-RISK pose-refinement idea has
an oracle UPPER BOUND of +0.14 pts (and that oracle is a homography fit directly against
test GT). Not worth the rule risk. Crossed off.

## The smoking gun

Shared radial displacement curve: flat (±0.1 px) out to r=0.5, then **+0.44 px outward at
r≈0.67**, swinging to **−1.52 px inward at the corner**. We hand gsplat
`radial_coeffs = [k1, 0,0,0,0,0]` — ONE coefficient, from COLMAP SIMPLE_RADIAL, held
CONSTANT. All higher-order radial + all tangential + all thin-prism terms are pinned to
zero. gsplat supports the full OpenCV model (radial 6, tangential 2, thin-prism 4).

**`radial_coeffs` and `tangential_coeffs` take NO GRADIENT** (probed directly: `.grad` is
`None` after backward). They are baked into the CUDA kernel as constants. So they cannot
be learned by backprop — the fix must FIT the field, then either retrain with it or
correct post-render.

## D7 — is the curve a real lens property, or overfitting? (2-fold CV)

| | dB |
|---|---|
| baseline | 25.1153 |
| curve fit on ALL (in-sample, can cheat) | 25.9054 (+0.7901) |
| **curve fit on the OTHER fold (HELD OUT)** | **25.9034 (+0.7881 = +0.473 pts)** |
| fold-A vs fold-B curve correlation | **r = 0.9977**, max diff **0.079 px** |

Held-out gain == in-sample gain. Two disjoint halves of the test set produce the same
curve to 0.08 px. **It is a deterministic camera/pipeline property, not noise.**

Polynomial-only warp (what `radial_coeffs` COULD express) = +0.4165 dB — barely half.
The curve is non-monotone (bump then dive); a 3-term odd polynomial can't track it.
**So don't fight the renderer: correct post-render and capture the whole thing.**

## D8 — full 2D field beats the radial curve

| | dB |
|---|---|
| shared 2D field, in-sample | 26.1888 (+1.0735) |
| **shared 2D field, HELD OUT** | **26.0918 (+0.9765 = +0.586 pts)** |
| fold agreement | corr dx .935 / dy .893, mean diff 0.124 px |

Field magnitude: mean 0.29 px, max 2.32 px. Beats radial-only because it also captures
tangential + thin-prism + principal-point components.

## D9 — THE DECISIVE ONE: fit on TRAIN, apply to TEST

The private set has NO test GT — the field must be fit against TRAIN photos. Risk: the
model was OPTIMIZED on those views, so its gaussians may have already absorbed the lens
error there, leaving nothing to measure.

| | dB | |
|---|---|---|
| TRAIN-vs-TEST field agreement | corr dx **+0.939** / dy **+0.883**, mean diff 0.141 px | shape identical |
| magnitude ratio train/test | 0.723 | model DID partially absorb it on train views... |
| train renders + train field | 27.1135 → 27.6892 (+0.576) | |
| TEST single + TRAIN field | 24.4790 → 25.3620 (**+0.883**) | |
| **TEST ensemble + TRAIN field** | 25.0871 → **26.0116 (+0.9245 dB = +0.555 pts)** | **95% of the cheating oracle** |

**...but the SHAPE survives, so it transfers anyway.** A field fit with zero access to test
GT recovers 95% of what an oracle test-fitted field would.

## Legality (Rule 10)

Fit **exclusively on train images** (the ones the model already trains on). One automatic
per-scene field, applied programmatically. No external data. No test GT. No manual
per-image editing. This is a lens/pipeline calibration — exactly what a correct camera
model does inside the renderer; we do it post-render only because gsplat won't backprop
its distortion coefficients.

## Proportion

**+0.555 pts from a post-render warp with NO retraining**, vs **+0.10 pts** for the
13-GPU-hour R7 member upgrade. ~5.5x the gain at ~1/20th the cost, and the two are
ORTHOGONAL (the field applies to whatever renders exist, including R7's).

## Tooling (in repo)

- `gsplat_track/render_train.py` — render a model at its own TRAIN poses (render_gsplat
  only takes a test-pose CSV). Pipeline-identical to the test render path.
- `gsplat_track/fit_field.py` — mean dense flow (train render → train photo), saved .npy
- `gsplat_track/apply_field.py` — apply field to a dir of renders, write lossless PNG
  (feed to build_submission_zip.py; never re-encode a JPEG)

## Status / next

- 5/5 public validation running (`fieldval.sh`, GPU1, chained after exp31).
- If 5/5 positive → R7 = existing R6 ensemble PNGs + per-scene train-fitted field.
  Needs only train renders of the 8 private scenes (~5 min/scene). No retraining.
- GEOM residual (+0.74 pts) is now the largest remaining lever → idea 5 (MVS seeding).

## CAMERA AUDIT (2026-07-14) — one camera, thirteen different k1's

All 13 scenes (5 public + 8 private) share ONE physical camera:
identical 1320x989, identical principal point (660.00, 494.50), focal 925.2-928.2 (0.3% spread).

But COLMAP fit a DIFFERENT k1 per scene:

| scenes | k1 |
|---|---|
| 11 of 13 | **+0.0076 .. +0.0138** |
| **HNI0131, HNI0265** | **−0.1148** (15x outlier, OPPOSITE SIGN) |

**One camera cannot have two lenses.** Two conclusions:

1. **The k1 scatter is itself proof the SIMPLE_RADIAL model is mis-specified.** A correctly-
   specified model fit to the same camera on 13 datasets would converge to the same k1.
   It doesn't — because a single k1 cannot express this lens, so BA lands in a different
   local optimum per scene. This independently corroborates the lens-field finding.

2. **HNI0131/HNI0265's k1 = −0.115 is a DEGENERATE COLMAP FIT, not a lens.** The true
   camera is k1 ≈ +0.009. This is the same pair that forced the round-8 `--ut_render warp`
   fold workaround (UT-native forward distortion folds at r_u=1.704). The fold was a
   SYMPTOM: we were feeding the renderer a lens that does not exist. Their gaussians have
   been contorting to compensate.

**Consequence for the field fit: PER-SCENE is correct, not a compromise.** The residual
field of scene s = (true lens) − (that scene's k1 model). Since k1 varies per scene, so
does the residual. Do NOT pool the field across scenes.

**BLIND SPOT (flagged):** no public scene has negative k1 (all +0.008..+0.010), so the two
negative-k1 private scenes cannot be validated on public. Mitigation: the field is fit on
each scene's OWN train photos, so it self-corrects; and a large fitted-field magnitude on
those two scenes would itself confirm the story. Inspect field magnitudes per private scene
before shipping. Do NOT attempt to "fix" their k1 by hand — the given test poses live in the
frame COLMAP fit under that k1, so changing it desynchronizes poses/points.

## AUDIT ROUND 9 (2026-07-14) — 2 confirmed bugs caught BEFORE they shipped

**Q1 LEGALITY: CONFIRMED LEGITIMATE.** The field is a function of (train photos, train
poses, our model) — the same epistemic class as training the model itself. Audit traced
every leak path and found none: `load_scene()` filters views by file existence so test
images (absent from `train/images`) are dropped; `render_train` never reads the test CSV;
`fit_field` matches by stem so a stray test render finds no train GT and is skipped.
Defensible one-liner if ever asked: *"a per-scene sub-pixel calibration map, estimated from
the scene's own training photographs, correcting a residual lens-model error our renderer
cannot represent."*

**BUT: the real risk was operator error, and it's now closed.** The D5-D8 ORACLE fields
(fit against public TEST GT to size the prize) are on disk, and nothing stopped one from
reaching a submission. Guards added and verified firing:
- `fit_field.py`: asserts `--gt_dir` is not a `/test/` path. **Verified: hard-fails.**
- `fit_field.py`: writes a `.meta.json` provenance sidecar.
- `apply_field.py --strict`: refuses any field that cannot prove it was fit on train photos.
- `apply_field.py`: PNG-only input (a JPEG in = a 2nd generation, −0.14..−0.26) and refuses
  to re-apply to an already-corrected dir (double warp ≈ doubles the displacement error —
  it would have *increased* error by roughly what it removed, silently).

**CONFIRMED BUG 1 (would have mis-targeted the field on the 2 worst scenes):**
`render_train.py` always rendered UT-NATIVE, but HNI0131/HNI0265 (k1=−0.115) ship their
test renders through the WARP path. Fitting a sub-pixel field on one pixel pipeline and
applying it to another is wrong at exactly the 0.2-2px scale the field operates on.
FIXED: `render_train.py` gained `--ut_render {native,warp}` mirroring render_gsplat's warp
branch (padded pinhole canvas + DistortionWarp, `with_ut/with_eval3d` retained); privfield.sh
selects `warp` for those two scenes.

**CONFIRMED BUG 2 (a scene would have been silently dropped from R7):** `utq.sh` kept its
claim dir on train failure, so neither worker would retry — the scene would vanish from the
ensemble and only surface at the stem-equality assert (or ship as a 4-member scene). FIXED:
release the claim (`rmdir`) on failure, skip scenes whose ckpt already exists, and print
`!!! MISSING MEMBER` at the end. Same class as the 12h zombie-respawn incident: invisible
until far too late.

**Q2 SCALING: DO NOT scale the field by 1/0.72.** That constant was derived by comparing
train-fit to test-fit magnitudes *using test GT* — baking it in would import a test-GT-derived
hyperparameter, i.e. test-time fitting in disguise. It is the one legally-gray move available
and it buys ~0.03 pts (the unscaled field already gets 95% of the oracle). **Ship unscaled.**
Clean way to buy the last 5%: fit on HELD-OUT train views (train on 90%, fit field on the
unseen 10%) — zero test GT, removes the absorption bias by construction. Do it only if a GPU idles.

**Q3 POOLING: per-scene CONFIRMED correct** (cameras differ per scene; the negative-k1 pair is
a different distortion regime entirely). Caveat: HCM1439 has only 103 train frames — if its
field looks noisy, pool it with the other positive-k1 scenes (check cross-scene field corr first).

**Q5 METHODOLOGY: JPEG-artifact fitting RULED OUT** — codec artifacts are image-specific and
average to ~0 over 240 pairs; they cannot produce a radially-structured, CV-stable field
(r=0.9977 across folds, r=0.94/0.88 train→test). That is a lens signature, not a codec one.
**Open upgrade:** the field is fit on ONE model's train renders but applied to the ENSEMBLE
mean. Strictly more correct: fit on the ensemble-mean train renders (same estimator as the
shipped image). Worth ~+0.05-0.1 dB. Deferred to R8 — R7 ships the composition that public
validation actually tested.

**Q4 GEOM (+0.74 pts): idea 5 (MVS seeding) NOT recommended.** The residual is geometry in the
wrong *place*, not missing geometry; MCMC relocation erases initialization advantages by 30k
(round-6 finding). Much of the residual is irreducible anyway (wind-swayed cables, moving
vegetation, rolling shutter on a moving drone). Expected +0.1-0.2 at 2-3h/scene — below the
reliable ladder. **Idea 5 stays closed.**

**Audit bug 1 confirmed MATERIAL (not theoretical).** Smoke-tested the new `render_train.py
--ut_render warp` path (pad=4, output 1320x989 ✓) and diffed the two pipelines on the same
train view: **native vs warp = mean 2.365/255, max 131/255, 26.7% of pixels differ by >2.**
That is far above the 0.2-2 px scale the field operates on. Fitting the field on the native
pipeline and applying it to warp-rendered output would have mis-targeted the correction on
HNI0131/HNI0265 — the two scenes already most fragile. Fix tested end to end.

**Field magnitudes are stable across public scenes** (a lens should do this; a scene-specific
artifact would scatter): HCM0181 0.267/1.87, HCM0193 0.225/1.70, HCM0204 0.242/1.62 (mean/max px).

## FIELD CORRECTION: 5/5 PUBLIC VALIDATION — PASSED (2026-07-14)

Single gsplatB9ut model per scene; field fit on that scene's TRAIN renders vs TRAIN photos
(zero test GT), applied to its TEST renders.

| scene | Score(vgg) before | after | delta |
|---|---|---|---|
| HCM0181 | 75.26 | 76.49 | **+1.23** |
| HCM0193 | 75.75 | 76.52 | **+0.77** |
| HCM0204 | 75.72 | 76.63 | **+0.91** |
| hcm0031 | 74.65 | 75.25 | **+0.60** |
| hcm0034 | 76.16 | 77.07 | **+0.91** |
| **MEAN** | **75.5089** | **76.3925** | **+0.8836** |

**5/5 positive, no scene regressed.** Submetrics: PSNR 25.013 -> 25.645 (**+0.632 dB**),
SSIM .8493 -> .8649 (**+0.0156**), LPIPS(vgg) .1245 -> .1236 (−0.0009, flat).

Score decomposition of the +0.8836: SSIM contributes **+0.47**, PSNR **+0.38**, LPIPS +0.04.
**The field buys PSNR *and* SSIM** — bigger than the +0.555 projected from D9 (which measured
PSNR only). It is the largest validated single lever in the project, and it required NO
retraining.

R7 = R6 ensemble PNGs + per-scene train-fitted field. Launched (privfield.sh, GPU0).
R6 LB was 77.6629 -> **R7 projects ~78.3-78.6**.

## D10 — held-out CV of the PRIVATE fields (train-only; no public analogue exists)

HNI0131's fitted field came back with **max 5.69 px vs 0.99-1.68 px on every other scene** —
consistent with its degenerate k1=-0.115, but OUTSIDE the regime the 5/5 public validation
covered (public maxed at 1.9 px) and near the ±6 px flow clip. And there is no public
negative-k1 scene to validate against (the flagged blind spot). Shipping a 5.7px warp on
faith was not acceptable, so: **2-fold CV within each scene's own TRAIN views** (fit on fold A,
apply to held-out fold B). Train photos only — legal on private.

| scene | base | HELD-OUT | delta | fold corr dx/dy | max\|d\| | clip-sat | |
|---|---|---|---|---|---|---|---|
| HCM0249 | 25.2670 | 25.4721 | +0.2051 | +0.855 / +0.717 | 1.82px | 0.60% | PASS |
| HCM1439 | 30.3253 | 30.4862 | +0.1609 | +0.918 / +0.885 | 1.13px | 0.03% | PASS |
| **HNI0131** | 26.6817 | **27.1422** | **+0.4605** | **+0.965 / +0.961** | 6.06px | 0.87% | **PASS** |
| **HNI0265** | 25.3414 | **25.7177** | **+0.3764** | **+0.978 / +0.968** | 5.84px | 0.48% | **PASS** |

**The concern inverted: the two negative-k1 scenes are the BEST-behaved and gain the MOST.**
Their fold correlations (0.96-0.98) are the HIGHEST in the set, and their held-out gains
(+0.46/+0.38 dB) are more than double the others' (+0.16/+0.21). Clip saturation is
negligible (<1%), so the ±6px clip is not truncating the fit.

This is exactly what the degenerate-k1 diagnosis predicts: a badly wrong lens produces a large
but PERFECTLY SYSTEMATIC misregistration -> a large, highly-reproducible field -> a large,
legitimate correction. **The 5.69px max was the signal, not a warning.** Ship all 8 corrected.

(Note: held-out gains measured on TRAIN views understate the TEST gain, because the model has
partially absorbed the bias on views it trained on — the 0.72 magnitude-ratio effect. On public
HCM0181 the same field gave +0.883 dB on TEST. These numbers are a validity check, not a forecast.)

## R7 SUBMISSION BUILT + VERIFIED (2026-07-14)

`/mnt/d/avv/submissions/sub_round7_field_private.zip` — **R6 ensemble + per-scene
train-fitted lens field. NO retraining.**

VERIFIED: CRC OK · 434/434 filenames match each scene's test_poses.csv EXACTLY ·
all 1320x989 · single-generation JPEG from lossless PNG · **350.0MB** (limit 350.0).

Encode ladder: HCM1439 q100, HNI0265/HNI0437 q99, the other five q98. **The field
correction INCREASES encoded size** — removing misregistration blur adds genuine
high-frequency detail — so the ladder had to step 3 scenes down to q98 to fit. A small,
known cost against a +0.88 gain.

Per-scene fitted field magnitude (mean/max px):
HCM0249 .195/1.67 · HCM0254 .220/1.68 · HCM0276 .191/1.56 · HCM1439 .132/0.99 ·
**HNI0131 .275/5.69 (warp)** · **HNI0265 .301/5.75 (warp)** · HNI0366 .169/1.23 · HNI0437 .156/0.98

The two k1=-0.115 scenes have 3-4x the peak displacement of every other scene, and passed
the D10 held-out CV most convincingly of all (fold corr 0.96-0.98). Degenerate-k1 story
confirmed from three independent directions: COLMAP's own k1 scatter, the round-8 UT fold,
and the fitted field magnitude.

**Expectation: R6 77.6629 -> R7 ~78.3-78.6** (public delta was +0.8836 on single models;
the ensemble gain in D9 was slightly larger than single-model in PSNR, so this is not
optimistic). Submit and report back.

# EXTERNAL CONSULT (ChatGPT, 2026-07-14) — reconciled, + P4/P6 results

**The consult read this log at 341 lines — BEFORE the lens field.** Its central bet
(§2c: "shared-parameter errors transfer fully — intrinsics, unmodeled k2 — disproportionately
likely on HNI0131/HNI0265") is EXACTLY what we found and shipped today. Strong independent
corroboration, arrived after the fact.

**Our fix is strictly stronger than the P2 it proposed.** It suggests fitting k2/k3 into
`radial_coeffs`; we MEASURED that a 3-term odd polynomial captures only **+0.42 dB of the
+0.79 dB** — the real curve is non-monotone and no low-order radial polynomial tracks it.
The non-parametric field gets ~2x what P2-as-specified would.

## Falsified by today's data

**P1 (per-image pose noise sigma~0.3px as THE cap): bounded out.** D6's oracle gives per-image
pose only **+0.233 dB = +0.14 pts**, and the consult itself argues (correctly — its sharpest
paragraph) that refining TRAIN poses will NOT transfer: test poses carry the same noise, and a
noise-blurred map is close to MMSE-optimal under matched test-pose noise. Small AND
non-transferable. **HONEST CAVEAT: our oracle was a per-image HOMOGRAPHY, which cannot express
depth-dependent parallax** — a true 6-DoF pose error does exactly that, so some pose error IS
hiding inside the +1.23 dB GEOM residual. Bounded from one side, not dead.

## P4 — SH-degree clamp at render: CLOSED, NEGATIVE

| sh | Score(vgg) |
|---|---|
| 0 | 67.7580 |
| 1 | 69.1129 |
| 2 | 72.4849 |
| **3 (baseline)** | **75.8989** |

Monotone: clamping SH is **strictly harmful**. No hidden view-dependent-colour tax at 11.8 deg
extrapolation. sh=3 reproduces exp29's 75.8989 EXACTLY -> harness sound, negative trustworthy.
Cost: minutes. Lever closed.

## P6 — unsupervised corner ring: CONFIRMED, effect is LARGE

Premise quantified from intrinsics alone: on HNI0131/HNI0265 (k1=-0.115) the same-K
undistortion pushes the undistorted preimage to x=1406 against a 1320 canvas ->
**11.90% of every frame was NEVER supervised for the FastGS gates. 0.00% on every other scene.**
Yet the gates carry **0.4 of the ensemble weight there**.

Validated against **TRAIN GT** (HNI0131, n=40) — the only legal route, since no PUBLIC scene has
negative k1 and the defect cannot be reproduced on public data at all:

| | inside mask | in the RING |
|---|---|---|
| FastGS gate | 30.02 dB | **17.80 dB** (−12.22) |
| UT member | 28.91 dB | **21.13 dB** (−7.78) |

**In the ring the UT member beats the gate by +3.34 dB**, and the gate's 17.80 dB is genuinely
broken output. Inside the mask the gate is BETTER (30.02 vs 28.91) — which is exactly why the
fix must be SPATIAL, not a weight change.

FIX: `gsplat_track/fov_mask.py` (validity mask from intrinsics; 8px cosine feather) +
`ensemble_renders.py --masks` (per-pixel weight zeroing + renormalization). The mask is
all-ones on normal scenes (min=1.000), so it is a strict no-op there — safe to pass everywhere,
no special-casing. Expected ~+0.9 dB on those 2 scenes -> **~+0.13-0.2 pts overall**
(above the consult's +0.05-0.15 estimate).

R8 = R7 + masked composition. Building.

## Adopted, still to do
- Train-view PSNR as standing telemetry (D2 was the most informative number in this file).
- Append metric-refit to the running retrain queue (avoid retraining 8 scenes twice).
- Family-weight re-sweep after refit (free CPU; easy to ship stale weights).
- P2-extended: refit k2 then RETRAIN — the one thing the POST-HOC field does NOT do, i.e. stop
  the gaussians contorting at the source. Could bite into the GEOM residual.
- **USER ACTION: read top-1's per-metric breakdown** (we calibrated psnr_max from official
  breakdowns, so they may be visible) — replaces the ceiling PROOF with a MEASUREMENT. And:
  did any phase/track ever distribute higher-res imagery? images.bin stores keypoints at
  5280x3956 against /4 cameras => organizers COLMAPped at FULL RES; full-res originals exist.

Revised landing zone WITH the field: **~79-80** (the consult's 78.3-79.3 predates it).

## R7 PRIVATE SCORE: 78.39740 — the lens field CONFIRMED on the leaderboard

| | R6 | **R7** | delta | pts |
|---|---|---|---|---|
| PSNR | 25.914715 | **26.470226** | +0.5555 dB | +0.333 |
| SSIM | 86.854 | **88.1601** | +1.306 | +0.392 |
| LPIPS | 9.8553 | 9.832 | −0.023 | +0.009 |
| **Score** | 77.6629 | **78.39740** | | **+0.7345** |

Formula reproduces 78.39737 exactly. Public predicted +0.8836, private delivered **+0.7345** —
close, direction right. **As on public, SSIM paid MORE than PSNR** (+0.392 vs +0.333): a
sub-pixel misregistration fix is a STRUCTURAL correction and SSIM rewards it harder than MSE.

**LB progression: 74.348 -> 76.166 -> 76.4005 -> 77.2964 -> 77.6629 -> 78.39740.**
R7's +0.73 is the 2nd-largest jump ever (only R2's ensembling +1.82 was bigger) — and it
required NO retraining.

## RETRACTION: THE PSNR CEILING PROOF HAS EXPIRED

At R6's PSNR (25.91), perfect LPIPS=0 AND SSIM=1.0 capped us at **85.55** — a genuine 0.39-pt
structural lockout below 85.94. At R7's PSNR (26.47) that ceiling is now **85.882**. We are
**0.096 dB of PSNR** from the bound becoming permissive. **The hard-impossibility argument is
dead.** Do not keep citing it.

**But the practical gap is UNCHANGED, and the retraction is not good news:**

| at these perceptual metrics | PSNR needed for 85.94 | we have |
|---|---|---|
| our current LPIPS .098 / SSIM .882 | **39.04 dB** | 26.47 |
| excellent .05 / .93 | **33.40 dB** | 26.47 |
| near-perfect .03 / .96 | **30.57 dB** | 26.47 |

What died is the clean PROOF, not the 4-7 dB gap. Top-1 still implies ~31-34 dB.

**TOP-1 FORENSICS PATH CLOSED:** user confirms other teams' submetrics are NOT visible. We
cannot read top-1's triple. Dropping that line of inquiry rather than speculating further; the
+9.2 discrete jump (76.75 -> 85.94) and the full-res-originals question stay on the record
unresolved.

## R8 BUILT + VERIFIED — masked composition (P6)

`/mnt/d/avv/submissions/sub_round8_maskfield_private.zip` — R7 + FoV mask on HNI0131/HNI0265.
VERIFIED: CRC OK · 434/434 exact · 1320x989 · **342.7MB**.

Only the 2 negative-k1 scenes change (the mask is all-ones elsewhere -> strict no-op).
Encode ladder identical to R7 except HNI0437 q99->q98 (masking sharpens the ring, which
costs JPEG bytes -- a small real cost against the gain).

Expected **+0.13-0.2 -> ~78.53-78.60**. Basis: in the 12.9% unsupervised ring the gates
collapse to 17.80 dB vs the UT member's 21.13 dB (+3.34 dB, measured on TRAIN GT, n=40);
propagating that through the 0.4-weighted mean gives ~+0.9 dB on those 2 scenes.

# PIPELINE HARDENING (2026-07-14) — for the 16/07 dataset change

The organizer upgrades the dataset on 16/07; contents unknown. **Every trained model, fitted
field and mask is scene-specific and dies with the data — only the method and the code
survive.** Hardening pays in every scenario (replaced / added / harder).

- **`scripts/run_dataset.sh`** — ONE COMMAND: scene folders -> verified zip.
  `--data_root DIR --out DIR --zip FILE [--tier 1|2|3] [--gpus 0,1]`
  Per scene: train UT 60k/8M (N seeds) -> render test -> render TRAIN -> fit lens field on
  TRAIN photos -> build FoV mask -> family-weighted + per-pixel-masked ensemble -> apply
  field -> build + verify zip. **AUTO-DETECTS the negative-k1 case** and routes BOTH the test
  render and the field fit through the warp path (getting this mismatched is silent and
  mis-aims the correction). Tiers trade GPU time: 1 = 1 UT (~3h/scene), 2 = 2 UT seeds
  (~6h/scene, default), 3 = + FastGS gates (~9h/scene, best).
- **`scripts/verify_zip.py`** — every pre-ship check, permanent and exit-non-zero: CRC, exact
  filename match vs test_poses.csv, no extras, dimensions, JPEG format, 350MB cap.
  Regression-tested: PASSES the known-good R7, PASSES R8.

Target: **new dataset -> verified submission in hours, not days.**

## R8 PRIVATE SCORE: 78.39980 (+0.0024 over R7) — NEUTRAL. My reasoning was wrong.

| | R7 | R8 | delta |
|---|---|---|---|
| PSNR | 26.470226 | 26.476083 | +0.0059 dB |
| SSIM | 88.1601 | **88.1484** | **−0.0117** |
| LPIPS | 9.832 | 9.8259 | +0.0061 |
| Score | 78.39740 | **78.39980** | **+0.0024** |

Predicted +0.13-0.2. Got +0.002. **Postmortem, because the error is instructive:**

I argued that since the gates produce 17.80 dB in the ring while UT holds 21.13 dB, giving
the gates 0.4 weight there must be a loss. **That treats the ensemble as if the worst member
drags the mean down. It doesn't.** The pixel mean is the MSE-OPTIMAL combiner and it averages
out UNCORRELATED error:

    uncorrelated:  MSE = 0.4^2*sigma_gate^2 + 0.6^2*sigma_UT^2
                       = 0.16(0.0166) + 0.36(0.0077) = 0.0054  ->  22.7 dB
    UT alone:                                                      21.1 dB

**The blend is BETTER than UT alone.** The gates were not polluting the ring — they were
contributing decorrelated information, which is the entire point of ensembling. The train-GT
measurement (+3.34 dB gate-vs-UT) was CORRECT; the INFERENCE from it was not.

**LESSON (we already knew this and I failed to apply it): individual member quality does not
predict ensemble contribution.** It is exactly why the UT member paid at all despite being a
weaker single model. Do not mask/drop a member because it is bad in isolation — measure it
IN the ensemble.

P6 keeps its mask (marginally positive, fixes a real defect) but it is NOT a gain. Do not
count it.

### Secondary finding, and this one IS free: the encode ladder wastes headroom

R7 landed at 350.0MB (lucky). **R8 landed at 342.7MB of a 350MB budget — 7.3MB (2%) of quality
discarded** — and had already knocked HNI0437 (an untouched scene) from q99 to q98 to get there.
The ladder only ever steps DOWN; the final step overshoots and the leftover is thrown away.
That is why SSIM went DOWN.

FIXED: `build_submission_zip.py` gained a RECLAIM pass — after the descent, step scenes back
UP while they still fit (greedy, cheapest-first, 0.5MB zip-overhead margin). Rebuilt R8 as
`sub_round8b_reclaim_private.zip`: **348.9MB, HNI0265 back up to q100.** Matters much more on
the unseen 16/07 data, where scene sizes are unknown.

# AUDIT ROUND 10 — 4 CONFIRMED findings, all fixed before 16/07

**1. CONFIRMED CRITICAL: `run_dataset.sh` silently swallowed a failed scene OR SEED.**
Bare `wait` always returns 0, so `set -e` never fires on a backgrounded failure. Worst case:
a single-seed failure -> `DIRS` has one dir -> weights renormalize to 1.0 -> **the scene ships
as a 1-member "ensemble" with zero warning.** FIXED: atomic-claim workers (replacing a
barrier-batched loop that idled a GPU for hours on unequal scenes), per-scene `done/` markers,
and a HARD GATE before compose that aborts with a manifest if any scene is missing a seed,
a field or a mask.

**2. CONFIRMED: `fov_mask.py` Newton returns finite GARBAGE outside the shipped regime.**
My instinct was right; the mechanism was worse than I guessed. Not a wrong branch — **NO ROOT
EXISTS** once rd_max > g(r_fold). Measured by the audit: ru_max = **265** at k1=-0.20; **1117**
at f=700 (wider FoV). No NaN, no exception. The mask would come back ~100% "unsupervised" and
**silently zero an entire member family**, and even the wsum assert stays quiet because the
other member is unmasked. Our scenes are safe (rd_max 0.891 -> ru 1.009 << fold 1.704) but an
unseen dataset is exactly where this bites. FIXED: three asserts (root exists / converged /
correct branch). **Regression-verified: shipped scenes still give 11.90% and 0.00%; k1=-0.20,
k1=-0.30 and f=700 now all HARD-FAIL.**

**3. CONFIRMED: tier-3 weights were inconsistent.** 0.5+0.5+0.1333x3 normalizes to w_UT=0.714,
not the tuned 0.6. FIXED: w_UT is 1/NSEED at tier<=2, 0.3 each at tier 3.

**4. Q3 SETTLED: `0.5/0.5` for two UT seeds is CORRECT, no sweep needed.** Family weighting
existed to balance two families with COMPLEMENTARY error profiles. With one family of
equal-quality members, round-3 already measured score-weighted == uniform to 4 decimals.

**NEW: `scripts/preflight.py`** — the audit's keystone: "a 30-second preflight is the
difference between 'the run failed at 06:00 with a clear message' and 'the run burned 18
GPU-hours and died at compose'." Checks, per scene, BEFORE any GPU time: sparse/0 exists,
exactly ONE camera (`load_scene` asserts this — a multi-camera release would hard-fail),
camera model supported (and WARNS that RADIAL's k2 is silently ignored — worse than an assert),
|k1| inside the fov_mask root-existence bound, image dir non-empty, test CSV single-camera and
parseable, W/H agree between cameras.bin and the CSV (a mismatch scores 0 while passing every
name check). **Verified on the real private set: PASSES, and correctly routes HNI0131/HNI0265
to `warp`.**

**`scripts/verify_zip.py` hardened**: dimensions on EVERY image (was first-only — one wrong-size
frame passed), plus a degenerate-content check (all-black / stuck frame / wrong scene under the
right names are invisible to structural checks). Regression: still PASSES R7 and R8b.

# AUDIT ROUND 11 — my R8 postmortem was wrong in BOTH directions; k2-retrain killed

## Q1: R8's null is NOT a bug. But I was wrong twice.

The audit diffed the actual shipped R7 vs R8 pixels on HNI0131:

| region | mean \|R8 − R7\| |
|---|---|
| ring (12.89% of frame) | **6.030 / 255** |
| supervised centre | **0.000 / 255** (bit-identical) |

So the mask applied exactly as intended and `ensemble_renders`' renormalization is EXACT.
No bug. But then it computed the bound I never did:

| ring error model | score delta |
|---|---|
| perfectly correlated | **+0.129 pts** <- the CEILING |
| fully decorrelated | **−0.080 pts** |
| **observed** | **+0.0024** |

**My "+0.13-0.2" prediction was literally the perfectly-correlated ceiling — unreachable in
principle.** And my postmortem overstated the other way: pure decorrelation predicts −0.080,
we measured ~0, so the errors are only PARTIALLY correlated. **The dominant fact was neither
of my stories: it is DILUTION.** 2 of 8 scenes x 12.9% of pixels cannot move a scene-mean score.

**Mask stays in R9, budgeted at ZERO.** It is insurance for the 16/07 data (where the ring
could be far larger), not points.

## Q4: THE k2-REFIT-AND-RETRAIN IS DEAD — and for a reason I had already used against k1

Re-running COLMAP with a k2 term produces a NEW bundle adjustment -> **new poses and new
points**. But `test_poses.csv` is expressed in the EXISTING reconstruction's frame. A refit
DESYNCHRONIZES the test poses from the model. That is precisely the objection I raised against
hand-editing k1 — and I failed to apply it to k2. Meanwhile **the lens field already captures
exactly the residual a k2 term would model**, without touching the reconstruction (+0.73 on the
LB, confirmed). The field is the CORRECT engineering answer to the mis-specified camera; k2 is
the same idea implemented in the one place that breaks the pose contract. **Closed.**

## Verified correct (no change needed)
- **fov_mask guards**: proven exactly right. g(ru)=ru+k1·ru³ is strictly increasing on
  [0, r_fold), so g(r_fold) IS the max on the monotone interval, and rd_max < g(r_fold)
  guarantees a root for EVERY pixel — the assert cannot pass while a subset fails. The branch
  assert is provably redundant (Newton converges monotonically from below for concave g) but
  free; kept.
- **run_dataset.sh hard gate**: closes the silent-1-member path. A worker dying mid-scene
  leaves no `done/` marker -> the gate fires loudly. `conda activate` in a backgrounded
  subshell is safe (each worker mutates only its own env).
- **Reclaim cannot ship >350MB** (final assert is a real backstop). And **cheapest-first is
  provably optimal**: the score is a mean over SCENES, not images, so every scene contributes
  1/N regardless of image count -> gain-per-byte is maximized by the fewest-bytes scene.

## Fixed this round
- **preflight: pose-convention check** — the one test that catches a convention change in a new
  release before it silently corrupts EVERY scene. **My first two attempts were BROKEN and
  produced FALSE FAILURES on known-good data** (treated COLMAP point IDs as array rows; and
  projected through a pure pinhole, so k1=-0.115 read as 12.22px of "convention error" that was
  really just unmodeled distortion). Now applies the SIMPLE_RADIAL forward model and fits the
  obs-scale (images.bin xys are at ORIGINAL 5280x3956 while cameras.bin is the /4 delivered size).
  **Residuals: 0.27px (HCM0249), 0.33px (HNI0131).** NEGATIVE-TESTED — it has teeth:
  R transposed -> **911.9px CAUGHT**; t negated -> **1154.6px CAUGHT**.
- preflight: usable-train-image floor (>=50 — a mostly-empty image dir would otherwise train to
  convergence on 12 views); points3D sanity; resolution-vs-VRAM warning. Dropped the on-disk
  FRACTION warning: the sparse legitimately holds train+test+dropped poses, so "half of
  images.bin is missing on disk" is EXPECTED, not a defect.
- verify_zip: degeneracy BANDED (hard-fail std<1.0, warn 1.0-5.0) — a false positive BLOCKS a
  valid submission, which is worse than a missed one.
- build_submission_zip: reclaim margin now scales with file count (fixed 0.5MB stops covering
  ~136B/entry past ~3700 files).
- run_dataset.sh: release the claim on failure so the other GPU can retry.
- r9.sh: BOUNDED member-wait (12h) — an unbounded wait would hang forever instead of failing.

**OPS LESSON: never edit a RUNNING bash script.** r9.sh was mid-wait when I edited it; bash
reads by file offset, so the edit could have made it execute garbage. Killed and relaunched.

# THE 86 QUESTION — bounded from six directions, and closed

## D11: test-pose proximity distribution (poses only, legal)

D3 reported a MEAN parallax of 11.8 deg and closed photo-reuse. But the score averages PSNR
PER IMAGE, so the mean is the wrong statistic. The DISTRIBUTION looked alarming:
**median nearest-train ANGLE is 1.19 deg; 23.1% of test views are within 0.5 deg.**

## D12: ...but photo-reuse is DEAD, definitively (per-image, not on a mean)

| | |
|---|---|
| paste-nearest-train PSNR | mean **10.22**, max **14.03** |
| our render PSNR | mean **25.01** |
| **test views where the RAW TRAIN PHOTO beats our render** | **0 / 290 (0.0%)** |

Even the best near-duplicate (0.41 deg, baseline 0.029) pastes at 14.03 dB vs our 26.29 dB.
**View-direction angle is NOT parallax** — the TRANSLATION destroys the paste. D11 was a red
herring; D3's conclusion holds and is now proven per-image. Photo-reuse: closed forever.

## D13: ROLLING SHUTTER — dead

Drone photos come from a MOVING platform, and gsplat supports RS natively
(`RollingShutterType.ROLLING_TOP_TO_BOTTOM` + `viewmats_rs`). RS displacement is per-image and
rotates with the flight direction, so the shared lens field CANNOT absorb it -- a perfect
candidate for the +1.23 dB GEOM residual. ChatGPT's consult flagged the same tell.

Signature: flow varying LINEARLY with row, aligned with the drone velocity.

| | |
|---|---|
| row-linear flow slope | mean **0.324 px** across the FULL frame height (RS would be several px) |
| alignment with drone velocity | **mean cos = +0.089** (none) |
| aligned / opposed / orthogonal | **37.5% / 30.0% / 32.5%** -- uniformly RANDOM |

**The residual is not velocity-coupled. RS is closed.**

## The six bounds on 86.12

| hypothesis | verdict |
|---|---|
| lens / intrinsics | **HARVESTED** (+0.73 on the LB) |
| global pose | −0.02 dB |
| per-image pose | oracle ceiling +0.14 pts, and does NOT transfer |
| photo reuse | **0/290** views beat our render |
| rolling shutter | random -- not velocity-coupled |
| **PERFECT registration (cheating with TEST GT)** | **28.6 dB -- STILL SHORT OF 31** |

**86.12 requires >=30 dB even at near-perfect LPIPS/SSIM** (and ~33.7 dB at a realistic
LPIPS .05 / SSIM .93). Our 26.47 dB is **SOTA-typical** 3DGS for outdoor scenes (published
3DGS: ~24-26 dB on Mip-NeRF360 outdoor, ~23-25 on Tanks & Temples). **31-34 dB is above what
any published NVS method achieves on real photos at this view spacing.**

Even outright cheating with the test ground truth does not reach it. Top-1 is not out-tuning
us; they have something structurally different. The full-res originals demonstrably exist
(organizers COLMAPped at 5280x3956, delivered /4). That is the user's hard line and we do not
cross it.

## The ONE lever still untested, and it has a real mechanism

**Our loss has never optimized PSNR.** Standard 3DGS trains `0.8*L1 + 0.2*(1-SSIM)`, and
**L1 optimizes the MEDIAN, not the mean.** The metric-exact loss
(`0.3*(1-SSIM) + 0.02606*ln(MSE)`) is the only remaining idea with a mechanism rather than a
hope. exp31 was killed by MY cost bug (LPIPS every step). Re-run it cheaply: arm B (no LPIPS)
is 5-10x faster and isolates exactly what log-MSE buys. Expected +0.2-0.5, not +7.

## R9 BUILT + VERIFIED (2026-07-15) — 60k/8M members, first full hardened-pipeline run

`sub_round9_ut60k_private.zip` — 8x 60k/8M UT members (replacing 30k/5M), lens field REFIT
from each 60k model's own train renders, FoV mask kept. **345.6MB, VERIFY PASSED** (CRC, every
image's dims, degeneracy screen, exact names, size). Ran fully unattended: warp routing on
HNI0131/HNI0265, stride-1 on HCM1439/HNI0265, all correct.

Field magnitudes (60k vs the 30k fields): the better members register TIGHTER —
HCM1439 0.091 (was 0.132), HNI0437 0.138, HNI0366 0.153; the negative-k1 pair 4.99/5.01
(was 5.69/5.75). The residual shrank, as expected from a better fit.

**Repriced expectation: +0.25-0.35 -> 78.65-78.75** (NOT the earlier 78.9-79.2 — that ignored
the measured 0.22 in-ensemble compression ratio: a member's +0.69 single-model gain becomes
~+0.15 in the ensemble, x the 0.30/0.167 weight ratio). Submit and confirm.

# AUDIT ROUND 12 — THE CAVE IS NOT SEALED. My D5 bound was over-extended.

## The flaw the audit found in my own reasoning

My "86 is unreachable" rested on D5: perfect registration (oracle dense flow vs test GT) ->
only 28.6 dB. **But dense flow only REPROJECTS the existing render's pixels — it moves blur
around, it cannot sharpen a soft cable, fix a wrong 3D SURFACE, or fill disocclusion.** So
28.6 dB bounds REGISTRATION of THIS under-fit render. It is NOT a ceiling on MODEL QUALITY.
The registration cave is sealed; the model-quality cave was never tested. Five of my six
bounds are sound; the sixth was mis-interpreted.

## The thread: 27 dB TRAIN fit is LOW

Vanilla 3DGS reaches 32-35 dB train on comparable data; we sit at ~27. That 5-8 dB is model
headroom the flow oracle cannot see. And because test poses are densely interleaved with train
(~1 sequence-step away, round-3), a rising train fit should TRANSFER rather than overfit.

## exp31b (warm-start metric-loss refit) — idea 3 CLOSED, but it cannot test capacity

| arm | TRAIN PSNR | TEST PSNR | Score(vgg) |
|---|---|---|---|
| exp29 baseline | ~27.09 | 24.4646 | 75.899 |
| arm B: metric loss, no LPIPS, reg off | **27.61** | 24.4641 | 74.96 |
| arm A: metric loss, tail LPIPS | 27.06 | 24.4314 | 75.63 |

The metric loss did NOT beat the standard loss (score dropped) — **idea 3 is closed.** Pure
metric pushed train only +0.5 dB and test moved 0.0. BUT this was a warm-start refit with
`refine_stop 0` — it CANNOT add or replace gaussians, so it cannot distinguish "loss is fine,
model under-fit for capacity/placement" from "content can't be fit." Inconclusive on the real
question.

## fidfit.sh — THE DECISIVE EXPERIMENT (running, both GPUs, ~3h/arm)

From-scratch, densification ON, three arms isolating the cause of the 27 dB ceiling:
- **fid_ctrl**: standard recipe (reproduces ~27)
- **fid_l2reg**: pure-L2 loss, reg ON (does the LOSS cap fit?)
- **fid_l2off**: pure-L2 loss, opacity/scale reg 0, noise_stop early (does REG cap it? — round-6
  showed opacity_reg drains thin structure)
Added `--pure_l2` to train_gsplat (loss = MSE only). TRAIN PSNR is the readout.

**VERDICT FRAMEWORK (audit r12, adopted):**
- any arm reaches TRAIN >= 32 dB -> model was UNDER-FIT, registration was never the wall,
  there is a real multi-dB path -> spend GPU on it.
- all arms cap <= 29 dB -> the scene CONTENT is the wall (wind-swayed cables, inter-shot
  exposure, non-repeatable pixels) -> 86 IS sealed -> switch to rank-maximization.
**Do not declare the impasse until this number lands.**

## Preflight audit (A): v3 pose-convention check CONFIRMED sound, two narrow gaps
- obs-scale fit is ROBUST (correct scale 0.26px vs wrong-scale 444-2663px; cannot mask a
  convention error — real errors blow up at EVERY candidate scale). Negative tests all caught
  (R-transpose 912, t-negate 1155, y-flip 503, xy-swap 538 px).
- GAP 1: a sub-5px principal-point convention shift only warns/passes (cx+Npx -> Npx residual),
  but it is exactly what the LENS FIELD absorbs, so low-risk. Suggested: check the median
  residual VECTOR (coherent 2px bias = convention flip; scattered 2px = noise).
- GAP 2: applies only k1, so a real k2 on a wider-FoV new lens could false-fail (v2's bug one
  order up). Fix: apply k2 in the reprojection for RADIAL.
- All four round-11 fixes confirmed implemented as specified.

# CONSULT #2 (Vietnamese) + THE REFRAME — 2026-07-15

Symmetric errors: I declared a WALL on the PSNR axis; the consult declared OPEN ROAD on the
perceptual axis. Neither measured reachability. Two cheap measurements settle it.

## Verified: perceptual axes hold the score-efficient headroom (consult right on arithmetic)
+2.35 (SSIM .8816->.96) + 2.73 (LPIPS .0983->.03) = +5.1 pts to reach 83.5 with PSNR flat.
The absolute floor for 86.12 at PERFECT L=0,S=1 is PSNR 26.87 (already in our log), not 30 —
I mislabeled "near-perfect" as 30 in the brief. BUT reachability of SSIM .96/LPIPS .03 is
UNPROVEN (6 lens-field-sized SSIM wins; 3.3x LPIPS cut on our best metric) — the consult
ASSUMED it, the same error it accused me of.

## Measured: registration oracle FULL SCORE (consult right it's not "dead", WRONG on size)
Perfect per-image registration (dense flow vs test GT, CHEAT), scored with the full metric on
HCM0181 single 60k:

| | PSNR | SSIM | LPIPS | Score(vgg) |
|---|---|---|---|---|
| baseline single | 24.46 | 0.8526 | 0.1089 | 75.90 |
| oracle-registered | 26.77 | 0.8964 | 0.1040 | **78.79 (+2.89)** |

Registration IS a score lever (+2.89, SSIM-heavy) — my dB-only reading ("+2.13dB, dead") hid
it. BUT the consult GUESSED the oracle at 83.7 (assuming SSIM .94/LPIPS .05); ACTUAL SSIM .896
/ LPIPS .104 -> 78.79. The consult overestimated by ~5 pts. Of the +2.89, the train-fitted
field already legitimately captured ~+0.88 (public); the remaining ~+2.0 is per-image RANDOM
registration that does NOT transfer (test poses carry the same noise).

## THE REAL FINDING: train fit is scene-specific, NOT capped at 27 — a CAPACITY signal

Per-scene TRAIN PSNR of the current 60k/8M members (train photos = legal GT):

| scene | k1 | n_img | TRAIN psnr | ssim |
|---|---|---|---|---|
| HCM1439 | +0.008 | 103 | **32.42** | .9475 |
| HNI0437 | +0.014 | 112 | 30.27 | .9366 |
| HNI0366 | +0.012 | 120* | 29.56 | .9294 |
| HNI0131 | **-0.115** | 120* | **27.17** | .9071 |
| HCM0276 | +0.008 | 120* | 27.12 | .9184 |
| HCM0254 | +0.010 | 120* | 27.03 | .9115 |
| HCM0249 | +0.009 | 120* | 26.09 | .9049 |
| HNI0265 | **-0.115** | 103* | **25.91** | .8689 |

(* = stride-subset of 240/205). **The method fits train to 32.4 (HCM1439) — it is NOT capped
at 27.** Low-fitting scenes are the DENSE (240-img) ones: 8M gaussians spread thinner across
more views. This KILLS "content is the wall" and points at CAPACITY (cap_max).

Consult's train-time-lens-misfit hypothesis WEAKENED: HNI0131 (k1=-0.115) fits train FINE
(27.17, mid-pack); only HNI0265 drags. Not a clean per-k effect.

## fidfit (from-scratch, isolating loss/reg) — running
| arm | TRAIN | TEST | gap |
|---|---|---|---|
| fid_ctrl (standard) | 26.99 | 24.45 | 2.54 |
| fid_l2reg (pure-L2, reg on) | **24.14** | 23.32 | 0.82 |
| fid_l2off (pure-L2, reg off) | running | | |

Pure-L2 made train fit WORSE (MSE gradient vanishes near the target; L1's constant-magnitude
gradient keeps pushing detail) -> the LOSS is not what caps fit. Idea 3 doubly closed.

## cap16.sh — THE decisive lever now (running GPU1)
cap_max 16M on HCM0181 (dense, train 26.99 @ 8M). Readout: does train rise AND transfer to test?
- train up + test up -> capacity is a real TRANSFERABLE lever, push it on dense scenes (multi-dB)
- train up, test flat -> generalization wall
- train flat -> not capacity

## Net: 86 still not legit (the +9.2 jump), but there's likely +2-4 legit on capacity + the
## perceptual axis (score-efficient). Target: low-to-mid 80s, maximize rank. Not 86.

## fidfit COMPLETE — loss and reg ELIMINATED; capacity is the last hypothesis standing

| arm | TRAIN | TEST | gap |
|---|---|---|---|
| fid_ctrl (standard 0.8 L1 + 0.2 SSIM) | **26.99** | 24.45 | 2.54 |
| fid_l2reg (pure-L2, reg on) | 24.14 | 23.32 | 0.82 |
| fid_l2off (pure-L2, reg off, noise early) | 25.13 | 23.62 | 1.51 |

**All three <= standard's 26.99 train.** Pure-L2 made fit WORSE (MSE grad vanishes near target).
Reg-off also worse. So on a DENSE (240-img) scene at 8M, ~27 dB train is the ceiling regardless
of loss/reg. Loss ELIMINATED (idea 3, third confirmation). Reg ELIMINATED.

By elimination the dense-scene 27-cap is CAPACITY or content. Evidence for capacity: HCM1439
(103 views) fits to 32.4 @ the SAME 8M -> capacity-per-view is the binding constraint.
cap16.sh (16M on HCM0181 dense) is the decisive test: break 27 + transfer = real multi-dB lever.

## TRANSFER TEST (audit r13 Q3, run myself while audit was rate-limited)

The capacity confound, examined: among the 240-image scenes (SAME capacity), per-scene train
PSNR ranges 26.1-29.6 — a 3.5 dB spread. So SCENE DIFFICULTY, not view count, drives train fit;
my "capacity-per-view" inference from HCM1439's 32.4 was CONFOUNDED (fewer views AND easier scene).

But the question that matters is transfer. Per-scene public train vs test PSNR (gsplatB9ut,
recipe fixed, scene the only variable):

| scene | TRAIN | TEST | gap |
|---|---|---|---|
| HCM0181 | 26.04 | 24.27 | 1.77 |
| HCM0193 | 27.56 | 25.68 | 1.88 |
| HCM0204 | 27.13 | 24.89 | 2.24 |
| hcm0031 | 27.37 | 25.09 | 2.29 |
| hcm0034 | 27.14 | 25.14 | 1.99 |

**r = +0.928, slope d(test)/d(train) = +0.80, gap 2.03 +- 0.20 (nearly CONSTANT).**

Higher train fit -> higher test, ~0.8 dB per dB, constant gap. STRONG (but correlational —
scene difficulty drives both) evidence that raising train fidelity transfers. This makes cap16
the clean INTERVENTIONAL test (within-scene, difficulty confound removed): if 16M raises
HCM0181 train, the constant gap predicts test rises ~0.8x. Pair closes the loop:
transfer.py = "train fit transfers"; cap16 = "capacity raises train fit".

## STOP ORDER + cap16 CLOSEOUT (16/07)

User ordered all training stopped ("ay bro, stop training right now, everything"). Verified:
no train/render processes alive, both GPUs idle (0%, 343/41 MiB residual). cap16 had already
died on its own: TRAIN FAILED (OOM) at ~11k/60k after 13h. Capacity lever CLOSED on evidence:
5M->8M gave +0.085 dB, so 16M was predictably near-zero even before the OOM; 16M also
impractical (>3 days/scene at observed speed). No new training until user go-ahead.

## ROUND-2 DATASET EDA (private_set2, landed 16/07)

Source: VAI_NVS_DATA_ROUND2.zip (1.27 GB) -> /mnt/d/avv/data/phase1/private_set2. 7 scenes.

| scene | train | test | model | k1 | WxH | notes |
|---|---|---|---|---|---|---|
| HCM0421 | 240 | 60 | SIMPLE_RADIAL | +0.0089 | 1320x989 | urban tower, overcast + horizon views |
| HCM0539 | 240 | 60 | SIMPLE_RADIAL | +0.0081 | 1320x989 | urban tower |
| HCM0540 | 240 | 60 | SIMPLE_RADIAL | +0.0089 | 1320x989 | urban tower |
| HCM0644 | 240 | 60 | SIMPLE_RADIAL | +0.0090 | 1320x989 | urban tower, near-nadir heavy |
| HCM0674 | 240 | 60 | SIMPLE_RADIAL | +0.0088 | 1320x989 | urban tower |
| bonsai | 248 | 28 | SIMPLE_PINHOLE | 0 | 1920x1080 | INDOOR video: bonsai on GLOSSY BLACK GLASS table (mirror reflections), scale 1/1 |
| chair | 205 | 58 | SIMPLE_PINHOLE | 0 | 720x1280 PORTRAIT | INDOOR phone video: office chair + tissue box, DoF/motion blur, scale 1/1.5 |

KEY FINDINGS (all verified by direct measurement):
1. PREFLIGHT PASSES ALL 7 (after one fix: obs-scale candidates now include fractional 1.5 —
   chair COLMAP ran at 1080x1920, delivered at 720x1280; median reproj 0.63 px at scale 1.5;
   was a preflight limitation, NOT a data problem. set1 regression still passes).
2. NO negative-k1 scenes this round — all towers +k1 ~ +0.009, videos pinhole. Native render
   path everywhere; NO warp-path scenes; lens-field fitting still applies (field exists on +k
   scenes too, ~+0.1-0.2 expected, fit on train as usual).
3. Train images now delivered at CAMERA resolution (1320x989); set1 shipped 5280x3956 (4x).
   Same COLMAP scale story (README: towers 1/4, chair 1/1.5, bonsai 1/1).
4. images.bin contains TEST FRAMES (poses bit-exact = test_poses.csv) AND their full
   triangulated observations (~2.5k keypoints/img linked to points3D) + 11-98 extra
   undelivered frames per tower. SET1 WAS THE SAME (verified HCM0249/0254) — not new.
   COMPLIANCE GRAY ZONE flagged: test-frame keypoints are measurements derived from test GT
   pixels; using them for per-test-image registration could be read as "inferring test GT"
   (Rule 10). NOT ACTING without user/organizer ruling. Registration oracle valued the full
   per-image cave at +2.89 pts (mostly non-rigid; keypoints would capture the low-order part).
5. Test-pose geometry: towers same difficulty class as set1 (medNN 0.10-0.16 normalized,
   medAng 8-11deg, maxAng ~32deg). Videos are interleaved holdouts (every ~10th frame bonsai /
   ~5th chair is test; 3-4deg from nearest train view) — registration easy, content hard.
6. Photometrics: towers stable (lum drift 15-35 levels), sharp (VoL ~4000). bonsai: drift 63
   levels (auto-exposure video) + blurry frames (VoL med 474, p5 126). chair: drift 49,
   VoL p5 331. Exposure drift is SMOOTH in time (neighbor interp err ~1 level median) ->
   per-test-frame appearance INTERPOLATION from temporal train neighbors is viable and legal
   (uses frame index only, no GT pixels).
7. points3D: towers 154-219k; bonsai 54k (low — glass table kills SIFT); chair 80k.
8. Pipeline compatibility: train_gsplat already accepts SIMPLE_PINHOLE (k1=0) + per-frame
   AE soak; render_gsplat takes W/H from csv (portrait OK). Submission = 386 imgs
   (300 tower 1320x989 + 28 bonsai 1920x1080 + 58 chair 720x1280) ~ est 315 MB PNG, under
   the 350 MB cap if that cap carries over; ladder handles it regardless.
OPEN QUESTIONS for user/organizer: (a) do round-2 submissions cover ONLY these 7 scenes or
set1+set2 = 15? (b) ruling on test-frame keypoints in images.bin (finding 4).
NEXT (awaiting go-ahead, training stopped per user order): tier-2 run_dataset.sh on set2
(8 scenes worth of GPU-days; towers standard recipe; videos same recipe + eval exposure choice).

## RECHECK: test-keypoint gray zone QUANTIFIED (user request, 16/07 evening)

Question: images.bin ships test-frame SIFT keypoints (~2.5k/img, real measurements — residual
scatter identical to train frames, median 0.28px, NOT synthetic projections; ~50% of points3D
have test frames in their tracks). How much would exploiting them actually be worth?

Method (public HCM0181, B11ut60k renders, GT used ONLY for scoring — experiment itself legal):
per-test-image warp fit from keypoint residuals (obs/4 - SIMPLE_RADIAL projection), three
complexities, vs our legal paths. scorePart = 0.3*PSNR/50*100 + 0.3*SSIM*100 (LPIPS excluded).

| arm | PSNR | SSIM | delta scorePart |
|---|---|---|---|
| raw render | 24.479 | 0.8547 | 0 |
| per-image TRANSLATION from test KPs | 24.470 | 0.8545 | -0.010 |
| train-KP global field (LEGAL) | 24.907 | 0.8651 | +0.571 |
| test-KP PER-IMAGE field (gray zone) | 24.914 | 0.8658 | +0.596 |
| ship DIS field (LEGAL, R7/R9 path) | 25.360 | 0.8755 | +1.155 |
| DIS + test-KP per-image residual | 25.392 | 0.8765 | +1.203 |

Controls: zero-net-shift double resample = -0.008 (PSNR +0.07 but SSIM -0.002 — resampling
blur inflates PSNR, score-neutral); Gauss 0.5 blur = -0.38. Early runs with sigma=3-cell
smoothing had CRUSHED the field (0.15px -> 0.05px) — redone at sigma=0.75/1.5.

FINDINGS:
1. Per-image translation is DEAD: median per-test-image KP shift 0.018px — BA poses are
   rigidly near-perfect (consistent with d11's +0.14 pose ceiling).
2. Our LEGAL DIS field already captures ~2x what the full test-KP exploit gives
   (+1.155 vs +0.596). DIS and train-KP fields agree in direction (corr 0.77-0.80) but DIS
   is 2.3x larger (0.268 vs 0.118px mean) — DIS also sees RENDER-side systematics that
   COLMAP geometry cannot. Independent cross-validation of the R7 field method.
3. Gray-zone increment ON TOP of ship path: +0.048 pts on this scene. Even with tuning,
   bounded well under ~0.2; the rest of the +2.89 registration cave is non-rigid/photometric
   detail that 0.3px-noisy sparse keypoints cannot resolve (cell-averaging kills it).
VERDICT: gray zone DEFUSED — exploit value negligible vs compliance risk; not using it. Also
means competitors can't be mining much from it either.
BONUS: train-KP global field (+0.571, zero renders needed) = independent legal confirmation
of the lens-field lever; could serve as a fallback/cross-check on scenes where DIS fit fails.
NOTE (user): round-2 submission covers PRIVATE_SET2 ONLY (7 scenes).

## AUDIT ROUND 13 (16/07 19:20) — full report received; one finding REBUTTED, two adopted

1. "leg-1 noise_stop=-1 is not R8-generation, costs 0.05-0.15" — REBUTTED BY OUR OWN A/B:
   exp20 (noise_stop 25k) = 74.5067 vs exp19 (noise through LPIPS) = 74.5072, EXACT TIE
   (log line 152: "noise-through-lpips was a non-factor"). Leg 1 continues unchanged.
2. CONFIRMED (adopted, post-run): run_dataset.sh k1 probe FAILED in the live run (conda run
   g++ env error) and silently coerced to native via awk string->0. Correct on set2 only by
   coincidence (all +k1/pinhole). FIX QUEUED for after both legs (script is running; never
   edit a running bash script): validate probe output numeric + hard-fail, or consume
   preflight's authoritative path. Also adopting the train_args stamp (write $TRAIN_ARGS to
   $M/train_args.txt; assert match when skipping via existing ckpt).
3. Gray-zone recheck: audit CONFIRMS methodology (controls fair, no leak, verdict stands;
   information ceiling argument: 2.5k anchors @0.28px noise resolve only ~21% of the cave
   regardless of interpolator; RBF could 2-3x the +0.048 residual, still <=0.15). Closed.
4. Preflight fractional-scale fix: CONFIRMED safe (convention failures are 500-1000px at
   EVERY scale; scale scan can only absorb isotropic magnification, its purpose).
5. WATCH ITEMS pre-registered for bonsai/chair (first non-drone scenes through pipeline):
   (i) bonsai leg-2 VRAM (1.59x pixels x 8M — nearest thing to cap16 OOM in this run; if
   OOM, drop bonsai cap rather than lose the scene); (ii) chair portrait W<H plumbing;
   (iii) bonsai FIELD MAGNITUDE vs tower range 0.1-0.3px — DIS pairs are exposure-
   contaminated (63/255 drift); a bonsai field >> towers = exposure bleeding into flow, not
   lens signal -> refit with per-pair mean normalization and rebuild zip BEFORE submitting
   (zips are not auto-submitted; natural gate). NOT editing fit_field mid-run — proven
   method stays frozen for the legs; averaging over ~124 frames partially cancels
   exposure-sign noise.
6. Video appearance lever endorsed by audit (drift 25% vs towers' 2-5% where appearance
   measured null — different regime; model IN training, interpolate per-frame gain at test
   from temporal neighbors). Queued as first post-leg experiment, pending user go.
7. Transfer-test epistemics (audit): constant gap carries NO causal information (consistent
   with both difficulty and transfer models); slope 0.8 = cross-scene attenuation, not a
   within-scene law. Correctly treated as correlational. Capacity thread stays closed.

## LEG 1 COMPLETE (17/07 01:08) — sub_round10_set2_ens30k5M.zip BUILT+VERIFIED, 341.5MB
## + VIDEO-SCENE FIELD GATE EXECUTED + BONSAI TRAINING COLLAPSE FOUND

Leg 1 (R8-style 30k/5M, 2 seeds, tier 2) ran 6h05 for all 7 scenes (vs 11h est). Zip verified.

FIELD GATE (audit r13 item 6, held-out train cross-validation — fit field on even train
pairs, measure warp effect on odd pairs; the field's job is exactly this transfer):
| scene | field mean|d| | fold-corr | held-out dPSNR | dSSIM | verdict |
|---|---|---|---|---|---|
| HCM0421 (control) | 0.211px | +0.903 | +0.442 | +0.0074 | KEEP (lever confirmed on set2) |
| bonsai | 1.03px (DIS raw) | +0.498 | +0.003 | +0.0001 | ZERO — junk |
| bonsai (photonorm refit) | 1.12px | — | — | — | not exposure: normalization didn't collapse it |
| bonsai (train-KP field) | 0.35px | — | +0.0006 | +0.0001 | ZERO — even clean geometry field earns nothing |
| chair | 0.26px | +0.432 | **-0.127** | -0.0036 | ZERO — actively harmful (DoF/RS flow noise) |
bonsai DIS field also DIRECTIONALLY UNCORRELATED with KP field (corr -0.07/+0.17 vs +0.8 on
towers) -> glass-table reflections move with viewpoint; DIS tracks them; mean = fake field.
Fields for bonsai+chair ZEROED in r2r8/fields (atomic swap before compose; raw backed up as
*_dis_raw.npy; reasons in .meta.json). Towers keep fields. VIDEO-SCENE RULE going forward:
never ship a field that fails held-out train x-val.

BONSAI TRAINING COLLAPSE (the big finding):
- test renders = fog; TRAIN renders = fog too. CSV poses verified bit-exact vs BA (not poses).
- trajectory: [5000] l1 0.038 @ 489k GS (HEALTHY) -> cap 5M reached ~10k -> [15000] l1 0.26
  (worse than init), thrashing 400k+ relocations/100 steps, never recovers.
- ckpt autopsy: median opacity 0.000 (!), median scale 0.0003 — 5M near-transparent
  micro-gaussians. chair same recipe: opacity 0.069, l1 0.0297, renders EXCELLENT (sharp,
  portrait correct). scene_scale NOT the separator (bonsai 8.4 between chair 5.8, towers ~10).
- hypothesis: MCMC cap+noise+relocation dynamics interact fatally with mirror-table content
  (reflections = view-inconsistent supervision) + 63/255 exposure drift; converging model at
  5k destroyed by forced growth to 5M. UNTESTED which factor is necessary — diagnostics
  queued: (D1) cap 1M + noise_stop 10k + refine_stop 12k; (D2) cap 5M + same early stops
  (isolates cap vs noise); readout = train fit @15-30k + visual.
- CONTAINMENT: bonsai pre-claimed in r2r9/claims -> leg 2 skips it; towers+chair proceed.
  When leg-2 GPUs free (~18h): bonsai diagnostics, then fixed-recipe bonsai into r2r9, then
  MANUAL compose for leg 2 (gives control of the field gate window; pipeline's own compose
  will hard-gate abort on bonsai(no-done) as designed).
- Leg-1 zip STATUS: verified, towers+chair good (tower spot-check: sharp), bonsai = 28 fog
  images (drags scene mean an est. 4-6 pts). RECOMMENDATION: hold submission until a fixed
  bonsai can be spliced in (rebuild as sub_round10b), unless an early LB datapoint is wanted.

## ROUND10 GRADED: 72.48450 (PSNR 23.9349 SSIM 82.9977 LPIPS 16.9394) — bonsai fog confirmed as the drag
Context: R9 on set1 was 78.579. Set2 round10 at 72.48 with 6 good scenes + 1 fog scene (bonsai).
Crude bonsai isolation: if the 6 good scenes ~ set1-class (~mid/high 70s each) and mean=72.48
over 7, bonsai is pulling ~30-40 range -> ~4-6 pt scene-mean drag as predicted. CANNOT drop it:
user confirms system won't grade a submission missing any test_pose. So bonsai MUST be fixed.

## PIVOT: per-scene EVAL-SPLIT MODEL SELECTION (user directive 17/07)
Method: split each scene's TRAIN into train-sub/eval, pick recipe by eval score, retrain on
FULL train for the test render. Eval must mimic hidden test.
KEY STRUCTURE FINDING: round-2 video test sets are SINGLE HELD-OUT GRID POINTS. bonsai train =
video idx 0,10,20..2750 (stride 10); all 28 test frames are isolated holes, each flanked by
train one stride away (28/28 interleaved, pose gap 0.054/4.3deg). chair stride 5, 58/58 holes.
=> honest eval = punch ISOLATED holes in the train grid (not random: adjacent removal=2-stride
holes harder than test; near-duplicate=trivial). scripts/make_eval_split.py does this:
bonsai 248->220 sub + 28 eval holes (spacing>=70); chair 205->147 + 58 (spacing>=10). Legal:
eval GT = train photos, eval poses from images.bin, no test pixel touched.
scripts/eval_score.py = exact competition metric (0.4(1-LPIPSvgg)+0.3SSIM+0.3PSNR/50)*100.

DECISION: killed leg 2 after banking its 2 in-flight seed-42 tower ckpts (HCM0421/HCM0539 @60k;
were 70% done — waited ~1h to save them; re-run leg 2 later will skip them). bonsai is the
critical path (blocks every submission), so both GPUs pivot to bonsai_select.sh (4 cap/churn
recipes) the moment the ckpts bank. Then round10b = leg-1 towers/chair + fixed bonsai.

## LEG 2 STOPPED (17/07 03:10) — 2 seed-42 towers BANKED, bonsai selection LAUNCHED
Stop was messy: the classifier that vets Bash was down ~15 min, so the first pkills killed
train pythons but not the run_dataset.sh worker LOOPS -> workers re-claimed the next scenes
(HCM0540 FAILED, then HCM0674/chair started fresh). Once classifier returned: killed the
worker bashes (66236/66651/66652) FIRST, then the orphan trains -> GPUs clean idle.
Lesson: to stop run_dataset.sh, kill the worker bash loops BEFORE the trains, else they respawn.
BANKED for the eventual leg-2 re-run (skips these, ckpt.pt exists): HCM0421_ut42, HCM0539_ut42
(60k/8M, 1.9GB each). Everything else in r2r9 is partial/failed and will be redone.

bonsai_select.sh LAUNCHED on both GPUs: capA_1M, capB_2M, capC_500k, capD_5Mearly, each
trained on the 220 train-sub and scored on the 28 eval holes (scripts/eval_score.py; lpips-vgg
+ fused_ssim validated working). Winner retrains on full 248 for the test render.

## CAREFUL EVAL SPLIT — one rule for both regimes (17/07, user emphasis: "carefully pick rather than random")
Goal: eval score must PREDICT test. Since we're GIVEN test poses (not pixels), "careful" =
make eval's difficulty distribution match the real test's. Validated on public HCM0181 (drone,
we own test GT): test->train gap med 0.116 / p75 0.168 / ang 7.74.
| eval strategy | gap med | p75 | ang | verdict |
|---|---|---|---|---|
| isolated every-k by capture order | 0.121 | 0.176 | 9.56 | BEST — matches test median AND tail |
| random | 0.118 | 0.237 | 9.38 | median ok, tail too fat (adjacent/clustered removals) |
| contiguous arcs | 0.139 | 0.210 | 12.79 | overshoots (too hard) |
| every-k + widen | 0.141 | 0.264 | 10.19 | overshoots |
UNIFIED RULE: eval = ISOLATED, evenly-spaced frames along the capture sequence (never adjacent).
video -> grid-holes (stride-matched); drone -> every-k by DJI filename/capture order. Works because
the hidden test IS an interleaved holdout of the same capture. make_eval_split.py now does both
(video branch unchanged; drone branch = ordinal every-k, min_sep 2). Drives per-scene model
selection; public scenes (drone) double as the eval->test calibration anchor.

## BONSAI FIXED — eval-hole selection working (17/07)
First recipe in: capC_500k EVAL n=28  PSNR 26.6273  SSIM 0.8364  LPIPSvgg 0.2874  SCORE 69.5706
vs the fog collapse (~30s scene). Visual (bonsai_capC_eval.jpg): recognizable plant/table/couch/
carpet, soft on glass reflections + edges (the LPIPS 0.287). LOW CAP is the fix; the 5M/8M cap +
MCMC churn was the collapse. Awaiting capA_1M / capB_2M / capD_5Mearly to pick the eval winner.
Confirms the eval loop produces a usable, discriminative signal on a real scene.

## BONSAI EVAL SELECTION — COMPLETE (17/07). Winner: capD (5M cap, CHURN stopped early)
| recipe | cap | refine_stop | noise_stop | EVAL PSNR | SSIM | LPIPSvgg | SCORE |
|---|---|---|---|---|---|---|---|
| capC_500k | 0.5M | 15k | 15k | 26.627 | 0.8364 | 0.2874 | 69.571 |
| capA_1M | 1M | 15k | 15k | 26.714 | 0.8419 | 0.2747 | 70.298 |
| capB_2M | 2M | 20k | 20k | 26.459 | 0.8490 | 0.2672 | 70.660 |
| **capD_5Mearly** | **5M** | **15k** | **8k** | **26.802** | 0.8485 | **0.2595** | **71.156** |
MECHANISM RESOLVED: the collapse was the MCMC CHURN (noise injection + relocation to 50k),
NOT the cap. capD keeps the full 5M cap but stops noise at 8k / refine at 15k -> best on PSNR
AND LPIPS. The original pipeline's noise_stop 50000 (through the whole run) was the killer on
this view-inconsistent glossy-glass scene. (Fog run: opacity 0.000 = noise never let it settle.)
=> winning recipe = --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 (30k iters).
bonsai_final.sh retraining this on full 248 frames, 2 seeds (GPU0 s42, GPU1 s7), no field
(bonsai field is junk), then splices leg-1 towers/chair -> sub_round10b_bonsaifix.zip.
Bug found+fixed en route: `local G=$1 SD=$2 M=$OUT/ut$SD` — $SD expands before assignment
under set -u (split to two `local` lines).

## ROUND10b BUILT + VERIFIED + GATED (17/07 08:51) — READY TO SUBMIT
sub_round10b_bonsaifix.zip: 386/386 files, 347.4 MB, VERIFY PASSED (CRC, exact names, dims per
image: towers 1320x989, bonsai 1920x1080, chair 720x1280). = leg-1 towers/chair (30k/5M, fields
kept on towers / zeroed on chair) + FIXED bonsai (capD: 5M cap, refine_stop 15k, noise_stop 8k,
2 seeds, no field). Gate: bonsai TEST renders visually confirmed (round10b_bonsai_test.jpg) —
plant/pot/glass-reflections/couch/carpet all clean across 4 viewpoints, no fog. Est ~75-76.

## ROUND11 tower+chair training LAUNCHED (both GPUs) — run_dataset.sh --scenes (no bonsai)
Fixed 2 run_dataset.sh bugs first (not running at the time): (a) export NVCC_PREPEND/APPEND
under set -u (conda nvcc activate.d aborts otherwise); (b) `rm -f claims/*` -> `rm -rf` (claims
are DIRs from mkdir; -f left stale leg-2 claims that would wrong-skip scenes). Cleaned stale
r2r9/claims + done manually. Now training HCM0421/0539/0540/0644/0674 + chair at 60k/8M (2 seeds;
HCM0421/0539 seed42 banked -> skipped). bonsai EXCLUDED (would re-collapse on default late churn);
gets capD scaled to 60k separately + manual field-gate + compose for round11.

## ROUND10b GRADED: 76.10509 (PSNR 26.039001 SSIM 85.6419 LPIPS 13.0272) — +3.62 over round10
Score formula reproduces round10 exactly (72.48447 vs 72.48450) -> computation trusted.
Only bonsai changed r10->r10b, so bonsai's implied TEST jump = 7*3.6206 = +25.3 pts (fog~45 ->
fixed~70). EVAL-SPLIT METHOD VALIDATED END-TO-END: capD eval 71.16 -> submission delivered the
predicted jump. Gap to top-1 (80.56) now 4.45 (was 8.08). Decomposition: 6 good scenes ~77,
bonsai ~70 (still weakest -> still the biggest lever; its LPIPS is the axis to attack next).
LB set2: round10 72.48450 -> round10b 76.10509.

## ROUND11 BUILT + VERIFIED + GATED (17/07) — READY TO SUBMIT
sub_round11_set2_ens60k8M.zip: 386/386, 349.1 MB, VERIFY PASSED. = 60k/8M 2-seed towers+chair
(HCM0421/0539 seed42 banked from killed leg2) + round10b's capD bonsai (unchanged — validated,
no reason to retrain). Field x-val gate (held-out train pairs): towers KEEP (+0.11..+0.44 dB),
chair ZERO (-0.18, junk again). All 6 towers/chair ckpts 1.9GB/8M, trained full 60k, no collapse.
round11 isolates the 30k/5M -> 60k/8M tower+chair upgrade over round10b (bonsai identical).
Fixed 2 run_dataset.sh bugs to get here (NVCC env under set -u; rm -rf claims dirs).
Est: round10b 76.11 + tower/chair 60k/8M upgrade -> ~76.5-77.

## BONSAI PERCEPTUAL PASS launched (both GPUs) — next lever toward 80.56
bonsai still weakest (~70) with LPIPS 0.260 = most headroom. 4 candidates on the 28 eval holes,
all keeping capD churn-fix, varying LPIPS phase/iters: pA_60k (60k,lpips@40k), pB_lpw2
(lambda_lpips 0.2,lpips@20k), pC_lpearly (lpips@12k), pD_60k_lpw2 (60k,0.2,lpips@30k). Baseline
capD 30k = 71.16. Winner -> full-train bonsai upgrade for round12.

## AUDIT ROUND 14 (code errors) + STRATEGY AUDIT (game-changers) — both landed 17/07
User directive: every audit cycle now = code-error audit (resume auditor) + game-changer audit
(fresh agent). Results:

CODE AUDIT — 3 CONFIRMED bugs, all fixed same-day, none invalidate shipped work:
1. eval_score.py used fused_ssim but score_submission uses repo utils.loss_utils.ssim (delta up
   to ~0.003 SSIM / 0.09 pts — ranking-safe, comparability-unsafe). FIXED -> repo_ssim; capD
   rescored 71.1564 (was 71.1563, negligible here) so perceptual-pass comparisons stay valid.
2. make_eval_split.py could silently under-fill n_eval (min_sep rejections; guard only checked
   len(cand)>=n_eval, real need ~2x). Dormant on all current scenes (verified margins). FIXED:
   assert len(pick)==n_eval.
3. bonsai_select.sh final grep reads eval_score.txt files that never exist (dead summary line;
   harmless). Noted, not fixed (script retired).
Also CONFIRMED clean: names-partition disjointness, isolation guarantee (adjacency impossible),
intrinsics/pose parsing, Rule-10, bonsai_final fix, rm -rf scoping. Field x-val gate: upward-
biased on video (95% content overlap) but ALL decisions on the safe side (bias only strengthens
the rejections; tower keeps corroborated by D9 transfer). Round11 label risk -> PROVENANCE
sidecars written for both zips (bonsai is 30k capD deliberately; upgrading it = fog regression).
Task #22 CLOSED: k1 probe = plain python + numeric hard-fail (no conda run); train_args stamp
on train + mismatch hard-fail on reuse; banked models retro-stamped.

STRATEGY AUDIT (fresh agent, read log cold) — ranked, honest:
1. SUBMIT round11 (+0.4..0.9, zero cost) — built, strictly dominates round10b.
2. Bonsai perceptual pass -> round12 (+0.3..0.6) — running.
3. GAME-CHANGER CANDIDATE (reopen): classical-flow FRAME-INTERPOLATION members for video
   scenes, per-pixel ensembled + eval-gated. IBR died on set1 at 11.8 deg parallax; video
   holdouts are 3-4 deg = different regime. Chair better bet than bonsai (glass reflections
   ghost). RIFE/FILM would need user OK (Rule 10); classical DIS first.
4. Per-frame exposure on video scenes (--ppisp narrow scalar variant, 25% drift vs 2-5% where
   it nulled) (+0.1..0.4, eval-gated).
5. FastGS gates tier-3 on 6 tower/chair scenes (+0.3..0.5, ~2 GPU-days) — newly viable on set2
   (no negative-k1 ring), set1-proven family weighting.
6. Eval loop on towers+chair (+0.1..0.3); FIRST get per-scene eval scores for all 6 (is chair
   a hidden laggard? its field was zeroed, has DoF blur).
Verdict: no single >1pt method exists; the >1pt prize is bonsai(+chair) closing to tower level
(+1.0..+2.0 mean) via stacked video levers. Realistic: 78-79 solid, 80 stretch. Top-1 80.56.

## ROUND11 GRADED: 76.60965 (PSNR 26.170275 SSIM 86.0496 LPIPS 12.2685) — +0.50 over round10b
The 60k/8M tower+chair upgrade delivered +0.5046, mid of the predicted +0.4..0.9 band (10th
consecutive submission improvement across both sets). Gap to top-1 (80.56) now 3.95.
LB set2: 72.48450 -> 76.10509 -> 76.60965.

## BONSAI PERCEPTUAL PASS — 3/4 in (repo-SSIM scorer; capD rescored 71.1564 = comparable)
| cand | iters | lpips_from | lambda | PSNR | SSIM | LPIPS | SCORE |
|---|---|---|---|---|---|---|---|
| pC_lpearly | 30k | 12k | 0.1 | 26.763 | 0.8478 | 0.2533 | **71.3605** (leader, +0.20 vs capD) |
| capD (base) | 30k | 25k | 0.1 | 26.802 | 0.8485 | 0.2595 | 71.1564 |
| pB_lpw2 | 30k | 20k | 0.2 | 26.363 | 0.8458 | 0.2534 | 71.0574 |
| pA_60k | 60k | 40k | 0.1 | 26.433 | 0.8441 | 0.2615 | 70.7211 |
FINDINGS: 60k HURTS bonsai (pA < base — scene wants short training); lambda 0.2 trades too much
PSNR for its LPIPS; longer LPIPS phase at 0.1 (pC) is the win. Perceptual axis on bonsai is
modest (+0.2, ~+0.03 mean) — reinforces the strategy audit: frame-interp is the bigger video bet.
pD_60k_lpw2 still running (expect < pC given pA).

INTERP PROBE launched (GPU0, no training): DIS-flow midpoint interpolation of each video eval
hole from its two flanking train-sub photos; baselines = nearest-flank copy + the render scores.

## INTERP PROBE — GAME-CHANGER CANDIDATE KILLED (17/07, measured in 20 min, zero training)
DIS midpoint interpolation of video eval holes from flanking train photos (the strategy audit's
lever 3 / "reopen IBR for 3-4deg video"):
| scene | interp | nearest-copy | render |
| chair | 50.62 (PSNR 18.1) | 41.89 | ~77 |
| bonsai | 59.54 (PSNR 20.6) | 53.15 | 71.36 |
Interp >> copy (flow works) but 6-8 dB BELOW the renders — 3-4deg still moves content too much
for 2D flow (occlusion/perspective). Not seam-limited -> no case for neural interp escalation.
As an ensemble member it is dominated (weights would drive to ~0). REOPEN CLOSED; photo-reuse
stays dead on video too. The video path forward is therefore: perceptual recipe (pC +0.2) +
per-frame exposure (--ppisp narrow variant) + possibly FastGS-family diversity — grind, not magic.

## CHAIR IS THE HIDDEN LAGGARD — strategy audit's data-gap #1 answered (18/07)
chair_base60k (production recipe on train_sub, 58 eval holes): PSNR 24.645 SSIM 0.7764
LPIPS 0.2571 SCORE 67.79 — BELOW bonsai (71.36). Decomposition rewrite: video pair ~68-71
=> the 5 towers must average ~79.5 (round11 76.61 mean). The >1pt prize is now TWO scenes:
chair 68->77 = +1.3 mean alone (caveat: eval trained on 72% of frames; full-train test likely
a bit higher, bonsai's eval->test gap was ~1pt).
VISUAL DIAGNOSIS (chair_eval_diag.jpg): render structurally EXCELLENT (mesh backrest, tissue
box, geometry all correct) — deficit is FINE-TEXTURE SOFTNESS, dominated by the carpet
(~half the frame, washed-out fibers vs crisp GT). SSIM 0.776 is texture washout, not failure.
=> levers: longer LPIPS phase (bonsai's pC winner — carpet is where perceptual bites),
churn-stop (running), possibly per-frame appearance (sub-px RS/DoF jitter blurs texture).
CANDIDATE QUEUE (GPU0): churnstop60k (running) -> lpearly60k (lpips_from 30k) -> ppisp?
GPU1: bonsai pC full retrain (2 seeds) for the round12 bundle.

## CHAIR EVAL LOOP RESOLVED + ROUND12 CHAIN LAUNCHED (18/07 12:30)
Chair candidates (58 eval holes, repo-SSIM scorer):
| cand | PSNR | SSIM | LPIPS | SCORE |
|---|---|---|---|---|
| lpearly60k (lpips_from 30k) | 24.749 | 0.7808 | 0.2434 | **68.536** WINNER (+0.74) |
| base60k (production) | 24.645 | 0.7764 | 0.2571 | 67.794 |
| churnstop60k | 24.633 | 0.7730 | 0.2591 | 67.607 |
churn-stop does NOT transfer to chair (glass-specific fix); the LPIPS-phase texture lever DOES
(all 3 axes up — matches the carpet-washout diagnosis). Same lever won on bonsai (pC):
"longer LPIPS phase at standard weight" is now the confirmed VIDEO-scene recipe theme.
NOTE: GPUs idled ~4h after the 08:00 results (no wake trigger); acceptable loss, resumed 12:23.
ROUND12 chain running unattended: chair lpearly full-205 retrain (2 seeds parallel) ->
ensemble no-field -> compose with r11 towers + bonsai-pC (ready, 28 imgs) ->
sub_round12_videofix.zip + verify + provenance. ETA ~16:00-16:30.
Expected: chair +0.74 eval and bonsai +0.20 eval => ~+0.13 mean; plus full-train>eval-train
margin. Modest but banked; the bigger video levers (ppisp) queue next.

## ROUND12 BUILT + VERIFIED (18/07 14:53) — sub_round12_videofix.zip READY TO SUBMIT
386/386, 349.1 MB, VERIFY PASSED, provenance sidecar written. = r11 towers (unchanged) +
chair lpearly full-205 2-seed (eval winner 68.54 vs 67.79) + bonsai pC full-248 2-seed
(eval 71.36 vs 71.16). Both video scenes no-field. Est ~76.75-77.0 vs r11 76.61.

## VIDEO PPISP CANDIDATES launched (both GPUs, eval-gated) — lever 4
GPU0 chair ppisp_lpearly (stacks PPISP on the 68.54 winner), GPU1 bonsai ppisp_pc (stacks on
71.36). PPISP: per-frame exposure/WB controller, activation 29/30 default (exp25b warning
respected: render only post-activation ckpts, --ppisp at render). If either beats its baseline
on eval, it rolls into round13.

## ROUND12 GRADED: 76.64900 (PSNR 26.143955 SSIM 86.0329 LPIPS 12.1181) — +0.039, new best, 11th straight
LPIPS improved as designed (12.269->12.118) but PSNR/SSIM ticked down; net +0.039 vs the
eval-predicted +0.134. CALIBRATION POINT (the audit's un-run public calibration, now measured
on the LB): eval-delta +0.94 scene-pts -> test-delta +0.28 => TRANSFER DISCOUNT ~0.3x for
small recipe nudges. Ranking transfers (r12 did improve); magnitude compresses (full-train
baseline closes part of the texture gap that lpearly exploited on the thinner train_sub).
STRATEGIC CONSEQUENCE: +0.2-0.7 eval wins are worth +0.06-0.2 real — grind-scale. To move the
3.91 gap meaningfully we need BIG swings: FastGS gate family (LB-proven ensemble mechanism,
not an eval-magnitude bet) is now the top queued lever; ppisp verdicts (eval) land today and
roll in only if they clear a HIGHER bar (eval +0.5 minimum to be worth a slot at 0.3x).
LB set2: 72.48450 -> 76.10509 -> 76.60965 -> 76.64900. Gap 3.91.

## PPISP VERDICTS: DEAD on both video scenes (18/07) — lever 4 closed
bonsai_ppisp_pc 71.2285 vs pC 71.3605 (-0.13, a LOSS); chair_ppisp_lpearly 68.5999 vs
lpearly 68.5364 (+0.06, noise). The 25%-drift regime argument did not cash out — the canonical
model absorbs smooth drift; the metric is texture-dominated. Both fail the new >=+0.5 eval bar
by an order of magnitude. No round from this lever; half a day well spent killing it.

## FASTGS GATES QUEUE LAUNCHED (lever 5, set1-proven) — 18/07
3 gate members x 6 scenes (5 towers + chair; bonsai excluded, collapse-prone):
champA=gate1, memB=gate5, memC=gate2, all on g15 champion base (GRAD_ABS 0.00015,
--lambda_lpips 0.1 @25k). No negative-k1 on set2 -> no ring, masks all-ones, gates fully
supervised everywhere. OUT=/mnt/d/avv/output_s2gates. ~25-30 min/member (set1 timing) ->
~4-5h wall. Then round13 = tier-3 family composition (w_UT 0.3x2 + gates 0.1333x3, set1-tuned
weights) on towers+chair, + bonsai pC -> zip. Set1 evidence: gates+family-weighting = +0.5-0.9.

## CONSULT #3 RECEIVED (user-run, Vietnamese) + TONIGHT'S KILL-TESTS EXECUTED (19/07 ~01:00)
Consultant's core: gap is 27.4 scene-pts; videos hold 20 of them (69->79 = +2.86 mean); towers
max +1.07. "Content ceiling ~70" challenged on 3 grounds (60k REVERSING = optimizer instability
not saturation; video test geometry is the EASIEST in the comp; benchmark arithmetic says decent
indoor = ~81, chair 23dB at 3-deg interpolation is abnormal). Key protocol: LB is deterministic,
~30 submissions left -> SINGLE-SCENE-SWAP A/Bs measure exact per-scene deltas (Dmean x 7), stop
shipping on the 0.3x-compressed eval alone. ADOPTED.

Kill-tests run tonight (CPU-only, gates queue owns GPUs):
- KILL-TEST #0 (train-fit): chair TRAIN PSNR 26.7, towers 27.8 -> UNDERFIT/BUG branch (train-test
  gap only ~2dB; no generalization cliff). Routes to M1/M2, kills exposure/generalization tracks.
- M3 exposure oracle: perfect per-frame 3ch gain+bias = +0.109 dB chair / +0.192 bonsai.
  DEAD FOREVER (threshold was +0.3). PPISP null now fully explained: exposure was never the issue.
- M4 blur/sharpen sweep: chair blur0.4 +0.16 axis-pts pre-LPIPS (nets ~0), sharpening HURTS both.
  No frequency-response lever; motion-blur modeling premise gone. Deprioritized.
- M2 AUDIT: consultant's scene_scale mechanism DISCONFIRMED (our scene_scale is camera-hull-based:
  8.41 vs hull 7.64, normal) BUT the point-tail is real and bonsai-specific: p100/p95 = 9.9
  (96.5 units vs p95 9.7) vs 3.5-4.5 on honest scenes = mirror-world reflection points in the
  INIT cloud. Revised mechanism: poisoned init seeds MCMC relocation fog. --init_clip added to
  train_gsplat (drop init pts beyond camera-hull x R).
- M1 discovery: NO CODE NEEDED — train_gsplat without --ut ALREADY uses rasterize_mode
  "antialiased" (gsplat forbids antialiased+UT; UT forced classic). On pinhole videos k1=0:
  UT = zero benefit + antialiasing lost. We cargo-culted --ut onto the videos.
Staged kill_tests.sh (fires when gates land): K1 bonsai noUT-aa, K2 bonsai init_clip+standard
churn, K3 chair noUT-aa 60k, K4 pC seeds 7/1234 (winner's-curse spread).

## GATES DONE (6/6, 0 failures) + KILL-TEST VERDICTS + ROUND13 BUILT (19/07 morning)
KILL-TESTS (consult#3):
- K1 bonsai noUT/ANTIALIASED: **71.90 (+0.54 vs pC 71.36) — WINNER, best-ever all 3 axes**
  (PSNR 26.98, SSIM 0.8503, LPIPS 0.2448). M1 CONFIRMED on bonsai: UT was pure cost on pinhole
  (k1=0), and dropping it unlocks gsplat antialiased mode. Clears the >=+0.5 bar.
- K2 bonsai init_clip1.5 + standard churn: 42.89 (PSNR 12.3) — STILL COLLAPSES. Mirror-point
  init was NOT the mechanism; churn-stop remains the only fix. M2 DEAD (clean kill).
- K4 seed spread (partial): pC seed7 71.219 vs seed42 71.361 -> spread ~0.14 => noise floor
  ~ +-0.15; K1's +0.54 is ~4x the floor. seed1234 + K3 (chair noUT-aa 60k) still running.
ROUND13 BUILT+VERIFIED: sub_round13_towergates.zip 386/386 344.5MB. Towers = family ensemble
(2 UT @0.3 + 3 gates @0.1333 + field); chair gates EXCLUDED (visibly degraded: smudges,
window corruption — FastGS classic struggles on the portrait video; rule: weak member dilutes);
chair+bonsai = r12 unchanged. Since only towers changed vs r12: LB delta x 7/5 = per-tower
gate gain — round13 doubles as the gates A/B.
NEXT: when K3/K4 land -> bonsai noUT-aa full-train (2 seeds) + chair (if K3 wins) -> round14
as video-scene swap on top of r13.

## K3 CHAIR ANTIALIASED: 69.3693 (+0.83 vs lpearly 68.54) — M1 CONFIRMED ON BOTH VIDEO SCENES
PSNR 24.98 (+0.23) SSIM 0.7911 (+0.010) LPIPS 0.2338 (-0.0096) — best chair ever, all axes,
~5x the +-0.15 seed-noise floor. Combined with K1 (+0.54 bonsai): dropping --ut on the pinhole
videos (unlocking gsplat antialiased mode) is the biggest video lever since the collapse fix.
ROUND14 CHAIN LAUNCHED (round14_build.sh, unattended): both winners full-train 2-seed
(bonsai 30k aa, chair 60k aa; GPU1 waits for K4_seed1234 then chair seed7) -> compose with
r13 towers -> sub_round14_videoaa.zip + verify + provenance. ETA ~15:30-16:00.
Eval-predicted delta: (0.54+0.83)/7 = +0.20 mean at face value; 0.3x calibration says ~+0.06;
BUT the aa change is a RENDERING-MODE change (structural, like the collapse fix) not a recipe
nudge — transfer factor genuinely unknown. r14-vs-r13 LB delta x 7/2 measures it exactly.

## ROUND13 GRADED: 76.05292 (-0.5961 vs r12) — GATES REGRESSION on set2 towers
PSNR -0.137 SSIM -0.581 LPIPS +0.850 vs r12. Only towers changed -> per-tower delta -0.835.
ROOT CAUSE (both RUNBOOK and consult#3 warned; I under-weighted it): STALE FAMILY WEIGHTS.
Set1's 0.3/0.3+0.1333x3 was tuned when UT members were 30k/5M (gates within ~0.15 of them);
set2 UT members are 60k/8M — far stronger — so 40% weight on 30k FastGS gates diluted all axes
(SSIM -0.58 = floater/blur signature). "Weak member dilutes" rule violated at scale.
DECISIONS: r12 (76.649) remains best. Towers REVERT to r12 composition permanently. Gates
shelved (a tiny-weight re-sweep is possible later but deadline economics say no). r14 must NOT
ship on r13 towers -> r14_fix.sh chainer armed: rebuilds sub_round14b_videoaa_r12towers.zip
(r12 towers + aa videos) when the training chain exits; the chain's own r14 zip is superseded,
marked DO-NOT-SUBMIT in provenance. r14b-vs-r12 LB delta x 7/2 = per-video aa gain (clean A/B).
LB set2: 72.485 -> 76.105 -> 76.610 -> 76.649 -> [76.053 regression, reverted]. Gap 3.91.

## AUDIT ROUND 15 (19/07 12:00) — CLEAN; both time-critical questions resolve FOR the r14b plan
1. r13 render-path alternative REFUTED (code+pixels): --mult is the SnugBox tile-footprint
   multiplier (auxiliary.h:383), NOT a resolution scale; gate renders' HF-energy ratio vs UT =
   0.956/0.964 (a 0.7x-res render would sit ~0.7-0.8) -> gates are a WEAKER MODEL, not soft
   renders. Stale-weights diagnosis STANDS; no re-render salvage; only-by-weight (w_gates
   {0,0.1,0.2} eval sweep possible; shelving = defensible default).
2. K1/K3 APPLES-TO-APPLES CONFIRMED: same split, identical args minus --ut; fused<->repo SSIM
   delta measured 0.0001 pts (my earlier 0.09 bound was repo-vs-skimage, inapplicable);
   render mode read from ckpt (aa ckpts render antialiased, no flag needed). K1/K3 stand.
3. init_clip correct (einsum inversion, xyz+rgb same mask, warm-start no-op by ordering);
   kill_tests seed override last-wins confirmed; chains: no deadlock/self-match (fragilities
   noted: start-races, no deadlines — carry-forward); r13 0.9999 weights renormalized (no
   bias); eval_score repo-ssim now exactly matches score_submission.
VERDICT: ship r14b as staged. Carry-forwards: pgrep-chain hardening, k1-probe fixed already,
init_clip log line cosmetic.

## OPS INCIDENT + RECOVERY (19/07 19:20): 5.4h GPU idle
round14_build's own zip came out 350.015MB (aa renders compress worse -> ladder overshot the
cap by 15KB, verify FAILED — superseded zip anyway). r14_fix.sh then died at conda activate
(NVCC unbound var under set -u — the ONE script I forgot the export in), so r14b never built
and the V1/T1 orchestrator was written but never launched (interrupted before launch, launch
never verified). GPUs idle 13:53-19:17. Lessons re-learned: (a) EVERY set -u script gets the
NVCC export; (b) verify every launch (pgrep) before ending a work cycle; (c) build_submission
now called with --max_mb 348 for aa-content zips.
RECOVERY: r14b built directly + VERIFIED (347.6MB, 386/386) -> sub_round14b_videoaa_r12towers
READY TO SUBMIT (provenance poisons the oversize r14). V1+T1 orchestrator launched+verified
alive: V1 = cross-family aa+UT video ensemble sweep on eval holes (bar >=+0.3/scene);
T1 = seed-13 UT members on 5 towers (~8h) for the proven 1/N ensemble mechanism.

## STRATEGY AUDIT #2 (fresh agent) — key deliverables adopted
- FRAMING FIX: towers aa-vs-UT already tested on set1 (exp19 74.507 vs exp27 75.214 — aa LOST
  by 0.71; k1!=0 pays undistort+warpback resamples). Tower aa as REPLACEMENT: dead prior
  (10-15%). As ENSEMBLE ADD member (T2, w~0.2, one-tower LB swap): open, 40-50% prior — queued
  behind T1.
- MECHANISM READ: composition changes historically transfer >=1x to LB (R2 +1.82, R5 +0.90,
  R3 +0.23); recipe nudges 0.3x. Plan = composition-heavy at matched generations.
- Gates: written off as planned lever; optional single-slot probe at w_gates~0.1 after 26/07.
- Honest ceiling: +0.25-0.6 by freeze -> ~76.9-77.3; gap to 80.56 does not close on any
  branch; goal = defend/extend near-2nd. Freeze 28/07, final = per-scene argmax over graded
  swaps.

## AUDIT ROUND 16 (whole-codebase sweep) + ACTIONS (19/07 20:00)
Fault-side CLEAN: auditor cleared the "invisible uniform cap" class — train/test projection
bit-identical both regimes; chair/bonsai intrinsics consistent (no 1.5x mismatch); no
gamma/colorspace bug; LPIPS-loss default 0.1 saves the video recipes that omit --lambda_lpips.
Game-changers ranked: B1 eps2d (hardcoded 0.3, over-smooths interpolated video poses), B2
per-scene 4:4:4 chroma, B3 sharper combine, B4 rolling-shutter (chair, but forfeits aa).

ACTIONS:
- B2 chroma: DEAD (free eval-hole A/B). chair 4:2:0 69.07 vs 4:4:4 69.02; bonsai tied
  (70.794 vs 70.797). 4:2:0 is equal-or-better AND 30% smaller. Keep ss2. (Side note: JPEG
  q100 encode costs ~0.3 vs PNG eval — the unavoidable encode floor, ladder already optimal.)
- B1 eps2d: PATCHED end-to-end (train --eps2d -> ckpt -> render reads it; matched). A/B running
  both GPUs: bonsai+chair at eps {0.2, 0.1} vs the aa baseline eps 0.3 (K1 71.90 / K3 69.37).
  This is the highest-EV untried lever (sharpness -> LPIPS, the video deficit).
- T1 seed-13 towers queued after eps2d on each GPU (proven 1/N ensemble mechanism).
- B3/B4 deferred (B3 needs >=3 members; B4 forfeits the aa win).

OPS: 3rd conda-var-under-set-u death (deactivate hook _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED)
cost ~43min idle. PERMANENT FIX: env-switching scripts now use `set -o pipefail` WITHOUT -u,
explicit || checks. moves_r16.sh verified surviving conda switches. r14b STILL AWAITING SUBMIT.

## ROUND14/14b GRADED (20/07): r14b = 76.78777 NEW BEST (+0.139 vs r12). Gap 3.77.
r14b (r12 towers + AA videos) = 76.78777. r14 (r13 towers + AA videos) = 76.19280.
VIDEO-AA gain = +0.486/video-scene (Dmean 0.1388 x7/2) vs eval-predicted +0.685 =>
TRANSFER 0.71x — STRUCTURAL changes transfer ~2.4x better than recipe nudges (0.3x). CONFIRMED
the strategy-audit thesis empirically; this is the operating rule now: chase structural levers.
CROSS-CHECK: r14b - r14 = +0.595 = the r13 tower gates regression (-0.596) to 3 decimals. Two
graded zips independently confirm both facts; deterministic single-scene-swap protocol validated.
LB set2: 72.485 -> 76.105 -> 76.610 -> 76.649 -> [76.053 r13] -> [76.193 r14] -> 76.788 (r14b).
NEW BEST = sub_round14b_videoaa_r12towers.zip. eps2d (B1) is the SAME structural class -> if it
wins eval, expect ~0.7x transfer (a +0.4 eval win -> ~+0.08 mean, per video scene x2).

## B1 eps2d DEAD (20/07) — sharpness axis tapped on video scenes
bonsai: eps0.3(default)=71.90 > eps0.1=71.60 > eps0.2=71.53. chair: eps0.3=69.37 > eps0.2=69.26.
LOWER eps2d (sharper) is WORSE on both -> the "over-smoothing" hypothesis is BACKWARDS: the video
GT is genuinely blurry (DoF/motion/defocus bg), so a sharper render diverges MORE. 0.3 low-pass
partially matches real softness. Consistent with M4 (blur ~neutral). Keep gsplat default (eps2d
patch stays available but unused). Audit r16 game-changers now: B1 DEAD, B2 DEAD, B3 defer
(needs >=3 members), B4 defer (forfeits aa). Fault-side clean. Reachable video wins exhausted on
the sharpness/chroma axes -> video scenes at content ceiling for current method class.
T1 seed-13 towers continue (queued in moves_r16 after eps runs) -> round15 tower-diversity swap.

## ROUND15 BUILT (20/07) — 3-seed tower ensemble (T1), AWAITING SUBMIT
moves_r16 done: eps2d DEAD (above) + all 5 seed-13 tower members trained (ut13, identical
r2r9 recipe 60k/8M refine50k noise50k lpips_from50k, --ut_render native). Composed round15 =
r14b with towers upgraded 2-seed->3-seed UT ensemble (ut7+ut42+ut13, w=1/3 each) + SAME
train-fit DIS field (seed-independent, reused r2r9/fields/*.npy). Video scenes UNCHANGED from
r14b (r14 aa ensembles). zip = sub_round15_3seedtowers_videoaa.zip, 344.8MB, 386 files, CRC OK,
VERIFY PASSED. NOTE: 3-seed@q100=358.7MB>350 -> ladder dropped HCM0421+HCM0540 to q99 (~-0.002/
scene, negligible vs ensemble gain). All-tower swap vs r14b -> LB delta x7/5 = per-tower 3rd-seed
gain. Expected +0.05-0.15/tower (proven 1/N: mean2 +0.61, mean3 +0.78 on HCM0181 eval, audit r3);
ensemble is a COMPOSITION change -> transfers ~1x. Predicted LB: ~76.83-76.89. Baseline r14b=76.788.
Zero GPU cost (members already banked). NOT auto-submitted — user submits + reports grade.

## ROUND15 GRADED (20/07): 76.95340 NEW BEST (+0.166 vs r14b). Gap 3.61.
PSNR 26.333868 (+0.102) SSIM 86.4132 (+0.242) LPIPS 11.9271 (-0.080).
All-tower swap -> per-tower 3rd-seed gain = 0.16563 x 7/5 = +0.232/tower, ABOVE the
predicted +0.05-0.15 (eval 2->3 increment +0.17 transferred ~1.4x). THIRD consecutive
confirmation: COMPOSITION changes transfer >=1x (collapse fix ~1x, aa 0.71x struct,
3rd seed 1.4x). q99 drop on 2 towers cost nothing measurable.
LB set2: 72.485 -> 76.105 -> 76.610 -> 76.649 -> [76.053] -> [76.193] -> 76.788 -> 76.953.
BEST zip = sub_round15_3seedtowers_videoaa.zip.
NEXT (T2, same mechanism): 4th tower seed (eval 3->4 increment +0.10/tower -> LB ~+0.10-0.14
after transfer) + 3rd video seed on chair/bonsai aa (2->3 increment ~+0.17 eval-scale, x2/7
scenes -> ~+0.05 mean). Combined projection round16 ~77.05-77.15.

## ROUND16 BUILT (20/07) — T2: towers 3->4 seed + video 2->3 seed, AWAITING SUBMIT
moves_r17 done: 4th tower seed (ut77, all 5 towers) + 3rd video seed (aa13, chair+bonsai)
trained, identical recipes to their siblings. Composed round16 = towers 4-seed UT ensemble
(ut7+ut42+ut13+ut77, w=1/4) + SAME DIS field, chair/bonsai 3-seed AA ensemble (aa42+aa7+aa13,
w=1/3, no field). zip = sub_round16_4seedtower_3seedvideo.zip, 348.6MB, 386 files, CRC OK,
VERIFY PASSED. q100 all scenes except HCM0421 (q99, 355.5MB>350).
NOTE: unlike r15, this round changes ALL 7 scenes at once (towers AND videos both extended)
-- not a clean single-axis swap. If the LB delta needs isolating later, fall back to two
follow-up zips (r15-towers swapped to 4-seed only / r15-videos swapped to 3-seed only) using
the same banked members, zero extra GPU cost. Baseline to beat: r15 = 76.95340.
Projection: towers 3->4 ~+0.10-0.14 mean (eval-scale, diminishing 1/N), videos 2->3 ~+0.05
mean (2/7 scenes) -> combined ~77.05-77.15. Zero incremental GPU cost, all members banked.
NOT auto-submitted — user submits + reports grade.

## ROUND16 GRADED (20/07): 77.09550 NEW BEST (+0.1421 vs r15). Gap 3.465.
PSNR 26.419089 (+0.085) SSIM 86.6485 (+0.235) LPIPS 11.8763 (-0.051).
Composition: towers 3->4 seed (ut7+42+13+77, w=1/4)+field AND chair/bonsai 2->3 seed
(aa42+7+13, w=1/3) CHANGED TOGETHER (not isolable this round -- both axes moved at once).
Landed inside the projected 77.05-77.15 band -- projection calibration holding on
composition-class moves (3rd straight: r15 predicted-vs-actual also inside band).
LB set2: 72.485 -> 76.105 -> 76.610 -> 76.649 -> [76.053] -> [76.193] -> 76.788 -> 76.953 -> 77.096.
BEST zip = sub_round16_4seedtower_3seedvideo.zip.
Diminishing returns visible: tower gain per seed 2->3 was +0.232, this combined 3->4+2->3
step (2 mechanisms at once) only netted +0.142 total -- consistent with 1/N flattening
(4th member marginal ~1/4-1/3 of 3rd) rather than a regression; can't fully separate axes.
Freeze 28/07 (8 days out). Target band 77.0-77.3 (strategy audit) -- ALREADY IN BAND.
Next: Fable-agent fresh audit launched to hunt further game-changer directions (not just
incremental seeds) -- diminishing 1/N returns mean seed-stacking alone won't reach 77.3+.

## FABLE AUDIT (20/07) + B3 TESTED-DEAD + opacity_reg claim REFUTED
Fresh unanchored strategy audit (fable model) ranked: (1) B3 sharper-combine [unlocked, was
deferred pending >=3 members], (2) bonsai opacity_reg/scale_reg -> 0 sweep, (3) chair carpet
fine-texture loss, (4) reuse stale gates renders at low weight.

VERIFIED before acting (per audit-practice discipline):
- (1) B3 TESTED, DEAD. HF-energy diagnostic first (free, CPU-only): naive ensemble-mean loses
  6-20% Laplacian-variance HF energy vs a single member (worst on bonsai -20%, chair -12%,
  towers -6..-8%) -- real, measurable blur. But testing an actual detail-preserving combiner
  (LF=blur(mean), HF=per-pixel winner-take-all by local |HF| magnitude among members) against
  REAL TRAIN PHOTOS (13-view sample, Rule-10-clean, same legality as fit_field.py) shows naive
  mean BEATS the combiner: chair 78.73 vs 78.39 (-0.33), bonsai 75.12 vs 75.07 (-0.04). The
  "missing" HF energy is mostly per-seed noise/misregistration, not recoverable signal --
  winner-take-all just reintroduces sharp-but-wrong detail. Same mechanism as eps2d kill
  (audit r16): sharper != closer to genuinely-soft GT. B3 status: DEFER -> DEAD (tested, not
  just theorized). Script: /home/bkai/.claude/jobs/1c9cf7e9/tmp/test_combiner.py.
- (2) opacity_reg/scale_reg->0 sweep: audit cited exp26 (+0.16) as support, MISREAD. The actual
  DIRECT test of this exact intervention (exp21, opacity_reg 0.01->0.002) was REJECTED --
  overfit, ALL test metrics down (EXPERIMENTS.md:158, set1 HCM0181, audit round 6). exp26's
  +0.16 came from a bundled warm-start+short-finetune recipe (10k iters, means_lr x0.3, FastGS
  champ ply init) that doesn't exist in the current gsplat_track pipeline for bonsai, and even
  THAT recipe still lost -0.165 to the champ baseline. NOT queued -- the isolated mechanism the
  audit recommended already has a negative result on record. If revisited, must go through the
  full exp26-style bundle (warm-start + short finetune), not a standalone reg sweep, and would
  need --init_ply wiring that may not exist in the current bonsai recipe.
- (3) chair carpet fine-texture loss: diagnosis CONFIRMED real (EXPERIMENTS.md:1570-1577,
  chair_base60k eval SSIM 0.7764, texture washout not structural failure, ~half frame carpet).
  No prior test of a targeted fix exists. Requires a real code change (GT-gradient-weighted
  loss term in train_gsplat.py) -- next candidate, needs implementation + eval-split validation
  before any GPU commit.
- (4) reuse stale gates renders at low weight: audit's own lowest priority, confirmed low-value
  -- /mnt/d/avv/output_s2gates only has 2/5 towers rendered (HCM0421 champA/memB), incomplete,
  cannot pre-validate via eval-split (gates trained on FULL train, eval holes would leak). Not
  pursued.
Net: cheap/zero-GPU tier of ideas now exhausted (seed-stacking flattening + B3 dead). Remaining
credible lever is (3), a real loss-function change -- moving to scope + implement it next.

## DEEP RESEARCH (20/07) — texture-washout loss techniques, citations VERIFIED
General-purpose research agent (web-enabled) investigated the chair carpet fine-texture
problem (SSIM 0.7764, worst-submetric LPIPS, texture washout not structural failure per prior
diagnosis; B3 post-hoc detail-injection already tested dead today). Found 4 candidate
techniques; VERIFIED the two load-bearing citations by direct WebFetch before trusting them
(per audit-practice discipline):
- **LEGS (arXiv 2606.07932) — VERIFIED REAL.** "Laplacian-Enhanced Gaussian Splatting with a
  Nonlinear Weighted Loss" (Guo/Huo/Sun/Gong). Second-order Laplacian structural guidance ->
  nonlinear response-to-weight map -> weights the L1 photometric term. Reports up to +1.68dB
  PSNR over standard 3DGS, +1.69dB when integrated into FastGS. Single-line loss substitution,
  no architecture change, weight map derived from the GT training photo itself (Rule-10 clean,
  no pretrained net). ADOPTING THIS FIRST.
- **MH-3DGS (arXiv 2506.12945) — VERIFIED REAL BUT MISCHARACTERIZED by the research agent.**
  Agent claimed "built directly on 3DGS-MCMC, reuses the stock renderer, ~5min/scene overhead."
  Direct fetch of the actual abstract CONTRADICTS this: the paper explicitly states it is "NOT
  built on 3DGS-MCMC" — it implements true Metropolis-Hastings accept/reject ("birth sampler"),
  whereas our pipeline's MCMC relocation is SGLD-based always-accept (the real 3DGS-MCMC
  algorithm). Porting this would mean implementing a different density-control algorithm, not a
  drop-in relocation-criterion swap. DOWNGRADED from "most mechanistically on-point, cheap" to
  "real idea, too large an implementation lift for the time remaining" — correctly caught before
  any GPU/dev time was spent on the false premise.
- AbsGS (gradient-collision diagnosis) and FFT/wavelet loss: informative context / complementary
  axis, not directly actionable for MCMC relocation as-is.
Lesson: SOURCED claims from a research agent still need the actual source pulled and read —
the agent's own citation, once fetched, refuted its own headline claim about implementation
cost for option B. Continuing to verify-before-act on every subagent recommendation.
NEXT: implement LEGS-style Laplacian-weighted L1 in gsplat_track/train_gsplat.py as an opt-in
flag, test via eval-split on chair (and bonsai, same generic mechanism) before any full-train
GPU commit.

## texture_weight IMPLEMENTED + SWEEP LAUNCHED (20/07)
Added --texture_weight flag to gsplat_track/train_gsplat.py (LEGS-inspired, arXiv 2606.07932,
simplified to first-order Sobel gradient magnitude since the paper's exact 2nd-order nonlinear
mapping wasn't read in full): weight map = clamp(sobel_mag(GT_gray)/mean, max=6), floor 1x,
l1 = mean(w * |render-gt|). Weight derived ONLY from the training photo already being trained
on -- Rule-10 clean, no external data/net, single-flag opt-in (default 0=off, byte-identical
to unmodified baseline). Cached per-view (static, GT-only). Smoke-tested 25 iters, no crashes,
loss decreasing normally -- clean.
Sweep launched (tw_sweep.sh, monitor bwfhlh3il): lambda={1.0, 3.0} bracket on chair (lpearly
recipe, existing eval baseline 69.37) and bonsai (capD/pC recipe, existing eval baseline 71.90),
on their existing train_sub eval-splits, both GPUs. ~2h/chair run, ~1.5h/bonsai run, ~4h total
per GPU (2 sequential values each). Zero risk to production: baseline zips untouched, this only
trains NEW candidate models under /mnt/d/avv/tw_test/.

## texture_weight SWEEP RESULT (21/07): DEAD — monotonic regression on chair, noise-level on bonsai
                  baseline    tw1.0       tw3.0
chair  SCORE      69.37       68.6935     67.8594   (-0.68, -1.51 -- MONOTONIC REGRESSION)
bonsai SCORE      71.90       72.0606     71.8613   (+0.16, -0.04 -- peaks near tw1.0, reverts by tw3.0)

Chair: clean, dose-dependent, CONSISTENT regression -- more texture-weighting = worse, no
ambiguity. This is the scene the whole intervention targeted (carpet fine-texture diagnosis)
and it went backwards. Bonsai: +0.16 at tw1.0 is within this project's established single-seed
noise floor (~0.15-0.3, cf. PPISP bonsai -0.13/chair +0.06 treated as small-but-real at similar
magnitude) -- NOT a confident win on one run, and tw3.0 already reverts most of it.
LIKELY MECHANISM (HYPOTHESIS, consistent with 2 prior kills on the same scenes): chair/bonsai
GT is genuinely blur-limited (phone-video DoF/motion blur, CONFIRMED via the eps2d kill --
sharper renders diverge further from soft GT). Sobel gradient magnitude on a blurry photo is
dominated by blur-edge / sensor-noise gradients, not recoverable fine texture -- up-weighting
those pixels pushes the optimizer to overfit blur/noise patterns in TRAIN frames that don't
generalize to held-out eval poses. Same root cause as the eps2d and B3 kills: sharpening/
emphasizing high-frequency content doesn't help when the GT's own high-frequency content is
degraded, not recoverable detail.
IMPLEMENTATION CAVEAT (for the record): weight map was `1 + lambda*w` (unnormalized) -- mean
total loss weight grows with lambda, so this ALSO shifts the effective L1:SSIM balance in the
total loss beyond pure spatial reweighting, a confound not isolated from the "reweight WHERE"
effect. A properly-normalized version (weight/mean(weight), preserving mean=1) would isolate
the pure spatial-redistribution effect. NOT re-testing: the monotonic direction + a mechanism
consistent with 2 independent prior findings makes a sign-flip from re-normalization unlikely,
and GPU time is better spent elsewhere with 7 days to freeze.
VERDICT: texture_weight DEAD as a lever on this dataset. THREE consecutive audit-sourced ideas
now tested and killed today (B3, opacity_reg refuted pre-test, texture_weight) -- all trace to
the same root cause: video-scene GT is blur-limited, and no loss/combine trick recovers detail
that was never resolved in the source photography. This is a DATA ceiling, not a method gap,
for chair/bonsai specifically. Towers remain the only lever class with a positive track record
(seed-stacking), though also flattening (r15 +0.232/tower -> r16 combined step only +0.142
total). Flag for strategy reconsideration: 7 days to freeze, r16=77.09550 already inside the
77.0-77.3 target band -- diminishing-return territory on further experimentation.

## USER-COMMISSIONED DEEP RESEARCH (21/07): pose refinement implemented + tested
User brought an external deep-research report ranking 6 game-changer candidates for the final
7 days, anchored on Darmon et al. "Robust Gaussian Splatting" (arXiv 2404.04211, ECCV24, Meta
Reality Labs) -- a ScanNet++ handheld-iPhone ablation (near-exact analog to chair/bonsai) where
pose optimization contributes +0.79 PSNR / +.026 SSIM / -.031 LPIPS, the largest perceptual-
metric lever, ranked above color correction, Mip-Splatting 3D filter, LPIPS/WD-R loss, blur
forward-modeling, and robust down-weighting. Report independently flagged MH-3DGS as not
worth pursuing -- matches yesterday's own finding.
VERIFIED before acting:
- Robust GS paper is real; abstract matches (motion blur as pose distribution, defocus
  compensation, color inconsistency correction) -- couldn't confirm exact Table 2 numbers from
  abstract-only fetch, but the paper and mechanism are real.
- CORRECTION: gsplat's `CameraOptModule`/`pose_opt`/`AppearanceOptModule` do NOT exist in our
  installed gsplat 1.5.3 (grepped the actual package -- only low-level NVIDIA SE3/pose-interp
  CUDA kernel wrappers under geometry/sensors, unrelated to a training-time pose-opt module).
  Those names are gsplat's GitHub `examples/simple_trainer.py` (a separate reference script),
  not our custom trainer, which deliberately avoids gsplat's Parser/Dataset. NOT a drop-in flag
  -- built as a genuine addition following the existing --app_affine architecture pattern.
- CORRECTION: report's #2 pick (per-image affine color) has ALREADY been tested on this exact
  trainer lineage: exp23/idea-1c, HCM0181, REJECTED -0.40 ("appearance variation is not low-dim
  affine"). Different scene (tower vs video), not conclusive against chair/bonsai, but a real
  prior the report lacked. Queued a cheap re-grade on chair (flag already implemented, zero new
  code) in parallel rather than skip or blindly trust.
IMPLEMENTED: --pose_opt in gsplat_track/train_gsplat.py. Per-training-view learnable SE3
residual (axis-angle via Rodrigues + translation), composed in the camera's own frame
(R_new=dR@R0, t_new=dR@t0+dt), L2-reg to identity (--lambda_pose_reg, translation scaled by
scene_scale), separate Adam optimizer (--pose_lr). Applied ONLY to training-view poses inside
the render call during optimization -- test poses (CSV, render_gsplat.py) are completely
untouched, so this can only improve the FITTED GAUSSIANS via better supervision, never touches
the render-time camera (no test-GT dependency, Rule-10 clean). Verified gradient flow through
the full corrected_w2c composition (Rodrigues + eye(4) slice-assign) in isolated autograd
checks before trusting the real run. Smoke-tested 30 iters, clean, loss decreasing normally.
Mechanistically distinct from today's 3 kills (B3, texture_weight, opacity_reg-refuted): those
all chased/preserved HIGH-frequency detail and failed because blur-limited GT has no recoverable
HF signal there; pose refinement corrects LOW-frequency GEOMETRIC misalignment across ALL
pixels via a rigid transform, which cannot overfit per-pixel noise the same way.
TESTING NOW: pose_opt_test.sh (monitor beb2euvyq) -- chair (GPU0) + bonsai (GPU1, after
appaffine frees it) on production recipes vs baselines chair=69.37, bonsai=71.90. app_affine
re-grade on chair also running (GPU1 first). Both zero-risk to production (new candidates only,
existing best zip untouched).

## app_affine RE-GRADE on chair (21/07): DEAD, second confirmation
chair_appaffine: PSNR 24.8863 SSIM 0.7904 LPIPSvgg 0.2450 SCORE 68.8448 vs baseline 69.37
(-0.53, regression). CONFIRMS exp23's prior verdict ("per-image affine absorbs real signal,
not low-dim") now on a SECOND, different scene class (video/phone vs tower) that has real
documented exposure drift -- this is not a scene-specific fluke, idea-1c/affine-color family
is dead across both scene types tested. Matches the deep-research report's OWN explicit risk
flag: correction can't be properly applied at test time (chair's test poses interpolate the
affine from nearest train frames by camera-center KNN, not a real GT-fit) -- exactly the
"test-time application impossible" failure the report warned about.
(Ops note: eval ran under the gsplat conda env due to a missed `source conda.sh` in a one-off
inline bash -c launch, not the pose_opt_test.sh/tw_sweep.sh pattern -- verified harmless:
identical torch 2.7.1+cu128 and byte-identical VGG weight checksum in both envs, so the score
is valid or comparable to prior fastgs2-env numbers.)
Waiting on pose_opt (the report's real #1 pick, mechanistically distinct from every idea killed
so far) -- chair ETA ~18:15-18:25, bonsai ~19:55-20:05.

## pose_opt FIRST ATTEMPT: CRASHED, root cause diagnosed (21/07)
chair_poseopt eval: PSNR 16.08 SSIM 0.4376 LPIPS 0.4205 SCORE 45.9593 vs baseline 69.37 --
CATASTROPHIC (-23.4), not a modest regression. Killed bonsai_poseopt mid-run (pid 19365,
~2min in) rather than burn another ~1.5h confirming the same bug on a second scene.
DIAGNOSED via an instrumented 8000-iter run (temp pose_r/pose_t magnitude print, kept in code
gated behind --pose_opt): pose corrections themselves stayed TINY and well-regularized (max
rotation ~0.005 rad / 0.3deg, max translation ~0.0137 / scene_scale 5.718 = ~0.24%) -- NOT a
runaway/gauge-drift problem, lambda_pose_reg is doing its job on magnitude. The actual failure:
gaussian count exploded 80k->2.97M in the first 8000/60000 iters (should reach 8M smoothly by
refine_stop=50000) while train-view SSIM oscillated wildly (0.74->0.47->0.70) instead of
converging. ROOT CAUSE (diagnosed, not just hypothesized): each of 147 training views got a
fully INDEPENDENT per-view SE3 correction with no cross-view consistency constraint. Since
these are continuous video frames, uncorrelated per-frame corrections (even individually tiny)
break multi-view triangulation consistency -- the same 3D point viewed by two adjacent frames
no longer agrees, and MCMC's density-control reads the resulting inconsistency as "needs more
geometry", explosively over-densifying to locally patch what is actually a self-inflicted
consistency problem. Matches the deep-research report's OWN caveat, which explicitly said
"restrict to training-view refinement plus a SHARED learned camera model" -- the first
implementation used fully independent per-view DOF instead, missing exactly that constraint.
FIX IDENTIFIED, NOT YET BUILT: temporal-smoothness regularizer between sequentially-adjacent
frames' pose deltas (video sequences have continuous camera motion -- neighboring corrections
should agree). Bounded addition to existing code, not a rewrite. Checking with user before
spending more time given today's tally: 3 dead ends (B3, texture_weight, opacity_reg-refuted)
+ 1 confirmed-dead re-test (app_affine) + this crash-then-diagnosed bug, all in one day.

## pose_opt smoothness-fix RETEST: INCONCLUSIVE, earlier diagnosis was premature
Added --lambda_pose_smooth (temporal-adjacency L2 penalty between consecutive frames' pose
deltas, default 1.0) and reran the same 8000-iter diagnostic on chair. Result: gaussian-count
growth trajectory is BYTE-IDENTICAL to the broken run at every 1000-iter checkpoint (102726,
167325, 272548, 443946, 723136, 1177905, 1918678, 2976494 -- same numbers both runs), and SSIM
oscillation pattern is essentially unchanged (still swings 0.44-0.75). SELF-CORRECTION: this
means the earlier "cross-view inconsistency -> MCMC over-densification" diagnosis was NOT
actually confirmed -- it was a plausible mechanistic story fit to a single broken run with NO
baseline (non-pose_opt) comparison at the same early checkpoints, and the smoothness fix
produced no observable change this early. Two live possibilities, not yet distinguished: (a)
this N-growth curve is NORMAL MCMC behavior on this recipe/scene regardless of pose_opt (never
verified a true baseline trajectory at iter 8000, only final-iteration numbers from unrelated
runs), and the real divergence happens LATER (interacting with lpips_from=30000 or
noise_stop=50000, similar in kind to bonsai's documented late-stage MCMC churn collapse) -- an
8000/60000-iter snapshot cannot see that; OR (b) the smoothness term is genuinely not the fix
needed. Only a full-length run distinguishes these. Launching the real (smoothness-fixed)
eval-split test on both scenes now rather than continue partial-diagnostic guessing.

## FABLE CONSULT 2 (depth analysis, 21/07): verdict + verification
Seeded with real candidates from consult 1 + our exact failure-mode constraint (3 HF-chasing
attempts already dead today; fix must REMOVE unreliable signal or fix LOW-freq error, not add
detail). Verdict: implement self-contained per-view Laplace/aleatoric uncertainty weighting
(Kendall&Gal NeRF-in-the-Wild style, generalized L1->Laplace-NLL), NOT RobustNeRF, deprioritize
DWTGS. Explicitly declined to force a positive recommendation if none fit.
VERIFIED before acting:
- DWTGS (arXiv 2507.15690) CONFIRMED sparse-view-specific by direct abstract fetch ("sparse-view
  scenarios... overfitting to high-frequency noise... from few views") -- wrong regime for our
  200-250 dense video frames, kill CONFIRMED SOLID.
- RobustNeRF-port claim PARTIALLY OVERSOLD: Fable cited github.com/paulungermann/Robust3DGaussians
  as explicitly stating a direct RobustNeRF port causes "too aggressive masking" -- direct fetch
  shows that EXACT framing is not in the README. What IS true: their enhanced method (+segmentation
  +trainable layer) outperformed the standalone RobustNeRF-port variant in their own results table.
  Weaker evidence than claimed, but Fable's OWN independent mechanism argument (RobustNeRF assumes
  TRANSIENT minority outliers across views; our blur is PERSISTENT across nearly every frame, so
  masking would create a self-reinforcing zero-gradient hole, and it would compound with MCMC
  relocation the same way pose_opt v1's over-densification did, just inverted -- starvation not
  explosion) is sound on its own and doesn't depend on the citation. AVOID verdict on RobustNeRF
  STANDS on mechanism grounds even after discounting the citation.
- Kendall&Gal aleatoric/Laplace NLL uncertainty weighting: well-established, foundational Bayesian
  deep learning technique (NeurIPS 2017, widely used incl. NeRF-in-the-Wild) -- verified sound from
  first principles, not just trusting the citation. loss = |img-gt|/b(x) + log(b(x)), reduces to
  plain L1 at b=1, log(b) term makes escaping large residuals costly (real equilibrium, NOT a free
  mask like RobustNeRF -- structurally can't zero-gradient-starve a region the way RobustNeRF could).
  Self-contained: NO external pretrained network needed, Rule-10 trivially clear.
- Charbonnier/Huber: confirmed mathematically weak by first-principles check (L1's gradient is
  already constant-magnitude/bounded -- Charbonnier only smooths the zero-residual kink, converges
  to identical behavior for large residuals). Free to try later, low expectation.
IMPLEMENTING NOW: per-view learned confidence field b(x), 16x16 low-res grid (forces spatial
smoothness, can't pixel-cheat), warm-up freeze until ~step 2-3k. Will smoke-test + launch
eval-split test once a GPU frees (bonsai pose_opt v2 ETA ~20:25-20:35, chair ~20:55-21:05).
Per Fable's own caution: test standalone against baseline first, NOT stacked on pose_opt, to
keep attribution clean.

## FABLE CONSULT 3 (trick-hunt, 21/07): verified findings, 2 immediately actionable
Practical/forum/GitHub-issue search (not papers). 9 candidates found, 3 verified directly
(not just trusted the citation):
- ROLLING SHUTTER (gsplat native): CONFIRMED via inspect.signature(rasterization) on our
  actual installed gsplat 1.5.3 -- `rolling_shutter` and `viewmats_rs` params ARE real,
  already present, MCMC-compatible per docs. Models per-scanline pose during fast handheld
  motion instead of one rigid pose/frame -- a GEOMETRIC fix, different mechanism class from
  everything tried today (not loss-weighting, not detail-chasing). Both video scenes are
  handheld phone captures, plausibly rolling-shutter sensors. HIGH relevance, real code lift
  (need per-frame velocity or interpolated second pose), CLEAR compliance. Strong next
  candidate after pose_opt/uncertainty_weight results land.
- DEAD-GAUSSIAN BLOAT: CONFIRMED on our OWN checkpoints (not just the cited gsplat issue #822)
  -- chair_appaffine ckpt: 26.9% of 8M gaussians at opacity<0.01, 46.5% at opacity<0.05 (mean
  opacity 0.195). HCM0421 tower ckpt: 20.3%/32.5% (mean 0.201). Real, verified, WORSE on chair
  than towers. Unclear yet whether this is a fixable inefficiency (MCMC relocation not
  aggressive/frequent enough) or expected steady-state equilibrium of the sampling scheme --
  flagged for investigation, not yet actionable with confidence.
- POSE-OPT LR/WARMUP finding: nerfstudio issue #1635 CONFIRMED real -- symptom (default
  camera-opt LR causes blurrier results on low-texture-strong-edge images) matches our
  situation; fix was 10x LR reduction (6e-4->6e-5); their hypothesis is pose-opt runs at full
  strength from step 0 before gaussian geometry stabilizes, compounding early noise. THIRD
  independent hypothesis for pose_opt v1's crash (distinct from cross-view-consistency).
  ACTIONABLE GAP FOUND: pose_opt has NO warmup (activates step 0), unlike uncertainty_weight
  which already uses a 3000-step warmup pattern. If pose_opt v2 (smoothness fix) still
  underperforms, v3 = --pose_lr 1e-5 (10x lower) + a --pose_warmup analogous to
  --uncertainty_warmup is a concrete, evidence-backed next attempt.
- Also: cap_max sizing heuristic (cheap sanity check, SOURCED from 3dgs-mcmc README), AsymGS
  EMA-of-parameters (arXiv 2506.03538, addresses oscillating-trajectory concern generically,
  real code lift), SfM point filtering COUNTER-EVIDENCE (gsplat issue #411: filtering points
  made floaters WORSE, argues against that folklore trick), random background augmentation
  (low relevance, full-frame indoor scenes), larger SSIM window (SPECULATIVE, cheap 1-line
  test), noise cooldown in final stretch (SPECULATIVE but notes chair's CHA recipe has almost
  NO cooldown -- noise_stop=50000 of iters=60000, only last 16% noise-free -- vs bonsai's BON
  recipe at noise_stop=8000/30000=73% noise-free training; real asymmetry worth a look).
Three more parallel scene-specific trick-hunts in flight (towers/lattice, bonsai/foliage+glossy,
chair/furniture+carpet+portrait-video) -- not yet landed.

## pose_opt v2 (smoothness fix) GRADED: STILL CRASHED, fix did not work
chair_poseopt_v2: PSNR 16.5524 SSIM 0.4454 LPIPS 0.3984 SCORE 47.3575 vs baseline 69.37 (-22.0)
bonsai_poseopt_v2: PSNR 21.1397 SSIM 0.7143 LPIPS 0.3361 SCORE 60.6676 vs baseline 71.90 (-11.2)
v1 (no smoothness) chair was 45.96 (-23.4) -- v2's smoothness fix bought only +1.4, essentially
NO improvement. Both scenes still catastrophically regressed. Final train-view SSIM ended low
on both (chair 0.6328, bonsai 0.6657 -- healthy convergence should be ~0.85+). pose_r/pose_t
magnitudes grew substantially larger over the FULL 60k/30k run than the earlier 8000-iter
diagnostic showed (chair pose_t max 0.0296 vs 0.0137 at iter 8000; bonsai pose_t max 0.0595) --
CONFIRMS the divergence happens LATER in training, past what any partial diagnostic could see.
CONCLUSION: temporal-smoothness regularization was NOT the fix. Two independent real attempts,
two failures. The nerfstudio-issue-#1635-sourced hypothesis (pose-opt runs at full LR from
step 0, before gaussian geometry stabilizes, compounding early noise; fix = 10x lower LR +
warmup) is now the leading remaining explanation, UNTESTED. Given 2 failed attempts + real GPU-
hours already spent, checking with user before a v3 (pose_lr 1e-5 + pose_warmup ~5-10k steps,
same warmup pattern already used for uncertainty_weight) rather than assuming another try.

## uncertainty_weight (Laplace/aleatoric) PARTIAL RESULT: bonsai essentially a wash
bonsai_uncweight: PSNR 27.0139 SSIM 0.8478 LPIPS 0.2524 SCORE 71.5466 vs baseline 71.90 (-0.35).
Within/at the edge of this project's established noise floor (~0.15-0.3) -- NOT a confirmed
win, but nowhere near the pose_opt catastrophe either. Note: final train loss went negative
(-0.8791), expected/correct for the Laplace NLL (log(b) can dominate) but the magnitude
suggests b drifted well below 1 broadly (model became more "confident" than baseline almost
everywhere) -- worth a look if pursued further, not alarming on its own. chair_uncweight still
running (GPU0), ETA ~23:00-23:10.

## uncertainty_weight FULL RESULT: FAILED on both scenes
chair_uncweight: PSNR 25.0494 SSIM 0.7808 LPIPS 0.2622 SCORE 67.9671 vs baseline 69.37 (-1.40)
bonsai_uncweight: SCORE 71.5466 vs baseline 71.90 (-0.35)
Both negative -- not catastrophic like pose_opt, but no win either. TALLY for today's video-
scene hunt: eps2d DEAD, B3 DEAD, texture_weight DEAD, app_affine DEAD (x2), pose_opt v1
CRASHED, pose_opt v2 CRASHED, uncertainty_weight FAILED (both scenes). SIX independent,
mechanistically-diverse attempts, six failures/regressions on chair+bonsai today. Strong,
mounting signal that these two scenes may be at a genuine method-class ceiling for
per-frame-parameter/loss-reweighting approaches specifically -- pivoting effort: (a) one more
cheap, evidence-backed pose_opt v3 attempt (10x lower LR + warmup, nerfstudio-issue-#1635
pattern) since GPUs are free and it's nearly zero-cost to rule in/out, (b) parallel: absgrad=True
flip test on TOWERS (free, gsplat-native, no prior failure history on towers -- towers remain
the domain with a positive track record via seed-stacking).

## NEW CANDIDATE (22/07, user idea): learned post-render RESTORATION model
Idea: a learned image-restoration net at the pipeline OUTPUT. Input = our render, trained on
(train-render, train-photo) PAIRS, applied to test-pose renders. This is the GENERALIZATION of
the confirmed lens-field correction (which is the LINEAR/low-capacity version of exactly this
pattern: fit on train render-vs-photo, apply to test renders, +0.73 on LB). Same Rule-10
legality class (TRAIN GT only, never test GT). Fable consult launched (agent a56...) on: relevant
literature (NeRFLiX-style render restoration -- but flag their EXTERNAL-dataset training as a
Rule-10 problem; need per-scene self-supervised variant), the CENTRAL train->test transfer risk
(net trains on train-view render errors, applied to differently-distributed test-view errors --
same failure class as the killed per-image affine + texture_weight, "absorbs train signal doesn't
transfer"), overfit defenses (residual-only bounded prediction, low-capacity, patch aug, freq-
limiting, early-stop on held-out train views), a concrete minimal-capacity architecture for ~200
per-scene pairs, the eval-split validation protocol to honestly estimate transfer, and an honest
go/no-go. KEY PRIOR: the linear version already won +0.73 -- open question is whether a bigger
model adds beyond it or just overfits. Verdict pending consult (will report when it lands).

## RESTORATION-MODEL CONSULT (22/07): VERDICT = worth ONE minimal test, prior = mostly overfits
Fable consult returned + VERIFIED (verify-before-trust): its load-bearing claim ("lens field =
95% of oracle recovery, r=0.94 train->test") is NOT hallucinated -- those are OUR OWN numbers,
pulled correctly from EXPERIMENTS.md ("unscaled field already gets 95% of the oracle",
"r=0.9977 across folds, r=0.94/0.88 train->test"). Consult was genuinely grounded in our data.
KEY IMPLICATION (sharpens the skeptical prior): the linear lens field already recovers 95% of
even the ORACLE test-fitted correction -> only ~5% systematic error remains, and that residual
is the LEAST cross-view-transferable part -- precisely what a high-capacity net overfits rather
than recovers. Combined with our killed per-image affine + texture_weight (both "absorb train
signal, don't transfer"), the honest prior is a big net mostly overfits.
LITERATURE (Fable, cited real papers): NeRFLiX/++ (arXiv 2303.06919/2306.06388) trains the
restorer ONCE on EXTERNAL data (LLFF-T+Vimeo90K), zero-shot per scene -- opposite design point,
Rule-10-ok-in-spirit but too heavy + wrong risk profile. NeRF-W (2008.02268): per-image
appearance embeddings must be RE-optimized at test time on half the image -> explicit admission
naive per-image correction overfits. WildGaussians/SWAG: per-image affine -> "overfitting/baked
shadows." Deep Image Prior early-stopping (2112.06074) + Perception-Distortion Tradeoff
(1711.06077, Blau&Michaeli): joint LPIPS+PSNR win only if GENUINE signal recovered, else
hallucination tanks PSNR. No paper does exactly our "small net on ~200 same-scene train pairs ->
novel poses" -> the gap is itself informative (known to overfit).
MINIMAL-FIRST SPEC (if/when we test it, Phase-2 candidate, ~1 day cap): 3-5 layer conv residual
net, 16-32 ch, NO downsampling, RF ~9-15px, GroupNorm/no-BN, patch-train w/ heavy aug, output =
render + alpha*tanh(net(render)), alpha~0.03-0.08 in [0,1] (HARD-bounded correction), L1 + light
patch-LPIPS. CRITICAL: apply AFTER apply_field.py (learn only the residual the field leaves).
VALIDATION (double-holdout via make_eval_split.py, no self-fooling): restoration net trains ONLY
on train_sub renders+photos; validate on eval-hole renders (same train_sub 3DGS model) scored vs
eval_gt (photos the net never saw). Compare vs field-only baseline on same holes. Report P/S/LPIPS
separately (watch perception-distortion trade). Ship only if it clears baseline by non-noise
margin. DECISION: queue as a Phase-2 candidate at MINIMAL capacity, but NOT top priority given
the 95%-of-oracle ceiling evidence -- one cheap read, kill fast if it doesn't beat field-only.
Fits the family-ensemble reframe: even a small/decorrelated win could be a member.

## OVERNIGHT RESULTS (22/07 morning) + config-batch DEADLOCK BUG
pose_opt v3 (LR 1e-5 + warmup, the nerfstudio-#1635 fix): chair 59.4744 (vs 69.37, -9.9),
bonsai 66.8379 (vs 71.90, -5.1). BETTER than v1/v2 (chair 45.96/47.36) -- the 10x-lower LR +
warmup genuinely helped -- but STILL a large regression. pose_opt DEAD across all 3 attempts
(v1 independent / v2 +smoothness / v3 +low-LR+warmup). The train-view pose correction on these
scenes systematically harms held-out novel views regardless of tuning; shelved.
absgrad A/B on HCM0181 tower: absgrad=True 74.9519 vs absgrad=False 74.9453 = +0.0066. NEUTRAL
(pure noise-level). Not a win; and near-identical models = not decorrelated enough to be a
useful ensemble member either. absgrad shelved.
OPS BUG (fixed): overnight_wave1.sh config batch DEADLOCKED and never ran -> GPUs sat idle
several hours. Cause: its wait-guard `while pgrep -f "poseopt_v3|absgrad_test.sh|HCM0181_absgrad"`
matched the MONITOR PROCESS's OWN command line (which contains those log-file names), so pgrep
always found a "match" and the guard never released. Same self-match class as earlier pgrep
issues. FIX: relaunched as config_batch2.sh directly (no wait -- training was already done) and
switched the monitor to key off a sentinel DONE file (config_batch2.DONE) instead of pgrep, which
structurally cannot self-match. LESSON for all future wait-guards/monitors: never pgrep a pattern
that also appears in the watcher's own argv; use sentinel files or match the actual python
process (train_gsplat) with a unique output-dir token.
config_batch2 relaunched (~10:20): chair capmax4M + noise40k (GPU0), bonsai capmax2.5M +
tower init_clip (GPU1). ETA ~14:00-14:30.

## USER DIRECTIVE 22/07: test ALL candidates BEFORE building any submission
User: "no, let go with testing all candidates first" (declined the safe 5th-seed r17 now).
-> hold ALL submission-building; exhaust the candidate sweep, keep winners + decorrelated
members per the family-ensemble rule, compose ONE strong ensemble at the end. Implementation
overlaps compute (plan principle): implement structural candidates WHILE tests run.
IMPLEMENTED (verified) this session, ready to test:
- --min_opacity (MCMCStrategy relocation threshold, default 0.005): reclaim the 27-47% dead-
  gaussian budget. Testing 0.02.
- --aniso_reg: penalize per-gaussian log-scale spread beyond ~10x ratio (needle/sheet -> floor
  sheets + thin-structure under-rep). CPU-verified loss term. Testing 0.1.
- --sky_dome N: seed N Fibonacci-sphere points at 3x hull radius, low-opacity grey, for tower
  open-sky floaters. CPU-verified shapes/geometry exact. Testing 50000 (tower).
- Mip-Splatting has NO native gsplat support (checked rasterization signature) -> needs manual
  3D-filter impl (~2-3h), deferred to a later wave.
QUEUED: phase2_wave1.sh (waits on config_batch2.DONE sentinel -- NOT pgrep) runs minop02/aniso01
on chair+tower + skydome on tower (5 eval-split runs, ~6h after config batch). Sentinel-based
monitor b1qjednt7 covers both batches. Next: keep implementing wave-2 structural candidates
(EFA-GS, Pixel-GS, glossy-downweight, planarity, foliage, EMA, depth-seed, rolling-shutter,
Mip-Splatting, restoration-model) while wave-1 runs.

## CONFIG BATCH results (22/07 ~15:00). Baselines: chair 69.37, bonsai 71.90, tower 74.9453.
- chair cap_max 8M->4M: 69.6460 (+0.28) -- MILD POSITIVE, only non-negative config signal.
  Aligns with the 47%-dead-gaussian finding: chair IS over-budgeted, halving helped. At noise
  edge (~0.15-0.3) but directionally real -> follow up with cap 2M to confirm the trend.
- bonsai cap_max 5M->2.5M: 71.4349 (-0.47) -- HURT. bonsai NOT over-budgeted, 5M was right. Dead.
- tower init_clip 2.0: 74.8282 (-0.12) -- HURT slightly. The 21.9% far-points are apparently
  useful (or harmless); clipping loses a bit. Dead-ish.
- chair noise cooldown 50k->40k: 69.3286 (-0.04) -- NEUTRAL. Dead.
Net: chair cap-cut is the one lead (mild). Appended chair cap2M follow-up to the pool.
Phase-2 pool now RUNNING (config sentinel fired), 8 jobs (min_opacity/aniso/sky_dome/EMA x scenes).

## POOL results (22/07, rolling):
- HCM0181 min_opacity 0.02: 74.9821 (+0.037 vs 74.9453) -- NEUTRAL on towers.
- chair min_opacity 0.02: 26.9870 (PSNR 10.1!) -- CATASTROPHIC COLLAPSE. Confirmed GENUINE (not
  a bug): training log ends at train SSIM 0.4241 / l1 0.1357 (vs tower's healthy 0.8699/0.0306),
  so it degraded DURING training. Mechanism: chair's low-opacity gaussians are LOAD-BEARING
  (soft/blurry content); min_opacity 0.02 makes MCMC aggressively relocate the ~47% low-opacity
  population every refine step -> churn-collapse, same class as the bonsai glossy-glass collapse.
  KEY ASYMMETRY: gently cutting CAP (8M->4M) HELPED chair (+0.28), but aggressively RELOCATING
  low-opacity gaussians DESTROYS it. chair min_opacity DEAD (no gentler retry -- mechanism wrong
  for chair). min_opacity DEAD overall (chair catastrophic, tower neutral).
Pool safe, other jobs unaffected; aniso + sky_dome + EMA + cap2M still queued/running.

## POOL results cont (22/07 ~19:40):
- chair aniso_reg 0.1: 69.2170 (-0.15 vs 69.37) -- NEUTRAL/slightly neg. Dead.
- HCM0181 aniso_reg 0.1: 74.8983 (-0.05 vs 74.9453) -- NEUTRAL. Dead. aniso_reg DEAD both scenes.
- HCM0181 sky_dome 50000: TRAIN FAILED (bug, not a real result) -- UnboundLocalError: my
  uncertainty_weight block had a redundant `import math` INSIDE main(), which shadowed the
  module-level math and poisoned the whole function, so the earlier sky_dome block (uses math.pi
  etc.) crashed. FIXED (removed the local import; math is module-level line 21). AST-verified no
  local math imports remain; smoke-tested sky_dome now inits +50000 dome pts and trains clean.
  Requeued to the pool. LESSON: never `import X` locally inside a function that already uses the
  module-level X earlier -- Python marks X local for the whole function -> UnboundLocalError.
TALLY so far: only chair cap_max 4M (+0.28) is non-negative; min_opacity dead, aniso dead,
init_clip dead, noise dead, bonsai cap dead. Scenes strongly near-ceiling. EMA (in flight),
sky_dome (requeued), cap2M still to come.

## EMA results (22/07 ~21:30): CHAIR IS A REAL LEAD, bonsai flat
chair_ema099: 69.8051 vs baseline 69.37 (+0.435) -- BEST result of today's entire sweep, clears
noise floor (~0.15-0.3) with margin. bonsai_ema099: 71.8829 vs 71.90 (-0.017) -- flat/noise.
LIKELY MECHANISM (coherent with other findings): EMA only activates in the POST-refine_stop
window (topology frozen). Chair's recipe: refine_stop=noise_stop=50000/iters=60000 -> EMA window
is the LAST 17% of training, which is the exact "almost-no-noise-cooldown" tail the trick-hunt
already flagged as suspicious (chair noise_stop=50000 vs bonsai's 8000/30000=73% noise-free).
The raw config test (--noise_stop 40000, forcing an EARLIER cooldown) came back flat/dead --
but EMA achieves the SAME underlying goal (denoise the late polish phase) at the PARAMETER
level instead of the schedule level, and it worked where the schedule tweak didn't. Bonsai's
recipe already bakes in an aggressive noise cutoff (capD fix, tuned specifically to prevent
late churn/collapse) -> its late window is already stable, nothing left for EMA to smooth,
explaining the flat result. Coherent story, not just a coincidence.
PushNotification sent (best sweep result, clears noise floor).
Waiting: HCM0181_ema099 (tower), chair_capmax2M -- both in flight.

## chair cap_max 2M result: TREND DOES NOT CONFIRM
chair_capmax2M: 69.3827 vs baseline 69.37 (+0.013) -- essentially flat/noise, does NOT extend
the 4M result's +0.28. Non-monotonic: 8M(baseline)->4M(+0.28)->2M(+0.01). This weakens confidence
that cap_max 4M's +0.28 was a real, robust trend rather than noise sitting at the top of the
~0.15-0.3 floor -- OR it's a genuine non-monotonic sweet spot at 4M specifically (budget curves
aren't always monotonic). Not chasing finer granularity given time cost; treating cap_max-4M as
a WEAK/UNCERTAIN lead, not a confirmed one. chair EMA (+0.435) remains the one clearly-above-
floor result today. Waiting: HCM0181 ema099 + skydome50k (both in flight).

## LEADERBOARD UPDATE (23/07): top-1 climbed to 82.17 (was 80.56)
Gap widened: r16=77.0955 vs new top-1=82.17 -> gap now 5.0745 (was 3.465). Someone made a real
move. Strategy-audit's "target band 77.0-77.3, defend 2nd" framing was calibrated against 80.56
-- worth revisiting once the candidate sweep concludes. Does not change current plan (test all
candidates per user directive 22/07) but raises the cost of further delay before composing a
submission: every day spent purely testing is a day top-1 could extend further. EMA (2/3 scenes,
clean mechanism) is the strongest lead so far -- worth prioritizing toward a submission once its
decay-sweep confirms, rather than letting it wait for the full candidate list to exhaust.

## SPEEDUP: built a SCREEN-TIER pool (23/07) -- ~4x faster candidate triage
User asked to find a way to test candidates faster. Built pool_runner_v2.sh: same claim-based
pattern as the proven full-length pool, but runs at 1/4 iters with every schedule milestone
(refine_stop/noise_stop/lpips_from) scaled by the SAME 1/4 factor (preserves the recipe's SHAPE,
just compressed) + reduced cap_max (proportional to iters -> less gaussians -> faster/step too).
chair ~2h->~30min, bonsai ~1.5h->~22min, tower ~2h->~30min -- roughly 4x throughput per candidate.
VALIDATION (in flight): chair baseline-screen vs chair-EMA-screen, to confirm the confirmed
full-length EMA win (+0.435) shows the SAME DIRECTION at screen length before trusting screen
mode for anything else. If validated: screen becomes the default for ALL new candidates (glossy-
mask, foliage-consistency, planarity-reg, EFA-GS, EMA decay-sweep, Mip-Splatting once built),
full-length reserved ONLY to confirm screen survivors before they count toward a submission.
CAVEAT logged for future self: screen mode is NOT safe for candidates whose failure mode is
specifically LATE-training (pose_opt's divergence only showed past iter 8000/60000 in the
original diagnostic) -- continue full-length-only for anything in that class.
Sealed the old full-length pool (touch phase2_jobs.SEALED) so it finishes its last 4 jobs
(chair/tower/bonsai ema999 + chair ema+cap4M stack) and fires its completion sentinel
(phase2_pool.DONE) cleanly rather than polling forever. Screen pool (v2) waits on THAT sentinel
via file check, NOT pgrep -- sentinel-only pattern is now the house rule after the earlier
pgrep self-match deadlock.

## USER CHOSE (b): peel GPU1 to EMA production now (23/07 ~01:55)
Restructured GPU allocation to answer "is the improvement real" directly:
- GPU1: dedicated to chair EMA PRODUCTION (decay=0.99, the CONFIRMED eval-split winner
  +0.435 -- validated on REAL private_set2 chair data, not a proxy). Full-train (not
  eval-split), rendering actual test_poses -- this is the first real production artifact
  from today's whole candidate hunt. Gated on job#11 (chair_ema999 eval test) finishing via
  its done-marker file, not pgrep. Output: /mnt/d/avv/r17/chair_ema099_seed42/test_png.
- GPU0: finishes draining v1's last job (tower_ema999, in flight), then automatically starts
  the screen-tier hunting pool (now GPU0-ONLY, killed+relaunched single-worker) for the
  remaining candidates (glossy-mask, planarity, foliage, EFA-GS).
NOTE on tower EMA: only validated on the SET1 HCM0181 PROXY scene (+0.366), not yet on any real
private_set2 tower -- deliberately NOT pushing tower to production yet. Chair is the safe,
directly-validated case to prove out the "is it real" question first; tower production should
wait for either (a) a quick real-set2-tower eval-split screen, or (b) chair's production result
confirming the mechanism transfers to full-scale before committing more GPU-hours to towers.
Removed jobs 13 (bonsai ema999 sweep, low value given bonsai was flat) and 14 (chair ema+cap4M
stack test) from the v1 queue to free GPU1 for production without losing in-flight work.

[02:06] chair_ema999 (decay=0.999, job#11) landed: SCORE 69.7416 vs baseline 69.37 (+0.37) --
wins, but LESS than chair_ema099's +0.435. Confirms 0.99 is the better decay (near-optimal,
not "more smoothing = better") -- production run (chair_ema099_seed42, GPU1) already uses 0.99,
so no change needed there. HCM0181_skydome50k also landed: SCORE 74.9212 (tower baseline
comparison TBD, but consistent with skydome being ~neutral per earlier bonsai/tower sweep).
v1 pool 11/12 done; only job#12 (HCM0181 ema999) left on GPU0, then phase2_pool.DONE fires and
screen-tier pool takes over GPU0. Chair EMA PRODUCTION started on GPU1 at 02:06:07 -- job#11's
marker cleared right on schedule, ETA ~04:06-04:15 for test renders.

[02:30] v1 pool COMPLETE (12/12), phase2_pool.DONE fired, screen-tier pool now running on GPU0
(chair_baseline_screen claimed first). job#12 HCM0181_ema999's score was missed by the 20-min
monitor polls (nothing captured it between 01:17 in-flight and 02:30 done) -- recovered by
re-running eval_score.py directly on its saved eval_png (cheap, no retrain): **SCORE 75.4293**
(PSNR 24.2410 / SSIM 0.8480 / LPIPSvgg 0.1139) vs tower baseline 74.9453 = **+0.484**.

INTERESTING: this BEATS HCM0181_ema099's +0.366 (75.3116) -- the OPPOSITE ranking from chair,
where decay 0.99 (+0.435) beat decay 0.999 (+0.37). So EMA's optimal decay looks SCENE-
DEPENDENT: chair prefers lighter smoothing (0.99), the tower proxy prefers heavier (0.999).
Mechanistic guess: tower's noise_stop=refine_stop=50000/60000 (only 17% noise-free tail, same
as chair) so that's not the differentiator -- more likely the tower's higher gaussian density /
finer high-frequency content (lattice, bolts) benefits from heavier denoising of the late SGLD
jitter than chair's blur-limited content does. NOT yet re-tested on a REAL set-2 tower (this is
still the set-1 HCM0181 proxy) -- updated PLAN_TO_85.md's F2 to sweep BOTH decays there, not
just adopt 0.99 by default.

[03:49] SCREEN-TIER VALIDATION, bonsai: bonsai_baseline_screen SCORE 64.5794 vs full-length
baseline 71.90 = **-7.3 pts gap** -- much worse than chair's screen-tier gap (67.92 vs 69.37 =
-1.45). The 1/4-iters recipe (BON_SCREEN: 7500 iters, cap_max 1.25M) preserves the SAME
fractional schedule as BON_FULL, but bonsai looks much more under-trained at that absolute
iteration count than chair is -- likely bonsai's sparser SfM init (54k pts, glossy table kills
SIFT) needs more absolute steps to converge, not just the same fraction of a shorter run.
FLAG: don't trust bonsai screen-tier deltas at face value yet -- if bonsai_ema099_screen (next
job) looks flat/noisy, that could be screen-tier floor-effect masking a real signal, not EMA
failing on bonsai. Will know once HCM0181 screen jobs land too (checking if towers have the
same problem). If confirmed, BON_SCREEN needs more iters (e.g. 10-12k not 7.5k) before it's
trustworthy for bonsai candidates -- chair's screen-tier stays validated as-is.

[04:09] bonsai_ema099_screen: SCORE 64.8173 vs baseline_screen 64.5794 = **+0.24** -- inside the
noise band (~0.15-0.3), i.e. reads as FLAT, matching the full-length bonsai EMA finding
(-0.017, also flat). Despite the -7.3 pt absolute under-training gap, the RELATIVE (baseline
vs EMA) comparison still landed on the same conclusion as full-length -- one data point, not
strong proof, but reassuring that screen-tier's DIRECTIONAL read may survive even when its
absolute score doesn't. Still want the HCM0181 screen arms (in progress) before trusting this
generally.

## R17 BUILT + VERIFIED (04:31) -- chair EMA single-slot swap

Chair EMA PRODUCTION finished 04:24:48 (58/58 test renders, 60k/8M, seed42, decay0.99, full
private_set2/chair/train -- not eval-split). Built `sub_round17_chairema_swap.zip`: r16's chair
aa42 member swapped for chair_ema099_seed42 (same seed, same recipe, only +EMA), aa7/aa13 and
all 6 other scenes byte-identical to r16. VERIFIED: CRC OK, 386/386 exact, all dims/formats
correct, 348.6MB (under 350MB cap, HCM0421 re-encoded q99 to fit). This is the cleanest possible
test: 1 scene changed, 1 of 3 members changed -- LB delta should isolate EMA's true in-ensemble
effect size on chair.
Immediately re-fed GPU1 (idle after chair production finished) with a REAL tower EMA test:
HCM0421, seed7 (matches r16's ut7 tower member for a future clean swap), decay=0.999 (the decay
that WON on the set-1 HCM0181 proxy, not 0.99 -- see 03:49 entry). This is the first EMA test on
actual set-2 tower data. ETA ~2.2-2.5h (chair production took ~2h17m end-to-end; real tower has
240 train imgs vs chair's 205, expect similar order).
Meanwhile GPU0 continues the screen-tier hunt (5/7 done: HCM0181 ema099/ema999 screen arms
still pending -- will cross-check the decay-flip finding at screen resolution too).

[04:50] SCREEN POOL COMPLETE (7/7). Last 2 arms landed and produced a SIGN-FLIP, not just a
magnitude difference:
  HCM0181_baseline_screen  72.2966
  HCM0181_ema099_screen    72.5782  (+0.2816)
  HCM0181_ema999_screen    72.3985  (+0.1019)
Screen-tier says decay0.99 > decay0.999 for towers. Full-length (03:49 entry) said the OPPOSITE:
decay0.999 (+0.484) > decay0.99 (+0.366). Combined with bonsai's screen-tier under-training gap
(-7.3 pts absolute), this is now the SECOND screen-tier reliability flag -- for towers
specifically, screen-tier doesn't just compress magnitude, it can flip which candidate wins.
DECISION: don't trust either proxy (set-1 HCM0181 full-length OR screen-tier) to pick the real
tower's decay. Launched BOTH decays on the SAME real scene+seed for a direct answer:
  GPU1: HCM0421 seed7 decay0.999 (already running since 04:33, was the full-proxy winner)
  GPU0: HCM0421 seed7 decay0.99  (started 04:51, was the screen-proxy winner AND chair's winner)
Whichever wins on REAL data settles it -- no more proxy-chasing. ETA both ~2.2-2.5h (~07:00-07:20).
SCREEN-TIER METHODOLOGY NOTE FOR LATER: 1/4-iters-with-proportional-schedule is safe for
directional triage on scenes similar to chair (modest absolute gap, ranking preserved) but
NOT yet trustworthy for towers (ranking flip) or bonsai (large absolute gap, though direction
happened to agree once). Needs a longer screen recipe (more iters, not just proportional
scaling) before it's used to make real decisions on those two scene classes.

## REAL TOWER DECAY SHOWDOWN RESOLVED (07:15-07:25) -- decay 0.999 wins

Both HCM0421 seed7 arms finished full production (60k/8M, 240 real train imgs, ~2h22-2h25m
each -- matches chair's timing). test_png (60/60 each) has no local GT (blind test poses), so
built a quick IN-SAMPLE check instead: ran `make_eval_split.py` on HCM0421 (40 isolated-hole
"eval" images, doesn't retrain anything -- just reused for scoring) and rendered BOTH already-
trained checkpoints at those 40 poses, scored against real GT:
  HCM0421_ema999 (decay0.999): PSNR 28.2174 / SSIM 0.9158 / LPIPSvgg 0.0599 / **SCORE 82.0100**
  HCM0421_ema099 (decay0.99):  PSNR 28.2116 / SSIM 0.9154 / LPIPSvgg 0.0621 / **SCORE 81.9059**
decay0.999 wins by +0.104. CAVEAT: this is IN-SAMPLE (both checkpoints were trained on ALL 240
images including these 40, unlike the proper held-out eval-splits used elsewhere) -- weaker
evidence than a true generalization test, but the direction AGREES with the full-length HCM0181
proxy (0.999 +0.484 > 0.99 +0.366, a 0.118 head-to-head margin -- nearly identical margin to
this in-sample check's 0.104). Screen-tier's flip (favoring 0.99) is now 2-votes-to-1 against:
full-length proxy + real in-sample data both say 0.999, only the already-flagged-unreliable
screen-tier says 0.99. DECISION: decay=0.999 for towers (opposite of chair's 0.99) -- ships as
HCM0421's ut7 swap now; the OTHER 4 towers' EMA rollout (F4 in PLAN_TO_85.md) uses 0.999 too.

## R18 BUILT + VERIFIED (07:18) -- HCM0421 tower EMA single-slot swap, built against r16

Built `sub_round18_towerema_swap.zip`: r16's HCM0421 ut7 member swapped for HCM0421_ut7_ema999
(decay0.999), ut42/ut13/ut77 + all other 6 scenes byte-identical to r16 (built against r16, NOT
stacked on r17, to keep tower and chair attribution separate). VERIFIED: CRC OK, 386/386 exact,
348.6MB. Baseline to beat: r16 = 77.09550. Both GPUs still on the chair_aa7/aa13 EMA follow-ups
(ETA ~09:00-09:30) -- next zip will be the full 3-seed chair EMA trio.

## 09:19 chair_aa7 + chair_aa13 EMA production DONE -- both GPUs free, fed tower rollout

Both finished (58/58 test renders each). Immediately launched `tower_ema_rollout.sh` (F4): the
remaining 4 towers' ut7 slot -> decay0.999 EMA, GPU1 HCM0539->HCM0644, GPU0 HCM0540->HCM0674.

BUG (caught fast, no real cost): the function used one combined
`local G=$1 S=$2 M=/mnt/d/avv/r17/${S}_ut7_ema999` line. Classic bash gotcha: all RHS values in
a single assignment command expand BEFORE any of that command's assignments take effect, so
`${S}` expanded to its stale/empty value, not the `S=$2` just written on the same line -> both
HCM0539 and HCM0540 started training into the SAME bogus dir `/mnt/d/avv/r17/_ut7_ema999`
(would have silently clobbered each other's checkpoint). Caught within ~1 min via `ls` on the
output dir showing a suspicious `_ut7_ema999` name. Killed both immediately -- but their
`;`-chained subshells (`tower 1 HCM0539 ; tower 1 HCM0644`) treated the kill as "command
finished" and fell through to the NEXT scene in the same subshell, so by the time the fix
landed, HCM0644/HCM0674 were already running (correctly, fixed code) while HCM0539/HCM0540 had
been skipped entirely. Fix: split into 3 separate `local` statements (forces sequential
evaluation). Zero wasted GPU-hours -- every bad attempt died within seconds of starting, before
any real training. Queued `tower_ema_rollout_makeup.sh` to run HCM0539 or HCM0540 the moment
its sibling (HCM0644 or HCM0674, same original GPU) finishes, so the GPU assignment plan is
preserved and nothing sits idle.
LESSON for future scripts: never combine `local a=$1 b=$2 c=derived-from-$b` on one line --
always split multi-variable `local` declarations when a later variable depends on an earlier
one in the same statement.

## R18 GRADED: 77.0983 (+0.0028 vs r16) -- null at blended scale, direction positive again

PSNR 26.419958 (+0.0009) SSIM 86.6513 (+0.0028) LPIPS 11.8727 (-0.0036). All submetrics moved
the right direction by dust-sized amounts, same as r17. Implied HCM0421 scene-level delta =
+0.0028 x 7 = +0.02 -- well under the eval-split prediction (+0.484 single-model x ~0.22
in-ensemble compression / 7 ~ +0.015 blended predicted; got +0.0028, ~5x less, but both are
sub-noise numbers so the ratio is unreliable). PATTERN NOW CLEAR across r17+r18: single-member
swaps are UNMEASURABLE on the LB -- stop spending submission rounds on them. r19 (all 3 chair
members) is the decisive test of whether production EMA moves the LB at all. If r19 shows a
real positive, the play is ONE combined zip with EMA everywhere it's confirmed (chair trio +
all-5-tower ut7 swaps at minimum). If r19 is ALSO flat, EMA's eval-split wins don't transfer
to production scale and the whole EMA family gets shelved -- r17+r18+r19 together would prove
that with unusual cleanliness.

USER DIRECTIVE (23/07, immediately after r18's null): **"stop doing fragment improvement"** --
no more single-slot swap zips. Bundle all confirmed changes into ONE combined zip per round.
Saved to memory (feedback_family-ensemble-and-zip.md). Concretely: r19 (chair trio, already
built) stands as-is if the user chooses to submit it; the NEXT build after the tower rollout
lands (~11:30-12:00) is ONE combined zip = chair EMA trio + ALL 5 towers' ut7->EMA(0.999) --
call it r20_combined. No intermediate variants.

[12:12] ORPHAN RECOVERY: post-mortem of the 09:20 kill-cleanup found collateral damage the
monitor couldn't see. The broad `pkill -f "tower_ema_rollout.sh"` + explicit kills took out the
SECOND (fixed) instance's subshells as well as the first's -- but not before those subshells'
`;`-chains had advanced and launched HCM0644/HCM0674's trainers with the FIXED code. Net state:
both trainers healthy and correct (right dirs, right args) but ORPHANED (PPID=1) -- on exit,
nobody would render their test poses or touch their DONE markers, stranding the makeup script
(HCM0539/HCM0540) forever on its marker-wait. Also, the killed FIRST instance's main shell had
fired its final `touch /mnt/d/avv/tower_ema_rollout.DONE` on the way down (09:16 timestamp) --
a PREMATURE/stale completion sentinel. Fixes: (1) removed the stale sentinel; (2) launched
`orphan_adopter.sh` -- waits per-PID via `kill -0` poll (can't `wait` on a non-child), verifies
ckpt.pt exists (won't mask a crash), renders on the trainer's own GPU, verifies 60/60 renders,
THEN touches the per-scene marker (so the makeup trainer never overlaps a render on its GPU).
Makeup script confirmed untouched by the pkill (pattern `tower_ema_rollout.sh` with `.` as
ERE-any-char still can't match `rollout_makeup.sh` -- no `sh` right after the wildcard slot).
REVISED ETA: HCM0644/HCM0674 trainers at 2h46m elapsed (slower than HCM0421's 2h17m -- heavier
scenes), expect exit ~12:30-12:45, renders ~5min, then makeup pair trains ~2.5h -> ALL 4 towers
done ~15:15-15:45, r20_combined builds then.
TWO LESSONS, both now bitten twice-adjacent: (a) when killing a script that has `;`-chained
work inside subshells, kill the SUBSHELL PIDs FIRST, then the leaf processes -- leaf-first lets
the chain advance into new work between kills; (b) after any kill-cleanup of a script that
touches sentinels in EXIT paths, always audit for premature sentinel files left by the dying
instance (same class of bug as the pgrep self-match: the ops plumbing lying about state).

## STRATEGY AUDIT LANDED (23/07 ~13:40) -- fresh-agent read, REPRIORITIZES THE WEEK

Full text in the session transcript; load-bearing findings:

1. MY FRAMING ERROR CORRECTED: "dilution+compression eats everything" was over-generalized
   from the WORST transfer class. Measured transfer taxonomy (all from our own graded rounds):
   recipe nudge ~0.3x | structural-in-member ~0.7x | COMPOSITION change (new decorrelated
   member/family/weights) >=1.0-1.4x (never below 1x in 9 graded rounds) | correlated
   same-seed member swap (EMA) ~0.05x. EMA died because it's the most correlated possible
   swap, not because everything is compressed. POST-ENSEMBLE render-space transforms have NO
   compression tax at all -- the lens field (+0.7345 LB in one move) is our own existence proof.
2. SCENE-EQUAL WEIGHTING CONFIRMED via r10->r10b natural experiment (image-equal weighting
   would imply bonsai-fog at -2.2 dB PSNR, physically impossible). Our x7 arithmetic is right.
   BUT: videos = 2/7 = 28.6% of score and hold a ~10-pt/scene deficit vs our real towers
   (~79.5-80/scene reconstructed) -- and ~85% of recent GPU went to tower polish worth
   +0.003/swap. Videos are the most under-allocated asset IF the deficit is attackable.
3. TOP-1 (82.17) EXPLANATIONS, priced: either towers +4-7/scene better (H-A: LPIPS-moving
   method class -- exact/AA rendering or PRETRAINED RENDER-RESTORATION post-processing:
   Difix3D+ CVPR25 / GSFix3D / GSFixer / GFix -- legal under the user-affirmed Rule-10 reading,
   same class as VGG-LPIPS) or videos +10-17/scene better (H-B: per-frame BLUR MATCHING --
   crucial insight: all 3 of our "blur ceiling" kills (eps2d, B3, texture_weight) pushed in the
   SHARPENING direction; matching each test frame's blur/exposure is bounded by GT itself, not
   the camera. bonsai VoL p5-median spans 126-474 = 4x within-scene blur spread a static render
   cannot match. Untested direction.)
4. TOP-5 NEXT (by pts/GPU-h): (1) H-B bound test: fit per-train-frame blur kernels on existing
   eval-split chair ckpt's train renders, interpolate by frame index to eval holes, score --
   ~3 GPU-h, decides H-B and the blur-kernel moonshot in one shot. (2) H-A zero-shot: released
   Difix3D+/GFix checkpoint at low strength on existing tower eval renders -- 4-8 GPU-h,
   post-ensemble = untaxed x5 scenes. (3) zero-GPU combiner sweep on existing renders:
   per-pixel median, agreement-weighted mean, train-fitted unsharp on tower ensemble means
   (lens-field legality class) -- free lottery tickets. (4) F7 5th tower seed + 4th video seed
   as idle filler (+0.07-0.12, proven). (5) Mip-splatting as NEW FAMILY branch (composition
   class, ~1x transfer), build 2-3h.
5. KILL-LIST (adopted): F4 all-member tower EMA (~40 GPU-h for +0.03-0.08), further EMA sweeps
   after r19/r20 grades, Pixel-GS, depth-seeded carpet init, SELF-trained restoration net
   (superseded by pretrained restorers -- cheaper, legality-cleaner), any screen-tier-only
   productionization on towers/bonsai, single-slot swaps (already user-directed).
6. Honest projection if H-A and H-B both miss: r16 + 0.1-0.25 by freeze (~77.2-77.35).

VERIFY-BEFORE-ACT NOTE (house rule): the Difix3D+/GSFix/GFix/BAGS citations come from the
agent's web search -- verify the released-checkpoint claim + license before building the H-A
test; the H-B bound test needs no external anything (pure our-own-data, frame-index interp
only, same legality precedent as the validated appearance-interpolation finding).

## CODE AUDIT LANDED (23/07 ~13:45) -- shipped path CLEAN, 2 real findings + 3 footguns

MOST IMPORTANT CLEARANCES (all CONFIRMED by trace):
- EMA block fully clean: init lands exactly at the frozen-topology boundary (after the last
  possible relocation AND last noise injection), all 6 param tensors covered, save uses the
  final blend, render loads EMA ckpts identically, no UT interaction. Quat-EMA valid
  (rasterization normalizes internally); opacity/scale EMA correctly in logit/log space.
- r17/r18/r19 attribution BYTE-CLEAN: per-file CRC diff vs r16 confirms r17 changed only
  chair (58/58), r18 only HCM0421 (60/60), r19 only chair. The null results are real nulls.
- In-sample decay comparison VALID as used; its bias direction (favors less smoothing =
  tighter train fit) went AGAINST 0.999 and 0.999 still won -- strengthens the decision.

FINDINGS:
1. CONFIRMED: --absgrad is a SILENT NO-OP under MCMCStrategy (only default.py consumes the
   absgrad buffer; mcmc.py never reads it). The 21/07 absgrad A/B compared two identical
   configs; +0.0066 was pure run noise. Shelving was accidentally correct. Fence added.
2. CONFIRMED mechanics: the screen-tier EMA sign-flip (04:50) now has a mechanistic
   explanation -- shrinking the window 10000->2500 steps at fixed decay changes the estimator:
   0.999^2500 = 8.2% residual weight on the init snapshot (vs 5e-5 at full length), so
   screen-ema999 tested a DIFFERENT estimator, not a noisier version of the same one.
   ema099 unaffected (0.99^2500 ~ 1e-11). If screen tier ever picks decays again:
   decay_screen = decay_full ** (window_full/window_screen). The 0.999-for-towers decision
   stands (chosen by the two unbiased votes).
3. HYPOTHESIS-class footguns (not triggered today, hardened/noted): eval_score.py silently
   drops mismatched pairs (hard-error added); ensemble_renders.py PNG-over-JPEG preference is
   alphabetical-accident not construction (works for our .JPG/.png casing; NOT touching the
   shipping path mid-flight, noted for post-freeze); apply_field re-apply stamp doesn't
   survive ensembling + field meta lacks dims provenance (current order ensemble->warp->zip
   is safe; noted).

## H-B BOUND TEST, CHAIR (14:12): WEAK -- +0.094 scene-level, mostly not exposure

Protocol (scripts/blur_bound.py + blur_bound_chair.sh): rendered 147 train_sub poses from the
EXISTING eval-split chair_ema099 ckpt (~zero marginal GPU, ran alongside the tower trainer on
GPU0), fitted per-frame 5-param transform (signed unsharp/blur aniso-Gaussian kernel a,sx,sy,th
+ gain,bias; 120 Adam iters/frame on L1), interpolated params to the 58 eval holes by frame
index, applied, scored:
  baseline (chair_ema099 eval)   69.8051
  chair_blurmatch (full 5-param) 69.8994  (+0.094)
  chair_gainonly (exposure only) 69.8183  (+0.013)
READ: per-frame blur variation is REAL (fitted a swings -0.93..+0.34 across frames = some
frames want strong blur-matching, others strong unsharp) but exploiting it at this model class
buys <0.1 scene-level on chair = ~+0.013 blended. Train-fit L1 improvements were also small
(median a few %), i.e. the ceiling is low REGARDLESS of interpolation quality -- the transform
class itself captures little of the render-photo residual. H-B on CHAIR: effectively dead
(consistent with chair's tighter VoL spread). BONSAI test running now (the audit's actual
prime suspect: 4x within-scene blur spread, 126-474 VoL p5-med). If bonsai also comes back
<+0.3, H-B dies entirely and with it the blur-kernel moonshot (train-time version cannot beat
its own post-hoc bound at matching, only at reconstruction-sharpening -- which is the
direction already killed 3x).

## H-B BOUND, BONSAI (14:53): FLAT -- H-B IS DEAD, blur-kernel moonshot killed

bonsai_blurmatch 71.8771 vs baseline 71.8829 = -0.006 (nothing). Telling detail: the fit
STRONGLY wanted the blur direction (median a = -0.71, vs chair's -0.07) and improved train-
frame L1 -- but at eval poses the correction nets zero. Interpretation: training on blurry
frames already bakes the scene's AVERAGE blur into the model; the per-frame residual around
that average either doesn't interpolate (blur is not smooth in frame index -- autofocus hunts,
motion spikes) or is perceptually negligible after ensembling. H-B verdict across both video
scenes: chair +0.094, bonsai -0.006 -> blended ceiling ~+0.013. DEAD. Blur-kernel moonshot
(PLAN Part-2 #2) killed with it -- the train-time version cannot beat its own post-hoc
matching bound, and the sharpening direction was already killed 3x. Total cost of settling
this: ~40 GPU-minutes opportunistically on a busy GPU. The video deficit (~10 pts/scene) now
has exactly one live attack vector: H-A (pretrained restoration, Difix zero-shot running).

## H-A ZERO-SHOT (15:40): STRICTLY DESTRUCTIVE -- dead at every strength, every axis

Setup pain first (for the record): released nvidia/difix needs diffusers==0.25.1 /
transformers==4.38.0 / peft==0.9.0 (repo requirements.txt; modern diffusers lacks
FromOriginalVAEMixin) AND cu128 torch 2.7.1 for sm_120 (cu121 wheels = "no kernel image").
Isolated conda env `difix`; pipeline runs at native res (+pad to /8), ~4s/img.

RESULTS (real set-2 data, local GT):
  HCM0421 tower (baseline 82.0100): s1.0 66.8587 | s0.5 76.6726 | s0.25 80.6018
  chair       (baseline 69.8051):   s1.0 63.7743 | s0.5 68.3798 | s0.25 69.6550
Monotonic regression toward baseline as strength->0 on BOTH scenes; no crossover, no sweet
spot; LPIPS itself WORSE at every strength (tower 0.0599 -> 0.0780 even at s0.25). The
generic restoration prior hallucinates detail our renders don't need fixing -- we are far
above the artifact regime Difix was trained on (Nerfbusters-class LPIPS 0.33; ours 0.06).
ZERO-SHOT restoration: CLOSED.

WHAT SURVIVES: only the FINE-TUNED variant (Difix "offline mode": fine-tune on OUR train
render<->photo pairs so the model learns our actual residual distribution -- the user's
original restoration idea with a pretrained backbone). Priors are now LOWERED (H-B showed
chair's transform-correctable residual ~0.1; fidfit showed the residual is largely
non-reproducible content) but this is the LAST remaining multi-point-class candidate.
DECISION: one bounded fine-tune on chair (the 10-pt-deficit scene, 147 pairs on disk,
repo training code, ~4-6 GPU-h) tonight after r20 builds. Kill bar: <+0.3 on chair eval
closes the entire restoration class for this competition.

## COMBINER SWEEP + ADD-vs-REPLACE (16:00-16:20): mean confirmed; ADD >> REPLACE -- r20 RETARGETED

Combiners (strategy-audit rank-3, scripts/combiner_sweep.py, existing eval renders, ~free):
  tower mean 76.3883 > agree 76.3729 > median 76.3705; chair mean 71.0682 > median > agree.
  Median/agreement-weighted DEAD. Pixel-mean stays.
THE REAL FINDING -- config-jitter ensembles at eval scale:
  tower: best single 75.43 -> mean4 76.39 -> 5-member 76.43 (+1.0 over best single)
  tw_add2 (baseline + its own EMA twin, SAME seed): 76.0368 vs replace-analog 75.4293 = +0.61
  chair: best single 69.81 -> mean4 71.07 -> 6-member 71.15 (+1.34 over best single)
  Every added within-band member helped, even same-seed twins and ~1-pt-weaker members.
  This is the set-1 family-ensemble effect (B4warm precedent) QUANTIFIED, and it says r17/r18's
  REPLACE framing left the pair-decorrelation gain on the table.
ACTION: r20 auto-build retargeted (killed armed v1, rewrote, re-armed) from swap to ADD:
  towers = 5-member (ut7 + ut7_ema999 + 3 seeds), HCM0421 6-member (+ut7_ema099);
  chair = 6-member (aa trio + ema trio); bonsai = r16. Zero extra GPU (all renders on disk).
  Expected +0.1-0.25 blended (composition class). sub_round20_emaadd.zip when towers land.

## HCM0674 PATHOLOGICAL RUN KILLED + NIGHT RECOVERY (24/07 05:17)

HCM0674's EMA tower run (pid 254586, started 23/07 04:37 on GPU0) ran **19h20m** without
finishing -- vs HCM0644's ~4.5h on the byte-identical script. Diagnosed BEFORE killing: State R,
100% one core, GPU0 100% util, CPU-time advancing in real time (not a Python deadlock) -- it was
genuinely computing, just ~5-6x too slow, and never produced a ckpt or output dir. Root cause
not definitively found (candidates: afternoon GPU0 contention from the H-A/H-B/Difix-zeroshot/
combiner test jobs I stacked on it for hours, compounding into a pathological densification/
relocation state; or a thermal/clock issue over a very long run). Cost of the wrong call
(stacking test jobs on the same GPU as a production trainer): this one tower run + a full night
of Difix-FT being blocked (it was gated on HCM0674's marker). LESSON: never co-schedule
throwaway test jobs on a GPU running a long production trainer -- give the trainer a clean GPU
or accept it may thrash. Test jobs belong on the idle GPU or after production drains.
RECOVERY: killed HCM0674 + its orphan_adopter branch; freed GPU0; launched the blocked Difix FT
there directly (gate removed); rewrote r20 build to wait ONLY on HCM0540 and keep HCM0674 at its
r16 4-member ensemble (the other 4 towers still get the EMA add-composition). Net r20 impact:
4/5 towers EMA-boosted instead of 5/5 -- a ~20% haircut on the tower portion of an already-small
(+0.1-0.2) expected gain. HCM0674 EMA can be re-run cheaply later as a filler if a GPU frees
before freeze; NOT worth blocking r20 for.
DIFIX FT SETUP FIXES (for the record): repo trainer needed (a) Accelerator project_dir for the
tensorboard logger, (b) init-from-released-weights via the repo's pipeline_difix.DifixPipeline
(NOT diffusers.DiffusionPipeline -- no DifixPipeline in diffusers namespace). Patched
train_difix_ft.py accordingly; 140 train / 7 val chair pairs, 3000 steps, init from nvidia/difix.

## R19 BUILT + VERIFIED (09:19) -- full 3-seed chair EMA trio, built against r16

Built `sub_round19_chairema_trio.zip`: all 3 chair members now EMA(0.99) (seed42/7/13), 5 towers
+ bonsai byte-identical to r16. VERIFIED: CRC OK, 386/386 exact, 348.5MB. This is the full-
strength follow-up to r17's diluted null result -- if EMA is real, this is where it should show.
Tower EMA rollout (F4) running on GPU0+GPU1 for the remaining 4 towers, ETA ~11:30-12:00
(includes the makeup jobs for HCM0539/HCM0540 from the bug above).

## BUCKET B/C DIRECTIVE + EXTERNAL LIT-MINING RECONCILED (24/07 ~07:40)
User: r20 is the LAST hyperparameter attempt; r20 renders are BANKED (reuse as ensemble base).
Spend the next 2 days on bucket B (loss/param) + C (architectural) to move >=1 of 3 metrics.
Wise/parallel: compose on the r20 base, decide at ENSEMBLE level, no hours-per-single-scene gates.
User pasted a lit-mining report (sourced from a DIFFERENT project's medical-imaging GS KB -- NOT
ours; flagged, treated as external literature to verify, reconciled vs our own data):

- REPORT #1 blur forward-model (BAD-Gaussians/DeblurGS: K virtual sub-poses per frame, render+avg,
  compare to blurry GT; Gaussians stay sharp). Its own gate: only works for MOTION blur, not DoF.
  RAN THE GATE (free, opencv Farneback on chair train frames): corr(inter-frame motion, render-GT
  L1) = pearson -0.008 p=0.93 / spearman -0.059; worst-fit frames motion 14.9px ~ best-fit 15.4px.
  => chair blur is DoF/defocus, NOT motion-correlated => BAD-Gaussians can't help. KILLED for chair
  without building it. (Also consistent with our H-B post-hoc-matching null.) Saved ~0.5 day.
- REPORT #2 depth-seeded densification in LOW-SfM regions (frozen Depth-Anything unproject ->
  seed Gaussians where SfM is sparse; bonsai glass table 54k pts). DISTINCT from our capacity
  tests (we varied COUNT 5M-16M and reg; never varied WHERE points seed). Rule-10 clean (frozen
  generic depth net). LIVE candidate, bonsai-targeted. Moderate build.
- REPORT #3 specular deferred rendering (3DGS-DR/MSGS/GaussianShader for the glass table).
  Architecturally heavy (deferred shading pass + env map + per-Gaussian reflection params),
  single scene (bonsai=1/7). Too expensive for a 2-day budget. DEPRIORITIZED.
- REPORT #4 ensemble-variance uncertainty diagnostic (K stochastic runs, per-pixel variance:
  high-var=fixable, low-var-high-err=ceiling). We ALREADY have K seed-members per scene -> nearly
  free to compute, but it's a DIAGNOSTIC not a fix, and fidfit+H-B already point to video ceiling.
  LOW priority.

ADOPTED 2-DAY PLAN (compose-on-r20, ensemble-level decisions, both GPUs different tracks):
  Track A (broad, main): Mip-Splatting 3D filter -- ONE build, POST-HOC on existing ckpts (per-
    Gaussian scale band-limit from train-cam sampling rate + opacity renorm), render ALL 7 scenes
    -> new decorrelated member on every scene -> add to r20 ensemble, score whole ensemble ONCE.
    Targets LPIPS+SSIM (anti-aliasing). gsplat 1.5.3 has 2D mip (rasterize_mode=antialiased, we
    use it on video already) but NO native 3D filter -> implement.
  Track B (running): Difix restoration head -- chair FT verdict today; if any life, extend.
  Track C (gated filler): depth-seeded densification, bonsai only, IF a GPU frees.

## R20 BUILT + VERIFIED (07:42) -- last hyperparameter/ensemble round
sub_round20_emaadd.zip: 346.4MB, 386/386 exact, VERIFY PASSED. ADD composition -- 4/5 towers
(HCM0421/0539/0540/0644) get the 5-member ensemble (orig ut7 + ut7_ema999 + 3 seeds), HCM0421
6-member; chair 6-member (aa trio + ema trio); HCM0674 + bonsai unchanged from r16. Baseline
r16=77.0955; expected +0.1-0.2. THE LAST bucket-A round -- everything after is B/C on r20 base.
Mip validation (Track A) launched on freed GPU1 07:43; Difix FT still on GPU0.

## TRACK A (Mip-Splatting 3D filter) -- DEAD post-hoc (24/07 07:55)
Built scripts/mip3d_filter.py (per-Gaussian scale band-limit from train-cam sampling rate +
energy-preserving opacity renorm; math unit-tested clean). Applied post-hoc to chair_ema099
eval ckpt, rendered eval holes, scored:
  baseline 69.8051 | filter_scale 0.2 -> 29.36 (PSNR 8.8) | filter_scale 0.03 -> 33.55 (PSNR 10.4)
CATASTROPHIC at every scale. Root cause (diagnosed, not a bug -- math verified): our MCMC models
are massively over-densified with sub-pixel Gaussians (median opacity x0.0039 at s0.2, x0.0287
even at s0.03 -- i.e. the median Gaussian is far smaller than one pixel). The physically-correct
energy compensation correctly identifies these as aliasing sources and suppresses them, but they
are so numerous and collectively load-bearing that suppressing them blacks out the render. Proper
Mip-Splatting integrates the filter DURING training so the model never forms these spikes;
applied post-hoc to a model trained without it, it just breaks it. Doing it right = retrain all
7 scenes with the filter in the gsplat loop (~21 GPU-h + integration) for a lever usually <0.3
PSNR on in-distribution test poses -- not worth it vs the 2-day budget + compose-on-r20 framing.
Track A CLOSED. Fast decisive reject (~40 min), no hours wasted. Architecture bets remaining:
Difix restoration head (Track B, running GPU0, verdict ~10:20) + depth-seed densification (C).

## R20 GRADED: 77.1903 -- NEW BEST, +0.0948 vs r16, all 3 metrics up
PSNR 26.4738 (+0.0547) SSIM 86.8067 (+0.1582) LPIPS 11.8399 (-0.0364). SCORE +0.0948 --
36x the r17/r18 fragment-swap nulls, cleanly above noise. VALIDATES the add-composition thesis
(strategy audit): composition-class changes transfer at >=1x, UNLIKE correlated single-swaps.
All 3 submetrics moved right, SSIM leading. Achieved with only 4/5 towers EMA'd (HCM0674 missing
after its run went pathological). LB set2: ...-> 77.096 (r16) -> 77.190 (r20). Gap to top-1
(82.17) = 4.98. This is the biggest real gain since r16 and confirms the composition lever still
has headroom (more decorrelated members -> more gain). Open option (user's call, since r20 was
declared the last bucket-A round): completing HCM0674's EMA member -> 5/5-tower r21, est +0.01-0.03.

## BUCKET-C #2: MONOCULAR DEPTH PRIOR (MICCAI endoscopic-GS transfer, 24/07 08:2x)
User steered to the endoscopic/surgical 3DGS literature (MICCAI) for the 85-road. Mined it
(WebSearch/WebFetch, sources logged): the field's DOMINANT technique for our exact problem class
(low-texture, sparse/garbage SfM, poor geometry monocular video = chair/bonsai) is monocular
depth priors. SurgicalGaussian (arXiv 2407.05023): depth-based Gaussian init (GIDM) + L_depth =
||D_pred - D_render||_1 @ weight 0.001 alongside photometric. Endo-4DGS: Depth-Anything depth
regularization. Textureless-Splatter/ISPRS papers: same. PR-ENDO (specular) SET ASIDE -- its
camera-aligned single-light model doesn't transfer to bonsai's room lighting, heavy build, 1/7.
IMPLEMENTED (bucket C, Rule-10 clean -- frozen Depth-Anything-V2 = VGG/AlexNet class):
  train_gsplat --depth_prior <w> --depth_dir <cached>. Renders RGB+ED (gsplat native expected
  depth), matches rendered depth to frozen Depth-Anything-V2 prediction via SCALE-AND-SHIFT-
  INVARIANT Pearson corr (mono depth = disparity, near=large; gsplat depth metric, near=small ->
  correct geometry = NEGATIVE corr -> loss = 1 + corr). Precompute: scripts/precompute_depth.py.
  Syntax-checked, 200-iter smoke passed (loss finite, densify OK).
Precomputed depths: chair 147, bonsai 220 train imgs. Launched chair eval-split full run
(depth_prior 0.05, else = chair_ema099 recipe) on GPU1; Difix still GPU0. Decision: solo vs
baseline 69.8051 AND ensemble-add vs ref mean 71.0682. ~2.2h. WHY this is the right 85-road bet:
genuine bucket-C (new supervision channel + pretrained model), broad (all scenes), multi-metric
(geometry helps PSNR+SSIM+LPIPS), pairs with Difix (depth=geometry, Difix=appearance). Honest
caveat: surgical gains are partly BECAUSE their SfM is worse than ours; towers (150-200k clean
pts) won't benefit much -> real target is the 2 video scenes (2/7 of score).

## ARCHITECTURE VERDICTS (24/07 11:30) -- mixed-to-weak, honest read
CHAIR DEPTH-PRIOR (0.05) DONE: solo 69.1583 vs baseline 69.8051 = -0.65 (worse standalone), but
ensemble-ADD 71.1461 vs ref mean 71.0682 = +0.078 (marginal decorrelated-member gain; moves PSNR
25.75->25.79 +0.041 and SSIM +0.0018, LPIPS flat). Consistent with family-ensemble pattern
(worse-solo-but-additive), but only ~+0.011 blended -> WEAK. A keeper-as-member at best.
DIFIX FT KILLED (stuck): the current run (relaunch #N, started 05:25) printed NOTHING past
"data.json" and wrote ZERO checkpoints/evals in 6h at 100% GPU -- STUCK (the model_1.pkl on disk
is STALE, timestamped 05:22 from an earlier crashed attempt). Same pathological-hold signature as
HCM0674. Killed 11:32, freed GPU0. Difix restoration head now failed operationally TWICE
(zero-shot destructive + fine-tune hangs); prior was already low (renders too clean). SHELVED.
REALLOCATION: bonsai depth-prior launched GPU0 (glass table, 54k SfM = sparsest = depth prior's
strongest-mechanism scene); HCM0181 tower depth-prior continues GPU1 (7min in). Two depth-prior
verdicts (bonsai ~13:00, tower ~13:30) will settle whether depth-prior is worth productionizing
onto r20 or is another marginal lever. Honest state: architecture bets are landing marginal
(chair depth +0.011 blended) or failing (Mip dead, Difix stuck) -- the ceiling evidence mounts,
but bonsai (depth's best case) + tower (5/7, untested assumption) are the two that matter and
aren't in yet.

## 2DGS MOONSHOT BUILT (24/07 ~14:10) -- last untried architecture card
Depth-prior CLOSED: all 3 scenes hurt solo (chair -0.65, bonsai -1.49, tower -0.79), ensemble-add
marginal (chair +0.078, tower +0.010). Mechanism failed because our SfM is NOT as broken as
endoscopic scenes -> Depth-Anything mono depth is LESS accurate than our multi-view geometry, so
the prior fights the photometric fit. Architecture bets now 0/3 (Mip, Difix, depth-prior).
Killed the seeded depth pool (jobs were known-waste post-verdict). Both GPUs redirected to bank
real r21 composition members (the ONE proven lever): chair depth-prior PRODUCTION (the +0.078
member) GPU0, HCM0674 EMA re-run (5/5 towers) GPU1 -- floor ~77.25-77.35.
2DGS: gsplat has rasterization_2dgs (2D surfels) but MCMC doesn't support it -> built a SEPARATE
trainer gsplat_track/train_2dgs.py (DefaultStrategy adaptive-density + normal-consistency +
depth-distortion losses; reuses load_scene; production trainer UNTOUCHED). Syntax + imports
verified. Bet: 2dgs is a MAXIMALLY-DECORRELATED primitive -> even if individually lower-PSNR, it
should ADD to the pixel-mean ensemble (composition transfers >=1x). Probe armed (fires when GPU0
frees ~16:00): 200-iter smoke -> chair eval-split 30k -> score solo + ensemble-add. Decision:
viable member if solo respectable OR ensemble-add positive; else kill fast + consolidate floor.

## R21 BUILT + VERIFIED (16:05) -- composition floor bank
sub_round21_chairdepth_hcm0674ema.zip: 345.9MB, 386/386 exact, VERIFY PASSED. r20 + HCM0674 5/5-
tower EMA completion + chair 7th (depth) member. Baseline r20=77.1903. Expected +0.02-0.06.
2DGS PROBE FAILED: gsplat 2dgs backward "grad.depths must be contiguous" during the 30k run.
Investigating (likely the RGB+ED depth-render or distortion-loss backward path); fixing + relaunch.

## R21 GRADED: 77.2066 -- NEW BEST, +0.0163 vs r20 (user labeled it "r20" but it's r21)
PSNR 26.4851 (+0.0114) SSIM 86.8356 (+0.0289) LPIPS 11.8380 (-0.0019). All 3 metrics up again.
r21 = r20 + HCM0674 5/5-tower EMA completion + chair 7th (depth) member. The +0.0163 is consistent
with completing the proven EMA composition (5th tower) + the marginal chair-depth add. Cumulative
vs r16: +0.1111. LB set2: 77.096 (r16) -> 77.190 (r20) -> 77.207 (r21). Composition floor keeps
rising slowly. Gap to top-1 (82.17) = 4.96. 2DGS video runs (chair/bonsai) still training -- the
last real architecture card for a bigger move.

## MODULE EXPLORATION (24/07 18:14, user Q: "add modules to fastgs+3dgut?")
2DGS video: bonsai CRASHED (CUDA kernel error before step 0, likely 1920x1080 resolution; chair
2dgs at 720x1280 trains fine on identical code). chair 2dgs still running GPU0.
Launched the top module candidate on freed GPU1: 3DGUT with_eval3d extended to VIDEO (chair --ut
k1=0 = 3D-accurate unscented Gaussian eval, zero distortion) -- a decorrelated video member vs
the 2D-EWA antialiased baseline. Directly answers "add more 3DGUT to more scenes". Eval-split,
solo + ensemble-add, baseline chair solo 69.8051 / ens-mean 71.0682. Honest framing given to user:
these module toggles are decorrelated-MEMBER plays (~+0.05-0.15, inch the floor to ~77.4-77.7);
the only STRUCTURAL 80-path module is learned restoration (Difix), which fails operationally --
proposed one disciplined single-process (no accelerate) Difix retry as the real swing.

## USER Q (18:2x): watch-training-loss + custom-loss?
Watch-loss: ALREADY answered by fidfit -- train fit caps ~27dB (towers: capacity/content wall;
video: blur-limited GT), confirmed across standard/pure-L2/reg-off arms. It's a FITTING WALL, not
undertraining -> WHY loss reshaping keeps failing. Watching live 3dgut-chair curve for 60k
convergence only. Custom loss: tested+dead already (metric_loss/texture_weight/uncertainty/pure_l2/
Charbonnier). ONE untested metric-aware angle: LPIPS weight -- the 0.4-weighted metric we keep at
0.1. Higher LPIPS wt trades PSNR(0.3) for LPIPS(0.4), net-+ if ~even; on TOWERS only (sharp, LPIPS
room 0.119->0.06; hurts blur-limited video). Armed LPIPS-weight sweep (HCM0181, lambda 0.3 & 0.5,
lpips_from 30k) gated on chair 2dgs freeing GPU0. Baseline lp0.1 ema999 solo 75.4293.

## 2DGS KILLED (21:00) -- pathological over-densification, impractical
chair 2dgs ran 4h50m, no ckpt, GPU0 mem 96% (15.7/16.3GB) = DefaultStrategy densification ran
away (2dgs has no cap_max like our MCMC; surfel count exploded -> memory-bound crawl). Killed to
unblock the higher-value LPIPS-weight gate. 2DGS verdict: IMPRACTICAL regardless of eval number --
~4h+/scene at 3.7x slower per-iter, bonsai 2dgs already CUDA-crashed (1920x1080), and DefaultStrategy
over-densifies without a budget. Would need densification-cap tuning to be viable; not worth it for
a chair-only marginal-member upside. 2DGS CLOSED. LPIPS-weight sweep (HCM0181 lambda 0.3, 0.5) now
running on freed GPU0. chair 3DGUT-video still on GPU1 (2h46m, near done).

## LPIPS-WEIGHT SWEEP -- NEGATIVE (gates WD-R) [2026-07-25]
HCM0181 eval-split, ema999, lpips_from 30k. Baseline lp0.1 solo = 75.4293.
  lp0.3 -> 75.3301  (-0.099)
  lp0.5 -> 75.2661  (-0.163)
Both HIGHER LPIPS weights scored BELOW the lp0.1 baseline, monotonically worse as weight rises.
VERDICT: the perceptual/LPIPS axis does NOT reward more weight even on the sharp towers (the best
case). The 0.4-weighted LPIPS term is already saturated at lambda 0.1; trading PSNR for it is net
negative. This is the FITTING WALL again (train fit caps ~27dB) surfacing on the metric side:
you cannot buy LPIPS by weighting it harder because the model can't fit the texture it's missing.
=> WD-R GATE IS NEGATIVE. Wasserstein-Distortion loss pushes the SAME perceptual axis harder;
if lambda 0.3/0.5 LPIPS already regresses, a stronger perceptual matcher has no headroom to exploit.
WD-R build NOT green-lit. Last untested loss is now dead. Loss-function avenue CLOSED end-to-end
(metric_loss/texture/uncertainty/pure_l2/Charbonnier/LPIPS-weight all tested, all <= baseline).
Composition (decorrelated members, >=1x transfer) remains the ONLY working lever. r21=77.2066 banked.

## SEED-101 BANK running (GPU1) [2026-07-25 12:37]
Bucket-B decorrelated seed member: fresh seed-101 UT on HCM0181 tower eval-split proxy, production
recipe (60k/8M/rstop50k/noise50k/lpips50k), --ut_render native. Scores solo (vs ema999 75.4293)
+ 2-member add (ema999+s101). ~3h. Purpose: bank a real r22-composable member AND get a hard data
point on whether a 6th decorrelated seed still adds at current saturation (r20->r21 seed/EMA add
was +0.016). GPU0 externally occupied (14.6GB, non-mine process, no visible CUDA app -- Windows/other
session) so single-GPU only, no stacking (HCM0674 contention lesson). Result -> seed101_bank.DONE.

## SEED-101 BANK result [2026-07-25 14:43]
HCM0181 tower proxy, seed-101 UT, production recipe, native render. n=60.
  solo:  PSNR 24.1232 SSIM .8441 LPIPS .1200  SCORE 74.9974  (vs ema999 solo 75.4293; raw seed < EMA-smoothed, expected)
  add (ema999 + s101): PSNR 24.5638 SSIM .8568 LPIPS .1094  SCORE 76.0653  (+0.636 over ema999 solo)
VERDICT: member is HEALTHY and native render path is CORRECTLY ALIGNED (a misaligned member drops
the mean; this lifts it +0.64). BUT this is a 1->2 shallow add, NOT the 5->6 marginal add to the real
production tower ensemble (ut7/ut7_ema999/ut42/ut13/ut77). True production marginal ~+0.01-0.02 by the
saturation taxonomy (r20->r21 deep add was +0.016). Bankable r22 member (guaranteed >=1x). Eval-split
ckpt is throwaway (production members retrain on full data) -> delete ckpt, keep eval_png for record.

## SEED SATURATION CURVE (HCM0181 proxy) [2026-07-25 16:2x]
seed-202: solo 74.9301 / add(ema999+s202) 76.0350 -- mirrors s101 (interchangeable healthy seeds).
Shallow saturation curve (ema999 baseline):
  1 member (ema999)           75.4293
  2 (+s101)                   76.0653   (+0.636)
  3 (+s101+s202)              76.2752   (+0.210)
Seeds are NON-REDUNDANT (2->3 still +0.21). Direction solidly positive; deep 5->6 production marginal
smaller (~+0.01-0.02, r20->r21 precedent) but both seeds worth banking. => productionizing s101 across
the 5 set2 towers (running); s202 as an optional 7th member if r22-with-s101 proves worth building.
Both eval-split ckpts throwaway; kept eval_png only.

## R22 BUILT + VERIFIED [2026-07-26 01:13]
sub_round22_seed101towers.zip: r21 + seed-101 as new equal-weight member on all 5 set2 towers
(HCM0421 6->7 members w=1/7; HCM0539/0540/0644/0674 5->6 members w=1/6; chair+bonsai byte-identical,
copied verbatim from r21 zip bytes, zero re-encode). Base = r20/r21 banked pre-field png_ens,
re-fielded with r2r9/fields/<T>.npy --strict (all provenance-clean, train-only). VERIFY PASSED:
386/386 exact, CRC OK, correct dims, 345.9MB/350MB (HCM0421 q99, rest q100). Expected +0.05-0.15
(saturation-tail composition add, not architecture-class). NOT SUBMITTED -- awaiting user go.

## SOTA LIT SCAN [2026-07-26 ~02:30] -- the metric arithmetic that prunes the field
**Δscore = -40·ΔLPIPS + 30·ΔSSIM + 0.6·ΔPSNR(dB).** LPIPS and SSIM are NEARLY ZERO-SUM: a method
buying 0.04 LPIPS while paying 0.04 SSIM nets only +0.1. THIS is why LPIPS-weight upweighting went
75.43->75.33->75.27, and it kills the whole perceptual-optimization literature for us (they all
trade SSIM for LPIPS ~1:1). Confirms WD-R skip from the paper's own table: Apple WD-R on MipNeRF360
indoor = LPIPS -0.041 (+1.64) but SSIM -0.041 (-1.23) and PSNR -0.51 (-0.31) => net +0.10 for 2.8x
train cost. WD-R FORMALLY CLOSED on arithmetic, not just on our gate.
2nd pruning rule: "Mind the Gap" (2607.01556) measures the interp-vs-extrap gap at 3-12dB and shows
hold-out-every-Nth eval measures NEAR-TRAJECTORY INTERPOLATION. Our test poses are interleaved =>
the entire sparse-view regularizer family (DropGaussian/CoR-GS/FSGS/flat-minima/SH-reduction) is
engineered for the WRONG regime. Explains why everything regularizing away from photometric fit hurt.
LIVE candidates it surfaced (ranked): (1) motion-blur + ROLLING-SHUTTER forward model on the 2 video
scenes -- gsplat 1.5.3 ALREADY ships `rolling_shutter` + `viewmats_rs` (config-only, no CUDA), but RS
requires with_ut=classic and we measured dropping --ut for antialiased = +0.49/scene, so RS must beat
that; legal via pose finite differences (NOT the paper's eval-frame fitting, which is illegal for us).
(2) train-fitted render-space restoration (our restoration head = superset, running). (3) Perceptual-GS
capacity allocation ported to MCMC relocation weights (+0.1-0.3). (4) DENSER (SoccerNet'26 winner):
shared-trunk ensemble BRANCHING from one ckpt = more members per GPU-h than independent seeds; DENSER
also won by finding a TRAIN/EVAL POSE-DISTRIBUTION MISMATCH -> check our tower test-pose distribution.
NTIRE'26 winner corroborates ensembling meta (n=91 averaging on 3DGS-MCMC).
Honest verdict from the scan: +3 is NOT available in 2 days; realistic assembly = +0.5..+1.2 -> 77.7-78.4.
CAVEAT I flag on their item 1: our own 24/07 gate killed motion-blur for CHAIR (Farneback corr of
inter-frame motion vs render-GT L1: pearson -0.008). BUT that gate tested VARIANCE (does motion
*explain error differences*); it recorded ~15px inter-frame motion in BOTH best- and worst-fit frames
-- i.e. motion is uniformly LARGE, so a correlation test is blind to a uniformly-present blur. The
gate does NOT rule out uniform unmodeled motion blur. Re-open via the cheap render-time K-render
row-weighted composite probe on existing ckpts (no retrain).

## *** LEARNED RESTORATION HEAD -- POSITIVE, biggest lever in weeks [2026-07-26 03:0x] ***
Organizer clarified ANY png-producing model is allowed (encoder-decoder/diffusion), so a per-scene
image->image restorer is in-bounds. Rule 10 still held: trained ONLY on TRAIN photos (the eval-split
holdout images are held-out TRAIN photos, never test GT), no external scene imagery, no manual edits.
REGIME POINT (why this isn't the usual overfit trap): a restorer trained on renders at TRAIN poses
would learn to fix an already-near-perfect render (train fit ~27dB) and would not transfer. Every
input here is a render at a HELD-OUT pose, so it learns to undo GENUINE novel-view degradation.
PROTOCOL: 60 held-out poses split 40 fit / 20 test; net never sees the 20. Small residual U-Net
(ch32, ~1.2M params), output zero-init so it starts as EXACT identity; L1 + 0.5*LPIPS on 256px crops,
3000 iters. Input = 3-member pixel-mean ensemble (ema999+s101+s202). Scored with the PROJECT scorer.
  baseline (ensemble only, 20 held-out): PSNR 24.5695 SSIM .8527 LPIPS .1091 SCORE 75.9590
  RESTORED                             : PSNR 24.5926 SSIM .8519 LPIPS .1009 SCORE 76.2742
  ** DELTA +0.3152 **
Decomposition vs the metric arithmetic: -40*(-.0082) +30*(-.0008) +0.6*(+.023) = +0.328-0.024+0.014
= +0.318 (matches). It moves LPIPS WITHOUT paying SSIM -- exactly the property the lit scan says is
the ONLY thing that pays in this metric. Same family as the lens field (post-ensemble train-fitted
render-space transform), which transferred to LB at ~1x with no compression tax.
OPEN QUESTIONS before productionizing: (a) cross-SCENE transfer -- we have eval renders for only
HCM0181/chair/bonsai(+HCM0421 gt); 4 set2 towers have NO eval renders, so productionizing needs
either cross-scene transfer or 8 GPU-h to generate them; (b) does it hold on video scenes;
(c) does more capacity/iters help.

## *** JPEG ENCODE TAX -- RECOVERED IN A GENUINE JPEG (keep_rgb) [2026-07-26 ~04:30] ***
The shipped encode (q100, subsampling=2, optimize, progressive) costs REAL points on the video scenes,
concentrated entirely in LPIPS -- JPEG destroys the fine foliage/grain texture LPIPS-vgg keys on, while
towers (smooth structures + sky) are immune. Measured on bonsai eval renders vs real GT (n=28):
  PNG (lossless ceiling)          71.8829   LPIPS .2448  46.2MB
  q100 ss2 prog  [WHAT WE SHIP]   70.8027   LPIPS .2695  20.1MB   <-- -1.0802 vs ceiling
  q100 ss0 prog                   70.8072   LPIPS .2686  30.8MB   (chroma is NOT the cause)
  q100 ss0 prog keep_rgb          71.7935   LPIPS .2465  45.6MB   (+0.9908 recovered)
  q98  ss0 prog keep_rgb          71.8115   LPIPS .2461  33.7MB   (+1.0088 recovered, SMALLER)
ROOT CAUSE: at q100 the quant table is all-1s, so the residual loss is (a) the RGB->YCbCr->RGB
roundtrip and (b) DCT coefficient rounding. Pillow >=9.5 `keep_rgb=True` stores JPEG in RGB and
removes (a) entirely. ss0 alone does nothing => the damage was almost ALL the colour transform.
** q98 ss0 prog keep_rgb is the winner: +1.0088 on bonsai, only 0.071 below lossless PNG, and 33.7MB
vs 45.6MB at q100. q98 slightly BEATS q100 (mild quantization acts as a denoise LPIPS likes). **
This is a REAL jpeg -- any decoder reads it, NO format spoofing, NO compliance risk, ZERO GPU cost.
Supersedes the "ship PNG bytes under .jpg names" idea entirely (user had approved that risk; not needed).
Blended estimate pending chair+tower numbers; bonsai alone = +1.01/7 = +0.144 blended.

## VIDEO SOURCE PROVENANCE + TAX CALIBRATION [2026-07-26 02:2x]
Before re-encoding the video scenes for r23, verified WHICH png dir actually produced the shipped
bytes (re-encode with shipped settings, compare bytes to r21 zip):
  chair : /mnt/d/avv/r21/video_ens/chair7/png     58/58 BYTE-IDENTICAL  (q100 ss2 prog)  <- TRUE SRC
  bonsai: /mnt/d/avv/r14/bonsai_ens/png            0/28 -- NOT the source (mean|px| diff 1.69!)
          /mnt/d/avv/r16/video_ens/bonsai/png     MATCH q100 ss2 prog=True                <- TRUE SRC
  (searched all 15 dirs on disk with the right filename set; r16 is the only byte-exact match.)
  => rebuilding bonsai from r14 would have SILENTLY REGRESSED it. Always byte-verify the source.
CALIBRATION of the keep_rgb gain (worry: +1.0088 was measured on a 1-member eval render, while
production bonsai is a 3-member mean -- smoother, so possibly less to destroy). GT-free encode
self-distortion (LPIPS between png and its jpeg round-trip):
  bonsai production 3-member  shipped .06581 -> hq .00479   (-92.7%)
  bonsai eval       1-member  shipped .06413 -> hq .00485   (-92.4%)
  chair  production 7-member  shipped .01698 -> hq .00218   (-87.1%)
  chair  eval       4-member  shipped .01758 -> hq .00238   (-86.4%)
Ensemble depth does NOT change the encode damage (.0658 vs .0641) => the +1.0088 estimate is NOT
inflated by the single-member measurement. 92.7% self-LPIPS reduction matches the 93.4% score
recovery measured against GT. Blended estimate (1.01 + 0.13)/7 - ~0.006 tower-quality cost = **+0.157**.

## R23b BUILT + VERIFIED -- encode fix, zero model risk [2026-07-26 02:29]
sub_round23b_jpegfix.zip: r22's EXACT pixels, re-encoded. 348.8MB, 386/386 exact, CRC OK, VERIFY
PASSED, every scene decodes as a real JPEG (fmt=JPEG mode=RGB, correct dims). No format spoofing.
Allocation chosen from a MEASURED tower quality curve, not the ladder's guess:
  tower: q100 76.2865(63.0MB) q99 76.2861(56.1MB) q98 76.2702(47.6MB) q97 76.2304 q96 76.1508
         q95 76.0124 q93 75.6873  => q99 is FREE (-0.0004, -7MB/scene); towers fund the video scenes.
  FINAL: HCM0421 q98, other 4 towers q99, chair+bonsai PINNED q98 ss0 keep_rgb.
Added --hq_quality to build_submission_zip.py: BOTH ladder passes assume higher q == better, which
is FALSE for keep_rgb video (bonsai q98 71.8115 > q100 71.7935, and 26% smaller). r23a had let the
ladder run free -> bonsai q100 / chair q97, worse AND bigger. r23a marked SUPERSEDED.
Expected: +1.106 summed across scenes = **+0.158 blended** over r22 (77.2318). Confidence HIGH.

## RESTORATION EXPERIMENT REDESIGNED (design flaw caught in my own plan) [2026-07-26 02:2x]
The evalgen pool was making 1 member x 5 towers. WRONG AXIS: the restorer is applied to 6-7 member
PRODUCTION ensembles, so training it on single-member renders teaches it to undo much LARGER noise
than an ensemble has => over-correction in production. Matching the INPUT DISTRIBUTION beats scene
coverage. Pre-claimed HCM0539/0540/0644/0674 so the pool skips them; queued HCM0421 seeds 7 + 13 to
join the running seed-42, giving a 3-member eval ensemble on a REAL set2 tower that mirrors the
HCM0181 setup where restoration measured +0.315. This also enables the decisive test: does an
HCM0181-trained restorer improve a DIFFERENT tower's ensemble (cross-scene transfer)? If yes, one
restorer covers all 5 towers with no further training.
OPEN, must measure not assume: the restorer was trained on UN-WARPED ensemble renders against the
true photo, but production applies the lens field before encoding. Since the field corrects a
sub-pixel misregistration and the restorer's target IS the photo, the restorer has probably already
absorbed part of the field's job -> applying both could DOUBLE-CORRECT and eat the +0.73 field win.
Fix: train the restorer on FIELD-WARPED input so the train and inference chains are identical.
Will measure both orderings when the HCM0421 members land.

## *** r23 GRADED 77.2059 -- WORSE THAN r22 (77.2318) BY -0.0259. MY PREDICTION WAS WRONG. ***
[2026-07-26] I predicted +0.158 from the keep_rgb encode fix. The LB says -0.026. Breakdown vs r21
(26.485135 / 86.8356 / 11.838):
   r23 PSNR 26.498611 (+0.0135)  SSIM 86.8948 (+0.0592)  LPIPS 11.9042 (+0.0662 = WORSE)
PSNR and SSIM moved EXACTLY as predicted (keep_rgb preserves more signal). LPIPS moved the WRONG
WAY, and at 0.4 weight it flipped the sign of the whole change.

WHERE MY VALIDATION FAILED -- the lesson, stated plainly:
I measured the tax on a SINGLE-MEMBER bonsai render. I then "calibrated" for ensemble depth by
measuring encode SELF-DISTORTION (LPIPS between png and its own jpeg round-trip) on production vs
eval renders, found them nearly identical (.0658 vs .0641), and concluded the estimate was safe.
THAT CALIBRATION WAS THE WRONG MEASUREMENT. Self-distortion is |encoded - png|. Score impact is
|encoded - GT| MINUS |png - GT|. Those are different quantities: the first says how far the encode
MOVES the image, the second says whether that movement is TOWARD or AWAY FROM the ground truth.
A move of identical size can help or hurt depending on which side of GT the render sits on.

NEW HYPOTHESIS (testing now, chair k=1/2/4 members x 3 encodes vs real GT): a deep pixel-mean
ensemble is SMOOTHER than GT, because averaging destroys the per-view stochastic texture LPIPS-vgg
keys on. q100-ss2 JPEG artifacts inject high-frequency structure that pushes a too-smooth render's
texture statistics TOWARD real GT (LPIPS improves) while pushing a noisy 1-member render past it
(LPIPS worsens). If confirmed: the "JPEG tax" is actually a JPEG BONUS for deep ensembles, our
shipped encode was accidentally load-bearing, and stripping it removed a free perceptual crutch.
This would also reconcile with the earlier grain-injection result (uncorrelated noise hurt) --
JPEG artifacts are STRUCTURED and image-correlated, unlike white grain.
ACTION: r22 (77.2318) remains the best submission. r23/r23b are NOT to be submitted. Do NOT ship
r23b -- it makes the SAME change more aggressively (pins video at the "optimal" keep_rgb) and would
be expected to lose MORE.

## ENCODE-vs-ENSEMBLE-DEPTH DIAGNOSTIC -- explains r23, and CLOSES the encode axis [26/07 02:4x]
chair, real GT, k = 1/2/4 members x 3 encodes (project scorer):
   k=1:  png 69.8051 | shipped 69.4974 (-0.3077) | q98keep_rgb 69.7564 (-0.0487)   kr-ship +0.2590
   k=2:  png 70.7828 | shipped 70.5596 (-0.2232) | q98keep_rgb 70.7539 (-0.0289)   kr-ship +0.1944
   k=4:  png 71.1203 | shipped 71.0094 (-0.1109) | q98keep_rgb 71.1248 (+0.0045)   kr-ship +0.1154
THE JPEG PENALTY SHRINKS MONOTONICALLY WITH ENSEMBLE DEPTH (-0.308 -> -0.223 -> -0.111) and the
keep_rgb advantage decays with it (+0.259 -> +0.194 -> +0.115). At k=4 the "lossless" PNG is ALREADY
no longer optimal -- q98 keep_rgb BEATS it (+0.0045). The optimum is migrating TOWARD compression as
the render improves. Production is deeper still (chair 7 members, full 240-image models, + lens
field), so its optimum plausibly sits at or beyond the SHIPPED q100 ss2 -- which is exactly what the
LB reported when I moved away from it.
ALL GT IS JPEG (towers DJI_*.JPG, video frame_*.jpg, and our eval_gt too), so this is NOT a
"clean GT vs jpeg GT" artifact -- it is a RENDER-QUALITY effect: the noisier the render, the more
JPEG hurts; the cleaner/smoother the render, the more JPEG's structured artifacts substitute for the
per-view stochastic texture that pixel-mean averaging destroyed.
=> ENCODE AXIS CLOSED. The shipped q100 ss2 profile has been implicitly validated across ~10 graded
LB rounds on PRODUCTION-quality renders. Our proxy renders are systematically NOISIER than production,
so any encode optimum measured on the proxy is systematically TOO CLEAN. Do not retune the encode on
proxy data again. r22 (77.2318, shipped encode) stands; r23/r23b are dead.

### THE GENERAL LESSON (applies to the restoration head, which is now at risk)
The proxy's models see 75-83% of the training photos, so proxy renders are systematically noisier
than production renders. ANY technique whose job is to CLEAN UP render noise will therefore measure
too well on the proxy and under-deliver (or invert) in production. The encode fix is the first
confirmed instance. The RESTORATION HEAD (+0.315, measured on a 3-member proxy ensemble) is the same
class of intervention and carries the same risk -- it is trained to remove noise that production has
less of, so it may over-correct. Downgrade confidence accordingly; do not ship it on proxy evidence
alone. Contrast: COMPOSITION (adding decorrelated members) has never transferred below 1x in 10
graded rounds, because it does not depend on the noise level -- it is a variance-reduction argument,
not a cleanup argument. Composition remains the only lever with LB-proven transfer.

## *** ADVERSARIAL VERIFICATION OF THE RESTORATION HEAD -- 5 agents, verdict: DO NOT SHIP ***
[2026-07-26 03:0x] Three independent skeptics attacked the +0.3152. Summary of what survived:

STATISTICS ATTACK **FAILED** (the number is not noise): the comparison is PAIRED, so image-to-image
variance cancels. Measured paired SE at n=20 across 6 real treatments = 0.009-0.065; fitted
sd = 0.368*effect^0.60 gives SE 0.041 at effect 0.315 => **3.2-7.7 sigma**. n=20 is ample (n~10
suffices). Chair's -0.0703 is only ~1.6 sigma => chair is UNINFORMATIVE, not evidence of harm.
Redirect: the unmeasured variance is the FIT-SET lottery and TRAINING SEED (both n=1), plus orbit
autocorrelation (n_eff = 21 of 60), plus the restored renders were never saved so nothing is auditable.

LEAKAGE ATTACK: temporal-adjacency leak REFUTED on towers (nearest fit frame is a median 7.91 deg
away; photo-to-photo PSNR 8.57dB vs 24.57dB render-to-own-photo -- they are NOT nearly the same
picture). Split reconstructs exactly, fit/test disjoint, no GT leak into the gaussian models,
"it's just an unsharp mask" REFUTED (unsharp 0.15 = -0.1356, 0.30 = -0.4953). BUT confirmed:
GroupNorm makes the DEPLOYED function != the TRAINED function (trained on 256 crops, inferred on
full 989x1320 frames; residual differs 62% relative L2), RNGs for weights/crops are UNSEEDED so the
run is not reproducible, and the param count was 0.47M not the 1.2M I logged.

*** THE KILLER (transfer lens), confirmed by direct measurement: ***
The restorer's input is UN-FIELDED; every production tower render IS fielded (field_applied.json
stamp present in r20/r21/r22 tower png dirs, absent from the restorer's input).
   RAW (restorer input) PSNR 24.6837 SSIM .8602 LPIPS .10779
   + FIELD              PSNR 25.5675 SSIM .8809 LPIPS .10731   => +1.170 pts
(the +0.8838dB reproduces the historical D9 HCM0181 number to 3dp -- airtight.)
HONEST CORRECTION the agent made against its own thesis: the field is +1.15 on PSNR/SSIM but only
+0.019 on LPIPS, and the restorer's gain is 96% LPIPS -- so the field does NOT re-explain +0.315.
Double-warp risk is also DEAD (the net learned only 1.5% of the warp). What DOES survive:
 (a) the restorer's side-effects (SSIM -0.0008) were measured on a MISREGISTERED image, where
     synthesized high-frequency detail cannot conflict with aligned true detail. On a registered
     render that SSIM cost will not stay at -0.0008, and this metric is near zero-sum.
 (b) production tower LPIPS is ~0.0757 vs the 0.1078 the restorer trained on -- **30% cleaner**.
     Gap = field 1.17 + ensemble depth 3->7 ~0.25 + data starvation (180/240) ~1.7.
 (c) THE ONE DISTRIBUTION-MATCHED TRIAL IS THE NEGATIVE ONE: chair ships UN-fielded and its
     restorer input was un-fielded => matched => -0.0703. Tower was mismatched => +0.3152.
     The sorting is exactly the wrong way round for the optimistic reading.
 (d) IN-HOUSE EXISTENCE PROOF of this exact failure mode: Difix zero-shot -- a restorer trained on
     LPIPS-0.33-class degradation, applied to our cleaner renders, was "strictly destructive at
     every strength, LPIPS itself WORSE at every strength". Same shift, larger magnitude.
This is the SAME mechanism that killed r23 (proxy renders are noisier than production, so any
cleanup-class intervention over-measures on the proxy). Restoration is cleanup-class.
=> RESTORATION NOT SHIPPED on current evidence. One cheap decisive test remains: retrain on a
FIELD-CORRECTED input (production-matched on the field axis) and see if the gain survives on a
registered render. Running it; prior is now low.

## *** MIP-SPLATTING 3D FILTER (during training) -- +0.6596 PAIRED, ALL THREE METRICS UP ***
[2026-07-26 ~04:40] HCM0181 proxy, seed 101, identical recipe (--ut 60k cap8M rstop/nstop 50k
lpips_from 50k), filter baked into the ckpt so the stock renderer needs no change. PAIRED vs the
seed-101 bank run:
  plain seed101   PSNR 24.1232  SSIM .8441  LPIPS .1200  SCORE 74.9974
  + mip3d 0.2     PSNR 24.3742  SSIM .8506  LPIPS .1121  SCORE 75.6570   ** +0.6596 **
  (baked filter: median sigma 0.00488 world units, mean opacity .1308)
PSNR +0.251, SSIM +0.0065, LPIPS -0.0079 -- ALL THREE IMPROVE SIMULTANEOUSLY. That is the crucial
difference from the restoration head (96% of whose gain was a single LPIPS-vgg movement): this is a
FIDELITY gain, not a judge-directed one, so it does not carry the fragility of a pure-LPIPS result.
It is also a MODEL change (better geometry) rather than a post-hoc cleanup of renders, which is the
class that keeps failing to transfer.
The earlier Mip rejection (24/07) was POST-HOC application to a model already full of sub-pixel
spikes -- it blacked out the render. Its own postmortem said "proper Mip-Splatting integrates the
filter DURING training so the model never forms these spikes", and that version had never been run.
Mechanism: band-limit every gaussian to the sampling rate of the nearest TRAIN camera, so gaussians
that no view can constrain (the median gaussian was FAR below one pixel) never form. Those spikes
are precisely the novel-view noise that pixel-mean ensembling was paying +1.82 to cancel.
CAVEAT (the r23 lesson): still a proxy measurement, and a data-starved proxy has MORE unconstrained
gaussians than production, so the benefit may shrink at 240 images. Validating cross-scene on
HCM0421 (paired vs its 75.7616 baseline) before spending 10 GPU-h productionising.
BONUS: a mip3d member is a NEW MODEL FAMILY, so it can be ADDED to the existing ensembles rather
than replacing them -- composition, the one class that has never transferred below 1x.

## RESTORATION DEPTH TEST -- does NOT decay like the encode did
HCM0181, seeded/reproducible, k = ensemble depth:
  k=1  75.0757 -> 75.4830   DELTA +0.4072
  k=2  75.7542 -> 76.0399   DELTA +0.2857
  k=3  75.9590 -> 76.2509   DELTA +0.2919
Drops k=1->2 then PLATEAUS (unlike the encode fix, which decayed monotonically 0.259->0.194->0.115
and inverted in production). Seeded k=3 (+0.2919) reproduces the original unseeded +0.3152 within
~0.02, giving the training-seed variance component that was previously unmeasured.
So restoration is real on the proxy and depth-stable. What still stands against it: input was
UN-FIELDED while production is fielded, production tower LPIPS is ~30% cleaner than it trained on,
the one distribution-MATCHED trial (chair) was negative, and Difix is an in-house existence proof of
this exact failure. Restoration stays UNSHIPPED pending the fielded test; mip3d now outranks it.

## RESTORATION -- DATA-STARVATION TEST (the last standing objection) [26/07 13:1x]
Same restorer, k=1 (single member), two scenes at DIFFERENT training-data density:
  HCM0181  180/240 = 75% data   DELTA +0.4072
  HCM0421  200/240 = 83% data   DELTA +0.2735 (s0) / +0.2740 (s1)  13/13 improved, seeds agree to 0.0005
THE GAIN SHRINKS AS TRAINING DATA GROWS: +8pp of data cost -0.133. Naive linear extrapolation to
100% data (production) lands the unfielded k=1 gain at roughly ZERO.
Offsetting: fielding ADDS ~+0.14 (HCM0181 k=3 unfielded +0.292 -> fielded +0.430) and production IS
fielded. Very rough production estimate: ~+0.1, with the sign not firmly established.
CONFOUNDS (do not over-read): the two points are different SCENES, different fit-set sizes (40 vs 27)
and different test-set sizes (20 vs 13), not a clean data-fraction sweep.
VERDICT: restoration is REAL on the proxy (depth-stable, fielding-robust, 20/20 and 13/13 per-image,
reproducible across seeds, not an unsharp mask, no leakage) but its transfer to production is
UNRESOLVED and trending DOWN with data density -- the exact mechanism that made r23 lose. NOT
recommended for r24 on current evidence. Composition (bonsai members) and mip3d (a fidelity gain,
all three metrics up) are the better bets.

## R24 BUILT + VERIFIED -- bonsai 3 -> 6 members [2026-07-26 15:53]
sub_round24_bonsai6.zip, 344.9MB, 386/386 exact, CRC OK, VERIFY PASSED.
SINGLE-VARIABLE build: towers+chair copied BYTE-IDENTICALLY from the r22 zip; build asserts
"28 files differ from r22, in scenes: ['bonsai']". Clean comparison vs r22 = 77.2318.
bonsai = 6-member mean (aa42/aa7/aa13 existing + aa101/aa202/aa303 new), all on the full 248-image
train set, r16 capD recipe, no --ut (antialiased), no field. Shipped encode q100 ss2 (NOT keep_rgb).
PRE-BUILD PROVENANCE CHECK: mean(aa42,aa7,aa13) reproduces the shipped bonsai 28/28 PIXEL-IDENTICAL
(max abs diff 0) -- so the 3 new members extend exactly the ensemble on the leaderboard.
Expected +0.04..+0.06 blended. Confidence HIGH (proven-transfer lever class).

## MIP3D VALIDATED ON A SECOND SCENE -- r25 greenlit
  HCM0181 (180/240 data)  74.9974 -> 75.6570  +0.6596   (PSNR +.251 SSIM +.0065 LPIPS -.0079)
  HCM0421 (200/240 data)  75.7616 -> 76.2188  +0.4572   (PSNR +.191 SSIM +.0064 LPIPS -.0038)
Paired, same seed/recipe, ALL THREE METRICS UP BOTH TIMES => fidelity gain, not a judge-directed
LPIPS movement. This is the key difference from the restoration head (96% single-metric).
SAME STARVATION TREND AS RESTORATION though: +8pp training data cost -0.20 of the gain. mip3d is a
REGULARIZER (it removes degrees of freedom no view can constrain), and regularizers help most when
data is scarce -- so expect materially less than +0.457 at production's 240 images.
=> DECISION: productionise mip3d as an ADDED 7th/8th ensemble member, NOT as a replacement. Adding a
decorrelated member is a COMPOSITION change (never below 1x in 10 rounds); if mip3d's quality edge
survives at full data we capture it too, and if it evaporates we still hold a normal decorrelated
member and cannot go backwards. Seed 555 (existing: 7/42/13/77/101). 5 towers x ~3h = ~7.5h on 2 GPUs.
OPS FIX: the two pool workers now stagger by 90s -- this morning bonsai-303 and the mip3d validation
both saw GPU0 free at 05:55/05:56 and raced into it, contending for 5.5h while GPU1 sat idle.

## *** ARITHMETIC CORRECTION -- "the video scenes ARE the gap" WAS WRONG *** [26/07 16:3x]
I have been repeating "video LPIPS deficit x0.4 weight x2/7 scenes ~= 3.7 points ~= exactly the gap
to top-1". That is an arithmetic error (inherited from an agent report, repeated by me unchecked).
   100 * 0.4 * 0.13 * (2/7) = **1.486**, not 3.7.
Closing the video LPIPS gap ENTIRELY (chair .220->.10, bonsai .240->.10) buys **1.49** of the 4.94
points needed. Verified against r23's graded submetrics, which reconstruct to 77.2059 exactly.
Also: +4.94 from PSNR alone would need +8.23dB overall; from LPIPS alone it would need dLPIPS
= -0.1235 when LPIPS IS 0.119 (i.e. negative -- impossible).
=> THE GAP IS SPREAD ACROSS ALL 7 SCENES AND ALL 3 METRICS. Top-1's towers are almost certainly
better than ours too (ours: LPIPS .0757, PSNR ~26.5 -- neither is a ceiling). The "towers are done,
only video matters" framing was licensing us to ignore 5/7 of the weight. RETRACTED.

## FABLE CONSULT -- three framing corrections + the one production-side gradient we own
1. (above) the gap is fidelity-shaped and spread, not video-LPIPS-shaped.
2. The H-B blur bound did NOT kill deblur-consistent training. blur_bound.py fits a global kernel
   mapping OUR CURRENT RENDER -> photo, which bounds POST-HOC correction. Deblur-consistent TRAINING
   is a different mechanism: with chair's 7.6x within-scene sharpness spread the fit receives
   INCONSISTENT SUPERVISION, so the converged scene is texture mush + floaters (bonsai's fog collapse
   was exactly this), and no global kernel applied to mush recovers anything. H-B was structurally
   guaranteed to read flat even if in-training blur modelling is worth points. Same blindness as the
   Farneback gate (uniform ~15px motion is invisible to a variance test).
3. "LPIPS/SSIM are zero-sum" is a property of ONE INTERVENTION CLASS, not of the metric. Judge-
   directed moves (sharpen/blur/WD-R/restoration-head) are zero-sum; FIDELITY moves pay on all three
   at once -- our own lens field (+PSNR +SSIM, LPIPS flat) and mip3d (+all three, twice) prove it.
   The zero-sum framing was pushing us toward LPIPS-only tricks, exactly the class the audit refused.
4. *** NEW INTERVENTION CLASS: TEXTURE INJECTION, which UNDER-measures on the proxy. ***
   Cleanup-class over-measures (r23/EMA/restoration/mip3d-as-quality). GT-matching measures ~1x
   (lens field). Texture injection is the mirror image: noisy proxy renders need LESS added texture
   than smooth production ensembles, so the proxy UNDER-states it.
   r23 is an LB-GRADED PRODUCTION-REGIME MEASUREMENT that our shipped renders are TOO SMOOTH:
   removing JPEG's structured high-frequency cost us 0.026 ON THE LEADERBOARD. That is the only
   production-side gradient we own and it points where the proxy discourages us from going.
   => P1 experiment: out = mean + lambda*HP(member_j - mean). Structured, image-CORRELATED texture
   (unlike white grain, which was catastrophic). SIGNATURE TEST: for this class the gain must GROW
   with ensemble depth k -- the opposite of the cleanup signature. Proxy-optimal lambda is then a
   LOWER bound on the production optimum, so shipping proxy-lambda is under-dosed but safe-signed.

## R24 GRADED: 77.2528 -- NEW BEST, +0.0210 vs r22 (77.2318)
PSNR 26.521433  SSIM 86.9119  LPIPS 11.8342.
SINGLE-VARIABLE round (towers+chair byte-identical to r22, asserted at build time), so the whole
+0.0210 is bonsai 3 -> 6 members = **+0.147 on the bonsai scene** (x7).
Predicted +0.04..+0.06 blended; delivered +0.021 -- POSITIVE and correctly signed, but ~half the
estimate. My bonsai 3->6 projection of +0.3..+0.4/scene was extrapolated from TOWER marginals
(2->3 seeds +0.232, 3->4 +0.142); bonsai's curve is flatter than the towers'. Recalibrate: at
6 members a further video-scene member is worth ~+0.05/scene = +0.007 blended. Video composition
is now as saturated as the towers.
COMPOSITION SCORECARD (still the only never-negative class, 11 graded rounds): +1.82, +0.90, +0.23,
+0.166, +0.142, +0.095, +0.025, +0.021, +0.016 -- monotonically decaying but never once negative.

## ADVISOR ROUND 2 -- long-schedule hypothesis KILLED with evidence I had missed [26/07 21:3x]
Q: is the 27dB train wall a convergence artifact? ANSWER: NO, and our own log already tested it.
 (a) TAIL: exp26/exp31b are warm-start tail-extensions with means_lr ~30x TERMINAL for 8-10k extra
     noise-free steps => train +0.52dB MAX, test +0.00, score flat-to-down. If 27dB were
     not-converged, a 10k extension at 30x terminal lr would have moved far more. Tail is NOT binding.
     (Mechanism: means-lr decays 0.01^(step/iters) -- at 50k of 60k it is already 2.15% of initial --
     but scales/quats/opacities/SH lrs are CONSTANT and converge fine in 10k Adam steps.)
 (b) NOISE schedule: tested twice, dead (exp20 vs exp19 = 74.5067 vs 74.5072 exact tie; chair
     noise_stop 40k flat). Also MCMC noise is scaled by the passed lr, so by 50k it is at 2% amplitude.
 (c) EXPLORATION phase (densify/relocate) is the ONLY live component -- that is what 30k->60k (+0.32)
     actually bought. That rung was never re-run. Prediction for 120k: proxy median +0.10..+0.15,
     80% band [-0.10,+0.30], P(negative) 20-25% (bonsai's pA_60k already REVERSED at one doubling).
Q: does the +0.80 train->test slope hold WITHIN scene? NO. That slope is CROSS-scene (difficulty
   drives both variables; our own r13 audit stamped it "carries no causal information"). Within-scene
   interventional points: exp31b train +0.52 -> test +0.00 (slope 0); exp21 opacity_reg down, train up
   -> test DOWN (negative). Honest within-scene prior: 0.2-0.4. Pattern: gains from better optimising
   VIEW-CONSTRAINED structure transfer ~0.3; gains from freeing UNCONSTRAINED dof ANTI-transfer.
=> 120k arm re-armed anyway (auto-GPU, race-safe double-check gate) because it is cheap, it is the
   only un-run rung, and IT IS ITS OWN OVERFIT DETECTOR: scored on held-out eval holes, so it measures
   within-scene transfer directly. Read: eval >= +0.1 => productionise; train +>=0.4dB with eval
   flat/down => overfit onset, kill without spending a production GPU-minute.
SKIP: SH degree 3->4 (test poses are trajectory interpolations; degree-4 view-dependence is not our
failure mode). Densification variants (min_opacity/aniso/absgrad/opacity_reg all dead by record).
Bilateral/AA (eps2d dead both directions, bilagrid dead, PPISP dead).
FREE WIN I WAS ABOUT TO SKIP: re-sweep the ensemble WEIGHT for the mip3d member BEFORE building r25,
not after. Uniform weighting was measured dead for SAME-family members, but the historical +0.23 came
from FAMILY weighting, and mip3d is a different family AND measurably better solo (+0.457). Script
mip_weight_sweep.py sweeps w_mip on the HCM0421 proxy where both members have real GT.
HONEST ENVELOPE (advisor, and I agree): r25 +0.10..0.25, 120k +0.05..0.15, deblur +0.1..0.3 at ~30%
odds, free sweeps +0.05 => landing 77.4-77.8, 90th-percentile parlay ~78.1. +4.9 is unreachable.
HARD RULE ADOPTED: reserve the LAST 3 HOURS for render/ensemble/zip/verify. A busted final zip is the
only way to LOSE points from here.

## R25 GRADED: 77.2954 -- NEW BEST, +0.0426 vs r24 (77.2528)
PSNR 26.548794 (+0.0274)  SSIM 86.9671 (+0.0552)  LPIPS 11.8101 (-0.0241, better).
ALL THREE METRICS UP -- the fidelity signature mip3d showed on both validation scenes. Arithmetic
checks: -40*(-.000241) + 30*(.000552) + 0.6*(.0274) = +0.0426 exactly.
Predicted +0.10..+0.25; delivered +0.0426 (~1/3 of the low end). THE STARVATION CAVEAT WAS RIGHT and
I should weight it harder next time: proxy showed +0.457 (HCM0421, 200/240 data) and +0.660 (HCM0181,
180/240), production delivered a fraction of that because production models see 240 images and mip3d
is a REGULARISER -- regularisers help most where data is scarce. Add the w=0.2 dilution into a
6-member ensemble and +0.04 is about what the mechanism predicts in hindsight.
Transfer ledger for the record: mip3d proxy->production realised ~0.1x as a diluted added member.
COMPOSITION SCORECARD (12 graded rounds, still never negative): +1.82, +0.90, +0.23, +0.166, +0.142,
+0.095, +0.043, +0.025, +0.021, +0.016.

## *** RESTORATION HEAD -- DEFINITIVELY DEAD (production-regime evidence) *** [27/07 09:xx]
Using the newly-calibrated PRODUCTION harness (public_set: models trained on 240/240, REAL test poses,
REAL test GT; harness 3->4 marginal +0.0919 vs LB +0.142, deviation 0.050 => calibrated):
 (1) SAME-SCENE, production quality: 77.1105 -> 77.4234 = **+0.3128**, 19/20 improved. This REFUTED
     the data-starvation objection that had benched it. Restoration was briefly back.
 (2) CROSS-SCENE (train on HCM0181, apply to 3 other public towers -- the only legal shipping route,
     since set2 test GT can never be used for training):
        HCM0193  PNG -0.2371 | after shipped JPEG -0.3856
        HCM0204  PNG -0.1524 | after shipped JPEG -0.2983
        hcm0034  PNG -0.1701 | after shipped JPEG -0.3401
     NEGATIVE ON ALL THREE, both as PNG and after the encode.
CONCLUSION: the restorer is SCENE-SPECIFIC -- it memorises one scene's texture statistics and actively
harms a different scene. And the JPEG encode makes it WORSE STILL (-0.24 -> -0.39), exactly as the
production encode result predicts: JPEG's structured artifacts are LOAD-BEARING at production quality
(shipped q100ss2 BEATS lossless PNG, 76.6937 vs 76.6745), and the restorer destroys the very texture
JPEG was supplying. Two mechanisms, same direction.
=> RESTORATION CLOSED. r27-as-restoration is off the table. Cost of finding out: ~1 GPU-hour of
inference. Worth it -- the same-scene +0.3128 was tempting and would have been a losing submission.

## ENCODE AXIS -- CLOSED PROPERLY (production regime, not inferred from one failed round)
PRODtower (4-member production ensemble, real test poses/GT):
  PNG lossless 76.6745 | q100 ss2 [SHIPPED] **76.6937** | q100 ss0 76.7031
OUR SHIPPED ENCODE BEATS LOSSLESS PNG. JPEG's structured, image-correlated artifacts substitute for
the per-view stochastic texture that pixel-mean averaging destroys. r23 (-0.026) was not bad luck --
moving toward a cleaner encode HAD to lose. q100 ss0 is +0.0094 better still but costs +24.7MB/scene
(+124MB total), which the 350MB budget cannot fund without degrading elsewhere. Axis closed.

## *** THE +0.7 PER-VIEW FIELD ORACLE IS A MIRAGE -- 3D-LIFT BRANCH DEAD FOR FREE *** [27/07 10:0x]
The per-view oracle field measured +0.66..+0.80 partial-score (= competition points, since partial =
0.6*PSNR + 30*SSIM), ~6x what our global field captures. It was the largest unclaimed prize on the
board and would have justified a multi-hour 3D-lift build (lift residual flow to 3D via rendered
depth, pool by surface point, re-project at test poses).
BUT the oracle fits view i's field using DIS flow between render_i and photo_i and then applies it to
view i -- it is fitted ON what it is scored on. SPATIAL HOLD-OUT TEST (HCM0421, n=25, score only the
BOTTOM half of each frame):
     none              44.6115   (-0.44 vs global)
     global_med        45.0538   baseline
     perview_tophalf   43.3041   -1.7497
     perview_offset    43.9091   -1.1447   <- CORRECTED variant: keeps the global field's spatial
                                              structure, adds only the per-view OFFSET estimated on
                                              the held-out-fitting half. Isolates predictability.
     perview_oracle    45.9256   +0.8718   <- fitted ON the scored rows
The per-view deviation does NOT generalise even to the OTHER HALF OF THE SAME IMAGE. The oracle is
fitting each view's own DIS-flow noise realisation, which NO legal predictor can reach -- not a 3D
lift, not interpolation, nothing. (I first ran a confounded version that also discarded the global
field's spatial structure, giving -1.75; the corrected isolation still reads -1.14. Same verdict.)
=> 3D-LIFT / per-view field DEAD. Cost: ~20 minutes. Saved: a multi-hour build chasing a mirage.
=> The GLOBAL field is the whole prize, and the median estimator (r26) is how we improve it.

## R26 BUILT + VERIFIED -- median-refit lens fields [27/07 09:20]
sub_round26_medianfield.zip, 344.1MB, 386/386 exact, CRC OK, VERIFY PASSED.
SINGLE-VARIABLE vs r25: identical pre-field ensembles (already carrying mip3d at w=0.2), identical
encode, chair+bonsai byte-identical. Build asserts only the 5 towers differ. ONLY the field estimator
changed: per-pixel arithmetic mean -> per-pixel median over the flow stack.
Refit field magnitudes are TIGHTER than the mean-fitted ones, as the outlier-rejection story predicts
(HCM0421 mean|d| 0.1112 -> 0.1013 px, max 1.007 -> 0.838 px).
Validation (leave-one-view-out, project scorer incl. LPIPS): HCM0421 +0.0625, HCM0539 +0.1082,
LPIPS improving slightly in both. Expected +0.05..+0.06 blended. Zero GPU.

## *** R27 BUILT -- the warp was resampling with the wrong kernel *** [27/07 12:52]
sub_round27_lanczos.zip, 345.1MB, 386/386 exact, VERIFY PASSED, NOT SUBMITTED. 4 min, zero GPU.
ONE FLAG: fit_field.apply_field's cv2.remap goes INTER_CUBIC -> INTER_LANCZOS4. Same median fields
(byte-identical .npy, not refit), same ensembles, same encode, chair+bonsai byte-identical.

The field warp is the LAST operator to touch every shipped pixel and we had been paying a bicubic
resampling tax on all of them. Round-trip isolation (warp +f then -f, no GT, so it measures
destruction only), production renders:
    INTER_CUBIC 47.3 dB / 0.951 Laplacian energy kept | LANCZOS4 51.8 dB / 0.976 | spline5 52.6/0.976
Bicubic was destroying 5% of the detail per frame.
PRODUCTION HARNESS vs REAL test GT (5 public towers, 290 imgs, full project scorer):
    PNG +0.1095 | after the shipped q100/ss2 JPEG +0.1078 | 5/5 scenes in BOTH encodings
    per-scene post-JPEG: 0181 +0.1337  0193 +0.1127  0204 +0.0909  0031 +0.0952  0034 +0.1062
PSNR, SSIM and LPIPS all improve TOGETHER in every scene -- pure fidelity, not a judge-directed
trade. Private-set LOVO (150 held-out views) independently agrees: +0.0980.
EXPECTED +0.05..+0.08 blended (5/7 scenes carry a field). r26's harness over-prediction (31%
transfer) has a cause that does not apply here: r26 changed the ESTIMATOR and private fields pool
120 views vs the harness's 60, so the mean was already half as noisy in production. A resampling
kernel does not care how many views the field came from. And HCM0181 -- the only harness scene using
a 4-member pixel-mean ensemble, i.e. the production regime -- shows the LARGEST gain.
Rejected on the same evidence: spline5 (+0.0024 over lanczos, inside noise, new dependency),
gaussian-smoothed field + lanczos (+0.0017), ds16 (-0.0007). Kernel of the FIELD UPSAMPLE measured
irrelevant (+/-0.0002) and stays cubic.

## NEXT: can the warp stop paying the resampling tax ENTIRELY? [27/07 12:5x, running]
lanczos4 still destroys 2.4% of the HF energy. Zero-padding the spectrum is an EXACT band-limited
upsample; on a k-fold fine grid an ordinary kernel is only asked about frequencies up to 1/k of its
own Nyquist, where it is flat. Round-trip isolation (n=4 production renders):
    cubic 47.3 dB / 0.951 | lanczos4 51.8 / 0.976 | spline5 52.6 / 0.976
    fft2+cubic 55.2 / 1.019 | **fft2+lanczos 62.2 / 0.9948** | fft4+cubic 58.5 / 1.017
fft2+lanczos destroys 0.5% instead of 2.4% -- a step of the same size as the one that just paid
+0.108. (fft*+cubic reads HF>1: cubic's negative lobes OVERSHOOT on an oversampled grid, which is
ringing, not detail; the round-trip dB is the honest discriminator and lanczos wins it.)
Cost 0.49 s/img CPU. Harness (post-JPEG, 290 imgs) launched -> res_harness_fft.json.
IMPLEMENTATION NOTE for anyone reusing bandlimit.py: zero-padding a shifted spectrum must ALIGN DC
(index N//2 for both parities), not centre the block. Centring is off by one bin when the padded
size is odd and the fine size even; one bin is a linear phase ramp, and it does not degrade the
image subtly -- the first run read 6.85 dB / 0.29 HF and looked like a decisive negative result.

## BAND-LIMITED WARP -- MEASURED NEGATIVE, AND IT CLOSES THE KERNEL AXIS [27/07 13:2x]
Production harness, post-JPEG, 290 imgs, vs the shipped median_ds8 cubic baseline:
    med_ds8_rlan [r27]  +0.1078   (5/5)
    med_ds8_fft4cub     +0.0771   (5/5)
    med_ds8_fft2lan     +0.0699   (5/5)
The near-ideal warp is WORSE than lanczos4 despite destroying 5x less HF energy (0.5% vs 2.4%).
Ruled out as a bug first: zero-field identity is exact (120 dB, max|d| 0.00000) for all three
kernels, and the residual shift between the lanczos and fft2lan outputs is 0.001 px, so there is no
sub-pixel bias. The outputs differ by 0.9/255 RMS -- a real kernel difference, not an error.
MECHANISM: a render's high-frequency content is only PARTLY signal. The rest is 3DGS noise,
floaters and rasterisation aliasing that does not match the photo. A mildly attenuating kernel
denoises it for free; a near-ideal kernel faithfully preserves it.
=> there is an INTERIOR OPTIMUM in warp sharpness. Fitting the three measured points
(cubic 0.951 -> +0.000, lanczos 0.976 -> +0.108, fft2lan 0.9948 -> +0.070) puts the peak at
HF-kept 0.978 worth +0.1088, i.e. **+0.001 over what r27 already ships**.
KERNEL AXIS CLOSED at INTER_LANCZOS4. Do not spend more time here; also do not "sharpen the final
image", the same curve says we are already past the point where more HF helps.
Cost of the closure: ~35 min CPU. bandlimit.py kept for the exact spectral resampler (unit-tested:
k-fold upsample reproduces the original samples to 5e-7).

## *** R27 GRADED 77.4024 (+0.0743 over r26) -- AND THE HARNESS IS NOW SUBMETRIC-PREDICTIVE ***
r22 77.2318 -> r24 77.2528 -> r25 77.2954 -> r26 77.3281 -> **r27 77.4024**. Five straight positives,
+0.1706 this session, no regression since r23.
r27 delivered 96% of the full-transfer estimate (+0.077). The production harness predicted every
submetric, not just the total:
    LPIPS  predicted -0.12   delivered -0.105
    SSIM   predicted +0.064  delivered +0.066
    PSNR   predicted +0.019  delivered +0.021
All three inside 15%. This also CONFIRMS the r26 transfer diagnosis: r26 under-delivered (31%)
because it changed the field ESTIMATOR and private fields pool 120 views vs the harness's 60, so
production's mean was already less noisy; a resampling KERNEL is view-count-independent and
transferred ~1x. => harness deltas may be booked at ~1x for view-count-independent operators, and
haircut for estimator/pooling changes.

## PER-IMAGE EXPOSURE / WHITE BALANCE -- DEAD, ORACLE IS TOO SMALL AND UNPREDICTABLE [27/07 14:2x]
Hypothesis: a drone runs auto-exposure/auto-WB, 3DGS has no per-image photometric parameter so it
fits the AVERAGE, and every test photo is offset from it -- a global low-dimensional error costing
PSNR, SSIM (luminance) and LPIPS at once. The filenames carry a capture sequence number and test
frames are INTERLEAVED (40/60 private test frames have both immediate sequence-neighbours in train),
so a legal temporal predictor looked available.
ORACLE (per-image photometric map fitted against real test GT -- diagnostic only), HCM0181, n=60,
production render + median field + lanczos + shipped JPEG:
    base 78.1999 | global scalar gain +0.0647 | per-channel gain +0.0706 | gain+bias +0.0652
    fitted gain 1.0025 +- 0.0128 (range 0.960..1.036)
The whole prize is +0.07 with a PERFECT oracle. Not a mirage (3-6 dof fitted from 1.3M pixels
cannot noise-fit) -- just small.
AND IT IS NOT PREDICTABLE: leave-one-out interpolation of the gain from sequence neighbours has
|err| 0.0121 against a |deviation from mean| of only 0.0099 -> explains **-21%**, i.e. interpolating
is WORSE than assuming the mean. Auto-exposure jitter is white in time, not a drift.
A scene-wide CONSTANT gain (legal, fittable on train photos) captures only the systematic 0.25% of
the 1.28% total, i.e. (0.0025/0.0128)^2 = 3.7% of +0.065 = +0.0026. Negligible.
=> EXPOSURE AXIS CLOSED. Cost: ~7 min.

## ENSEMBLE MEMBERS ARE ALREADY MUTUALLY REGISTERED -- no blur to reclaim from pooling [27/07 14:0x]
Hypothesis: the pixel-mean ensemble averages members that each have their OWN sub-pixel geometry, so
it low-passes itself. Measured member-to-mean DIS displacement on HCM0181's 4 production members
over 60 real test poses:
    pooled (fixed) mean|d| 0.0096 / 0.0070 / 0.0125 / 0.0089 px
vs the render-to-PHOTO field's ~0.10 px, the one worth +0.73 on the LB. An ORDER OF MAGNITUDE
smaller. The per-view component is 0.071-0.087 px but pools to ~0.01, i.e. it is uncorrelated across
views (DIS noise or genuine per-view jitter, not a fixed misregistration).
=> pooled-field member alignment cannot pay; the systematic part does not exist. Per-view alignment
arm scored separately (aligned_pv) as the upper bound.

## CHAIR FIELD RE-GATED -- the median FLIPS its sign, but the prize is ~+0.004 blended [27/07 14:5x]
The chair field was zeroed on 17/07 (held-out x-val dPSNR -0.127, blamed on "DoF/RS flow noise").
That gate used the MEAN estimator, CUBIC remap and leg-1 30k/5M renders -- all three since
superseded, and the median estimator is specifically an outlier-rejector for the failure mode that
was blamed. Honest re-test, LOVO n=50 on current production renders (r2r9 chair_ut42, 103 train
pairs), project scorer incl. LPIPS, vs NO FIELD:
    mean + cubic  [the 17/07 config]   **-0.2035**   <- original kill REPRODUCED
    median + cubic                     +0.0215
    median + lanczos                   +0.0207
    median ds16 + lanczos              +0.0234
    median + gauss(1) + lanczos        **+0.0300**   <- best
So the 17/07 verdict was right about that estimator and wrong about the field. The mechanism is
confirmed: median rejects the flow outliers, mean swallows them. But chair is 1/7 of the score, so
+0.030 = **+0.004 blended** -- below LB noise. NOT worth a round on its own; bundle only if a chair
rebuild happens for another reason. bonsai stays dead on stronger evidence (its clean train-KP
GEOMETRY field earned +0.0006, so it is not an estimator problem: glass-table reflections move with
viewpoint and there is no fixed field to find).
Note the field-smoothing result inverts vs towers (g1 = +0.0017 there, +0.0084 here), consistent
with the chair field being genuinely noisier.

## PHOTO-REUSE / IBR NOW DEAD ON THE VIDEO SCENES TOO -- measured, not inherited [27/07 15:3x]
D3 killed IBR using a TOWER (nearest train pose 11.8 deg, paste-nearest 9.4 dB vs render 24.5) and
the verdict was then applied to all 7 scenes. The video scenes are a genuinely different capture --
measured today with train poses restricted to those that actually have a photo on disk (images.bin
carries the test poses too, and including them makes every nearest-distance read exactly 0.00):
    scene      nearest train pose        as % of scene radius
    HCM0421    0.563  ( 8.46 deg)        16.5%
    HCM0674    0.652  ( 9.89 deg)        16.6%
    chair      0.271  ( 3.56 deg)         7.4%
    bonsai     0.273  ( 4.01 deg)         7.1%
    HCM0181    0.551  (10.58 deg)        13.5%
2.4x closer in angle. So the probe was re-run in the video regime: hold out every 4th chair TRAIN
frame (its nearest remaining frame is then ~5 video frames away = the spacing a real test frame
sees), Rule 10 clean, no test imagery:
    paste nearest train photo   PSNR 14.3469  SSIM .3846  LPIPS .4609  SCORE 41.7100
    our production render       PSNR 28.0466  SSIM .8852  LPIPS .1431  SCORE 77.6606
**-35.95.** A 13.7 dB deficit at 3.57 deg. Flow correction cannot close 13.7 dB; the tower verdict
holds set-wide, now on direct video-regime evidence rather than extrapolation. IBR / RIFE / FILM /
frame-interpolation CLOSED for all 7 scenes. Cost: 5 minutes.
INCIDENTAL BUT IMPORTANT: that 28.05 dB is a TRAIN-view number (chair_ut42 trained on all 205
frames). Our train fit is ~28 dB while top-1 scores 31-32 PSNR on TEST. They beat our TRAINING fit
on held-out views -- that is a model-class gap, not a regularisation or post-processing gap, and it
is not closable in the time remaining. Confirms the strategy: bank composition, not moonshots.

## CORRECTION to the member-alignment entry above -- I MEASURED THE WRONG THING [27/07 16:5x]
My memalign run concluded "members are already mutually registered, no blur to reclaim". A campaign
agent measuring the same lever independently got **align-then-merge = +0.0617 in the full production
chain** (+0.0505 isolated on a 7-member pool, +0.0463 and +0.0297 on two disjoint 4-member pools),
with the control that settles it: an identical Lanczos remap driven by a ZERO flow lands at exactly
+0.0000 on all three pools, so the gain is alignment and not the resampler.
BOTH ARE RIGHT ABOUT DIFFERENT THINGS, and the reconciliation is exact:
  - the agent measured GLOBAL translation between members at 0.007 px. I measured the pooled
    member-to-mean field at 0.0070-0.0125 px. Same number. There is no systematic offset.
  - the disagreement is 0.150 px mean / 0.32 px p90 of purely LOCAL geometry, and it destroys
    8-12% of gradient energy in the pixel mean.
MY ERROR: I built the per-view field at ds8 (INTER_AREA down, cubic up). A 1/8-resolution field
cannot represent a local 0.15 px deformation -- it smears the correction across 8x8 blocks, which
is why my aligned_pv arm read -0.0493. The pooled-vs-per-view decomposition I ran is only valid for
the part of the signal that survives 8x downsampling, and the entire lever lives below it.
LESSON, and it generalises: a null result from a band-limited probe is a null about the BAND, not
about the lever. Both the field work and this one used ds8 because the LENS field is genuinely
smooth; member disagreement is not, and I carried the resolution assumption across without checking.

## *** ENSEMBLE ENERGY RESTORATION -- THE BIGGEST LEVER OF THE CAMPAIGN, NEARLY LOST *** [27/07 17:1x]
When the verification workflow was stopped to free GPUs, 8 of its agents had already reported. TWO
of those reports were SHIP verdicts that had never been read. Read them; both are real.

LEVER: averaging k members destroys local HF energy wherever they disagree, and the deficit is the
ensemble's OWN DISAGREEMENT MAP. Harness diagnostic: the pixel mean sits 15% below GT energy at
Laplacian level 0, and levels >=1 are ALREADY at GT energy -- all headroom is in the finest band.
    r   = sqrt(1 + (k/(k-1)) * E(L0_i - L0_mean) / E(L0_mean))   clamped to 4, E = 3x3 box of
          the RGB-summed square
    out = mean + lam * (r - 1) * L0(mean)          inserted BETWEEN the ensemble and the lens field
REPRODUCED INDEPENDENTLY by me through the full shipped chain (mean -> restore -> median field
lanczos4 -> JPEG q100/ss2), HCM0181, 60 real test poses, real test GT. My base 78.1995 vs the
agent's 78.1992 -- agreement to 0.0003:
    lam 0.5  +0.1413 | lam 0.75 **+0.1756** | lam 1.0 +0.1883 | DEV-1 lam0.75 +0.1587
WHY THIS IS NOT THE SHARPENING LEVER THAT ALREADY DIED: the MSE-optimal global unsharp was -0.618
and the best radial LINEAR filter is bounded at +0.043 (power-spectrum agent: 96% of our missing HF
is INCOHERENT, only 3.95% is amplitude mismatch). This clears that bound because it is not a linear
filter -- it is spatially varying and driven by the members, landing where averaging destroyed
energy (corr(r, log E_mean) = -0.63), not on edges.
CONTROLS (production harness, real test GT):
    real map +0.1988 | same histogram SHUFFLED **-0.9766** | same histogram ordered +E **-2.4921**
    | same mean boost, no map -0.0870 | structure-only (no ensemble info) +0.1313
ANTI-CLEANUP SIGNATURE: gain GROWS with depth -- k=2 +0.099, k=3 +0.165, k=4 +0.199, k=5 +0.232,
k=6 +0.244. EMA (+0.484 proxy -> +0.0028 LB) and the JPEG "fix" (+0.158 -> -0.026) both DECAYED
with depth before inverting. Production ships 7 members, so the k=4 number is a LOWER BOUND.
Cross-scene eval-split, positive at every lambda: tower +0.226, chair +0.536, bonsai +0.221 (lam1.0).
Eval-split -> production regime factor measured 0.88, so it is not proxy-inflated.

THE BINDING CONSTRAINT IS THE 350 MB ZIP, NOT THE SCORE. Measured byte growth is wildly
scene-dependent (6 images/scene, shipped encode):
    lam        0.4     0.5     0.75    1.0     1.25
    tower   1.0035  1.0051  1.0100  1.0155  1.0206
    chair   1.0135  1.0190  1.0338  1.0504  1.0671
    bonsai  1.0569  1.0759  1.1215  1.1713  1.2271     <- 15x the towers' byte cost per unit lambda
r27 splits 298.64 MB towers / 28.71 chair / 17.69 bonsai, so uniform lam=0.75 = 350.88 MB and does
NOT FIT. Allocating as a knapsack in blended-score per MB: chair lam0.75 0.053, towers lam0.75
0.042-0.048, bonsai lam0.5 0.009. **Bonsai gets nothing** and its bytes go to the towers.
Checked the alternative: towers 0.5 + chair 0.75 + bonsai 0.5 fits (348.88 MB) but scores +0.165
blended vs **+0.177** for towers 0.75 + chair 0.75 + bonsai 0. Building the latter.

## *** BUDGET CORRECTION: THE CAP IS 350 MiB, NOT 350e6 BYTES *** [27/07 18:0x, user-supplied]
The organiser's 350MB is 350*1024*1024 = **367,001,600 bytes**. We had been enforcing the decimal
reading (350,000,000) in scripts/verify_zip.py since round 1.
    cap       367,001,600 B = 350.00 MiB
    r27       345,092,353 B = 329.11 MiB
    headroom   21,909,247 B =  20.89 MiB   <- we thought it was 4,907,647 B. **4.5x more budget.**
verify_zip.py now measures MiB and prints both readings.
CONSEQUENCE: byte budget is directly convertible into score (it is what caps the energy-restoration
lambda), and the constraint is now SLACK. Every allocation up to lambda=2.0 on all seven scenes fits
(363,171,575 B = 346.35 MiB). Lambda is therefore chosen by SCORE, not by bytes.
This retracts the knapsack in the entry above: bonsai was starved to lambda=0 purely because it
looked 6x byte-inefficient, and that reasoning is void. The half-built lambda 0.75/0.75/0 zip was
killed before it could bake in the wrong allocation.
Re-running the lambda sweep to 2.0, plus align-then-merge composition arms, to pick the score
optimum now that bytes no longer bind.

## LAMBDA OPTIMUM = 1.0, AND ALIGN-THEN-MERGE IS SUBSUMED BY IT [27/07 15:0x]
Production harness, HCM0181, 60 real test poses, real test GT, FULL shipped chain (mean -> operator
-> median field lanczos4 -> JPEG q100/ss2). Bytes measured on the same encode.
           arm     SCORE   d(base)      dMB   score/MB
          base   78.1995   +0.0000   +0.000        --
         e0.75   78.3751   +0.1756   +1.048     0.1675
        **e1.0   78.3877   +0.1883   +1.391     0.1354**   <- OPTIMUM
         e1.25   78.3815   +0.1820   +1.721     0.1058
          e1.5   78.3565   +0.1570   +2.044     0.0768
          e2.0   78.2699   +0.0704   +2.674     0.0263
         align   78.2442   +0.0448   +0.046     0.9825
    align+e0.5   78.3688   +0.1693   +0.756     0.2241
    align+e1.0   78.3955   +0.1960   +1.452     0.1350
    align+e1.5   78.3484   +0.1489   +2.109     0.0706
TWO CONCLUSIONS.
(1) lambda=1.0 is the peak and the curve falls off hard by 2.0 (+0.070). The old 350e6 byte cap was
    never the binding constraint on lambda -- the SCORE was. Correcting the cap to 350 MiB did not
    unlock a higher lambda, it just removed a constraint that happened to sit near the optimum.
    r28 was already building at towers lam=1.0, which is correct.
(2) ALIGN-THEN-MERGE IS SUBSUMED. Alone +0.0448 (the agent measured +0.0617). Stacked on e1.0 it
    adds only **+0.0077**. They are substitutes, not complements, exactly as predicted: both recover
    the SAME level-0 energy deficit, one by removing misregistration blur (real structure) and one
    by rescaling the band (synthesised amplitude), and aligning first shrinks the disagreement map
    that energy restoration feeds on.
    => NOT SHIPPING align. +0.008 does not justify the pipeline risk, and it would require
    reconstructing the exact tower member weighting -- which I probed and it is NOT a uniform mean
    (best 7-member uniform reconstruction of r22's png_ens is 0.201/255 off, so r22's "5 1"
    weighting composes non-uniformly through the nested base means). Not a clean job for +0.008.
NOTE ON REGIME: the eval-split tower optimum was lam=1.25 (+0.228); the production optimum is
lam=1.0. The peak shifts DOWN by about one notch from eval-split to production. Chair's eval-split
peak is 1.25 and bonsai's is 1.0, so their production optima are probably 1.0 and 0.75. r28 ships
chair 1.25 / bonsai 1.0, i.e. one notch high on both -- the curves are flat near the top (chair
lam1.0 +0.536 vs lam1.25 +0.566 eval-split) so this costs ~0.004 blended. Not worth a rebuild.

## R28 BUILT + VERIFIED -- energy restoration, all 7 scenes [27/07 15:05]
sub_round28_energy.zip: 386/386 exact, CRC OK, VERIFY PASSED. NOT SUBMITTED.
    357,078,727 bytes = **340.54 MiB** = 357.08 MB decimal.  Cap 350 MiB = 367,001,600 B.
    HEADROOM 9.46 MiB.
lam: towers 1.0 (harness optimum), chair 1.25, bonsai 1.0. Disagreement map samples EVERY available
member per scene (towers 6 of 7 incl. mip3d, chair 7 of 7, bonsai 6 of 6).
*** THIS ZIP EXCEEDS 350e6 BYTES. *** It is only shippable under the MiB reading of the cap, which
is user-supplied and has never been tested against the organiser -- every prior round fit under
350e6 by accident. Flagged to the user before submission; decimal-safe fallback (towers 0.5 /
chair 0.5 / bonsai 0, ~348 MB decimal, ~+0.10) is one rebuild away.
Built into a FRESH dir after the first attempt (lam 0.75, 4-member map) was killed mid-flight by the
budget correction and left three towers of stale PNGs on disk -- a build writing into a dirty
directory can silently assemble a zip mixing two lambdas, which is invisible in the output and
unattributable on the leaderboard.
EXPECTED +0.13..+0.19 blended over r27 (77.4024). Zero GPU.

## MINED THE REMAINING AGENT REPORTS -- resolutions, kills, and one big pointer [27/07 15:2x]

### THE ALIGN CONTRADICTION, RESOLVED
Two agents reported opposite verdicts on align-then-merge:
    ab1c7db7  align = +0.0617 full chain; inter-member LOCAL deformation 0.150 px mean / 0.32 p90
    a49e8c88  align = +0.0001; inter-member shift 0.0041 px (k4), 0.0069 px (k6) -- "DEAD"
They are measuring DIFFERENT QUANTITIES and both are right. a49e8c88 measured and corrected a
per-member global TRANSLATION (0.004 px, three orders below a pixel -> nothing to gain, +0.0001).
ab1c7db7 used DENSE LOCAL flow (0.150 px) -> +0.0617. My own run agrees with both halves: pooled
field 0.0096 px (global, matches a49e8c88), dense per-view +0.0448 (local, matches ab1c7db7).
=> a49e8c88's conclusion "no alignment can recover it, the HF deficit is genuine per-member content
disagreement" OVERREACHES: it is true of translation, false of local deformation. But its framing is
still useful, because the part that IS genuine content disagreement is exactly what energy
restoration monetises instead of trying to align away. Decision unchanged: align not shipped
(+0.008 marginal on top of lam=1.0).

### THE BIGGEST UNCLAIMED POINTER: MEMBERS 4->6 = +0.2608
a49e8c88, free from its k6 control: adding gsplatB8pure + e17visnorm to k4 moves the production
harness 76.6487 -> 76.9095, **+0.2608 for two members**, consistent with the calibrated +0.0919
3->4 marginal. Its words: "still the highest-yield axis available". This directly validates the
r28_members training pool (chair/bonsai mip3d, then tower mip3d seed 777) as the right GPU spend.

### METRICS DISAGREE ABOUT WHERE OUR ERROR IS (a31aff, and it is actionable)
    region                    %px    %of squared error   %of LPIPS
    near strong edge (<=3px)  27.1        **67.9**          24.1
    far from edge (>12px)     34.7          10.0          **39.0**
    sky                        2.6           0.5     3.6 (density 1.377, the highest of any region)
Per-VGG-layer: relu2_2 (25.4% of LPIPS) has flat-region density 1.259 vs 0.607 at strong edges.
LPIPS's per-location feature normalisation makes small absolute errors in SMOOTH areas expensive.
=> PSNR lives at edges, LPIPS lives in flat regions and sky. Energy restoration boosts where members
DISAGREE, which is textured/edge content -- so the flat/sky half of the LPIPS budget is untouched by
everything we ship. That is a genuinely complementary axis and it has never been probed.

### KILLS AND CONFIRMATIONS FROM THE SAME REPORTS
- Sharpening/deconvolution DEAD BY PROOF: the HF amplitude ratio <|K|/|G|> is 0.98-1.00 in EVERY
  ensemble-variance decile, max MSE reduction 0.09%. Our amplitude is already optimal everywhere,
  including variance-adaptive versions. (Energy restoration is not caught by this: it does not
  correct amplitude toward GT globally, it restores band energy the MEAN destroyed.)
- Chroma 4:2:0 subspace projection DEAD (-0.020). Corollary worth keeping: the shipped
  q100/ss2-beats-PNG win is NOT chroma subsampling (that alone is -0.02) -- it must be the LUMA DCT
  structure, which independently corroborates "structured artifacts substitute for texture".
- Trajectory transfer of a per-view sub-pixel translation: honest +0.0177 same-scene, cross-scene
  mean +0.0075 (3/4 positive) against a +0.0505 oracle. Correct sign, too small. Side finding that
  matters: the pure-TRANSLATION component of the lens field is worth -0.0002, so the shipped median
  field's entire value is in its non-translational dense part.
- REGISTRATION ORACLE LADDER: dense per-view flow +1.5488 (the mirage), local residual sigma=4
  +0.7463, low-freq sigma=32 +0.2168, global affine colour +0.0638, global rigid shift +0.0011,
  best radial linear filter +0.043. Mean residual displacement AFTER the shipped field is 0.51 px
  and it is LOCAL. Registration = 28.0% of all squared error, 45.4% of edge squared error.
  Flow decomposition: view-CONSISTENT part 0.2813 px = 22.9% of flow energy; view-SPECIFIC residue
  0.3833 px = 77.1% (unreachable). Shipped mean-ds8 field captures 35.9% of the oracle gain; the LOO
  median captures 41.4% -- and ds=16 through ds=1 are IDENTICAL, so field resolution is worth
  exactly zero. Confirms our own ds sweep.

## R28 DECIMAL-SAFE COMPANION BUILT [27/07 15:24]
sub_round28_energy_decimalsafe.zip -- same energy-restoration operator at lam towers 0.5 / chair 0.5
/ bonsai 0. 348,347,364 B = 332.21 MiB = 348.35 MB. Fits the DECIMAL cap with +1.65 MB margin.
386/386 exact, CRC OK, VERIFY PASSED; asserts bonsai byte-identical and exactly 5 towers + chair
changed. ~+0.14 expected vs ~+0.19 for the MiB build.
WHY BOTH EXIST: the MiB build is 357.08 MB decimal. If the organiser enforces decimal MB it is
rejected outright, and a rejected submission is strictly worse than a smaller gain. Every prior
round fit under 350e6 by accident, so the boundary has never been tested. User decides.
BUG FOUND AND FIXED: one heredoc patch to build_r28_energy.sh silently never applied (trailing
space in the search string), so SKIP_SCENES was never inserted. Latent at lam=1.0 (nothing skipped);
it only surfaced at bonsai lam=0 when the assembler went looking for PNGs that were deliberately
never produced. LESSON: string-replacement patching of build scripts fails SILENTLY -- the build ran
45 minutes of correct restore work before dying at the assembler. Verify the patch landed, not just
that the file parses.

## FLAT-REGION BAND ATTENUATION -- DEAD, AND IT FALSIFIES ITS OWN PREMISE [27/07 ~16:0x]
Motivated by the error decomposition: 35% of pixels far from any edge carry only 10% of squared
error but **39% of LPIPS**, and sky has the highest LPIPS density of any region (1.377). Everything
we ship acts on textured content, so that 39% is untouched. Hypothesis: in flat regions the mean's
finest Laplacian band is mostly 3DGS noise (GT sky really is smooth), so attenuate it there.
Operator = the natural completion of energy restoration, same band, opposite sign:
    out = mean + [ lam*(r-1) - mu*f ] * L0(mean),  f = 1/(1 + E(L0_mean)/median(E(L0_mean)))
Production harness, HCM0181, 60 real test poses, full shipped chain:
            arm     SCORE     PSNR    SSIM    LPIPS   vs mu=0
   lam1.0_mu0.0   78.3877   25.8879   .8883   .0948   +0.0000     <- r28, the optimum
   lam1.0_mu0.1   78.3547   25.8971   .8878   .0954   -0.0331
   lam1.0_mu0.2   78.2837   25.9022   .8869   .0966   -0.1040
  lam1.0_mu0.35   78.0930   25.9019   .8849   .0999   -0.2948
   lam1.0_mu0.5   77.7827   25.8924   .8818   .1052   -0.6051
   lam0.0_mu0.2   77.9420   25.9373   .8848   .1041   -0.4458   <- attenuation alone
MONOTONICALLY NEGATIVE, and **LPIPS ITSELF GETS WORSE** -- the metric the operator was designed to
improve. Attenuating the flat-region band buys a sliver of PSNR (25.888 -> 25.902) and pays far more
in SSIM and LPIPS.
=> THE PREMISE IS FALSE. The mean's finest band in flat regions is NOT mostly noise. The
39%-of-LPIPS-in-flat-regions finding is real but is NOT caused by excess HF energy there, so it is
unreachable by ANY band-gain operator. Axis closed rather than left as an untested maybe.
Also confirms lam=1.0, mu=0 is the joint optimum of the two-parameter family -- what r28 ships.
NOTE it does not contradict the interior-optimum result from the warp kernel: lanczos's attenuation
is UNIFORM and tiny; this is targeted and large. Different operators, different answers.
OPS LESSON: the first run of this probe rebuilt every member's Laplacian pyramid inside all 6 arms
(6x redundant) and starved both GPUs to 0% util between training steps. Hoisting the shared work
out cut it to ~4 min. Also, `pgrep -f flatband.py` MATCHES THE POLLING SHELL ITSELF (its own command
line contains the string), so every "still running" reading was a false positive -- use
`pgrep -x python -a | grep name`.

## R29 MEMBERS: 4 OF 7 LANDED, ALL GATED, ONE NaN SCARE RESOLVED [27/07 ~19:5x]
mip3d seed-777 members (towers) / seed-555 (chair, bonsai). Collapse gate (member_gate.py, no GT --
compares local detail, mean, std, dead frames against the peers it would join; validated in both
directions earlier: passes a known-good member, rejects a synthetic fog control at 0.00001):
    member    detail    peer range        floor     verdict
    bonsai   0.00058   0.00061-0.00078   0.00038    PASS
    chair    0.01206   0.01352-0.01403   0.00988    PASS
    HCM0421  0.03033   0.03034-0.03395   0.02095    PASS
    HCM0539  0.02997   0.02995-0.03206      --      PASS
All four land slightly SOFTER than every peer, which is the expected mip3d band-limiting signature,
and all sit far above the collapse floor.

*** NaN SCARE ON HCM0539 -- INVESTIGATED, COSMETIC ***
Its bake printed "median sigma 0.00448407, **mean opacity nan**" where the other three printed
0.0381 / 0.0945 / 0.1147. Likely a scale underflow in mip3d_apply's sqrt(s2.prod/d2.prod) giving
0/0 for a handful of gaussians. Checked the OUTPUT rather than trusting the statistic:
  - scanned all 60 renders: 0 anomalous-black, 0 saturated-white, 0 near-constant frames
  - per-image brightness tracks the existing mip3d member to 0.088/255 mean (max 0.37)
  - deviation from the 7-member mean: mip777_NEW 1.6657/255 vs mip555 1.6609 -- essentially
    IDENTICAL, and both LESS than every plain member (ut7 2.156 ... ut77 2.062)
The two mip3d members cluster as one family exactly as they should; a member that had silently
dropped gaussians would be an outlier, not a twin. SAFE TO INCLUDE.

BONSAI SATURATION CHECK before adding its 7th member (its map already ran hot):
    r28 (6 members)  r_mean 2.041  3.27% of pixels at the r<=4 clamp
    r29 (7 members)  r_mean 2.123  4.50% at clamp
Still 95.5% unsaturated, so the map stays spatially selective rather than degenerating toward the
"no map" control (-0.087). Keeping bonsai at lam=1.0 rather than pushing it.
Remaining: HCM0540, HCM0644 training; HCM0674 queued. ~1h out. Building r29 once all 7 have landed
so it changes every scene and is therefore independent of which r28 variant is chosen as its base.

## *** R28 GRADED 77.5029 (+0.1005) -- AND THE MiB CAP IS CONFIRMED *** [27/07]
r22 77.2318 -> r24 77.2528 -> r25 77.2954 -> r26 77.3281 -> r27 77.4024 -> **r28 77.5029**.
Six straight positives, +0.2711 this session.
*** THE 357,078,727-BYTE ZIP WAS ACCEPTED. The cap is 350 MiB = 367,001,600 B, NOT 350e6. ***
We had enforced the decimal reading since round 1 and were throwing away 17 MB of budget. Settled
empirically now, not just by assertion. The decimal-safe companion build was not needed.

SUBMETRIC DECOMPOSITION -- the gain is ALL LPIPS, and SSIM went the WRONG WAY:
                     predicted      delivered
    LPIPS            -0.47 pp       **-0.447 pp**   (95% transfer, essentially perfect)
    PSNR             -0.053         -0.078
    SSIM             **+0.11 pp**   **-0.105 pp**   <- sign flip
    check: 0.1786 (LPIPS) - 0.0314 (SSIM) - 0.0467 (PSNR) = +0.1005 exactly.

WHERE THE SHORTFALL CAME FROM. Harness says towers alone at lam=1.0 give +0.1883 * 5/7 = +0.134
blended (+0.118 at the measured 0.88 regime factor). Delivered +0.1005. So chair+bonsai contributed
roughly **-0.02 to -0.03 blended -- NET NEGATIVE**.
This is the risk I logged when choosing them and shipped anyway: the production optimum sits about
one notch BELOW the eval-split optimum (towers: eval-split peak lam1.25, production peak lam1.0),
and I shipped chair at 1.25 and bonsai at 1.0 -- one notch high on both -- on the reasoning that the
curves are flat near the top. They are flat on towers. They are NOT on the video scenes, and
bonsai's disagreement map was already saturating (4.5% of pixels at the r<=4 clamp).
=> LESSON: the eval-split -> production optimum shift is NOT a small correction on scenes whose
disagreement map runs hot. Do not transplant an eval-split lambda to a scene with no
production-regime measurement of its own.
=> R29: towers keep lam=1.0 (harness-validated AND now leaderboard-confirmed); **chair and bonsai
get lam=0** -- no energy restoration at all -- plus their new mip3d members. That should recover the
~0.025 the video scenes are currently giving away, on top of the composition gain from the members.

## *** DIVERSITY-OVER-DEPTH REFUTED BY DIRECT MEASUREMENT *** [27/07 23:0x]
I had read the agent's "+0.2608 for two different-family members" datum as evidence that ensemble
DIVERSITY is what pays, and recommended spending r30's ~7 GPU-hours on non-UT tower members.
Tested it directly: HCM0181 4-member UT pool, add each available variant as a 5th member, full
shipped chain (restore lam=1.0 -> median field lanczos4 -> JPEG q100/ss2), n=60 real test poses.
              arm     SCORE    vs k4   decorr/255
       k4_UT_only   78.3877   +0.0000        --
      +e17visnorm   78.6325   +0.2448      5.3366
       +e15ceil95   78.6265   +0.2387      5.2976
    +gsplatB8pure   78.5676   +0.1799      5.0796   <- the NON-UT one, ranks THIRD
  +m31b_taillpips   78.5139   +0.1262      4.1040
   +gsplatB7ppisp2   78.4864   +0.0986      5.7542   <- MOST decorrelated of the good ones, LEAST gain
 +gsplatB6bilagrid   77.6459   -0.7418     27.7989   <- wildly different, catastrophic
corr(decorrelation, gain) = **-0.985** over all six; excluding the bilagrid outlier it is **+0.18**,
i.e. NO RELATIONSHIP. Decorrelation does not predict the gain.
=> WHAT PREDICTS THE GAIN IS MEMBER QUALITY, NOT MEMBER DIFFERENCE. The +0.2608 datum helped
because those members were GOOD, and I attributed it to their being DIFFERENT. My error.
=> r30 = more members on the BEST KNOWN RECIPE (3rd mip3d, 60k/8M UT), which is what was originally
planned. Do NOT spend GPU-hours on a non-UT family.
=> bilagrid at -0.74 from ONE member at 1/5 weight is exactly why member_gate.py exists: a bad
member does not average out, it drags the mean hard.
CAVEAT ON MAGNITUDE: these are 4->5 marginals. Production towers are at k=7 going to 8, where the
1/k curve is far flatter, so the +0.10..+0.24 here does NOT transfer to r29/r30 directly.

## *** R29 BUILT + VERIFIED -- field gain 1.30 + 7 new members + video lam=0 *** [28/07 04:33]
sub_round29_members.zip: 349,771,217 B = 333.57 MiB = 349.77 MB. 386/386 exact, CRC OK,
VERIFY PASSED, all 7 collapse gates PASSED, all 7 scenes changed. NOT SUBMITTED.
FITS BOTH CAP READINGS (350 MiB with 16.4 MiB spare, 350e6 with 0.23 MB spare) -- setting the video
scenes to lam=0 shrank them enough that the decimal question does not arise this round.

THE HEADLINE FIND: **a train-fitted lens field systematically UNDERSHOOTS by ~30%.** At train poses
the model has already absorbed part of the misregistration into its own geometry, so DIS sees only
the unabsorbed remainder; at NOVEL poses the absorbed warp does not cancel and the full displacement
appears. STRUCTURALLY INVISIBLE to every train-side protocol we own -- our LOVO says the optimum is
1.00 and is contradicted wherever real test GT exists. Scored route: gain 1.30 = +0.1328 (41/45,
all three submetrics up together). Scoreless geometric route: residual 0.5082 px, view-consistent
part 0.2812 px, shipped field only 0.2112 px, best alpha = 1.3185 (R2 .901 -> .957), and alpha* =
1.32-1.43 on all five public towers. Verifier replicated with the repo's own code at +0.1036 +-
0.0128 (t=8.09) and cut the blended expectation to +0.05..+0.07. Plateau 1.25-1.40; 1.60 = -0.05,
1.80 = -0.13, so 1.30 is the conservative argmax. Rule 10 clean (fields still train-fit; only a
scalar comes from public test GT, the same surface that set lanczos4 and lam=1.0).
=> THIS IS THE SECOND TIME TODAY THE PRODUCTION HARNESS OVERTURNED A TRAIN-SIDE CONCLUSION. The
first was the video-scene lambda. Train-side protocols are now formally distrusted for anything
that interacts with novel-view geometry.

CAMPAIGN COST/YIELD: 19 agents, 2.42M subagent tokens, 1457 tool uses, 2h15m. 10 completed,
9 died on the session limit (resets 03:40) including the synthesis step and 5 idea lenses
(diversity, jpegalloc, geography, lpipsdirect, fieldrefit). Resumable from cache via
Workflow({scriptPath:'.../final-campaign-wf_491d6216-8ef.js', resumeFromRunId:'wf_491d6216-8ef'}).
The one landed find is worth more than the nine that died.

EXPECTED r29 77.60..77.66. r28 = 77.5029.

================================================================================================
r29 GRADED: **77.6644** (PSNR 26.637417  SSIM 87.3084  LPIPS 11.2765)   [28/07]
================================================================================================
+0.1615 over r28 (77.5029) -- the LARGEST single-round jump of the campaign, at the TOP of the
predicted 77.60..77.66 band. Submetric decomposition closes EXACTLY:
    0.6*(+0.130468 dB) + 30*(+0.003412) - 40*(+0.000480)
  = +0.078281   +0.102360   -0.019200  =  +0.161441   vs +0.1615 graded.

READING THE SPLIT:
- FIELD GAIN 1.30 TRANSFERRED AND OVER-DELIVERED. Harness forecast per tower was PSNR +0.105 dB /
  SSIM +0.00207 / LPIPS -0.00019; scaled 5-of-7 scenes that is +0.075 dB and +0.0015 SSIM. We got
  +0.130 dB and +0.0034. The verifier's haircut (0.89 production factor, private fields 18% weaker)
  was too conservative. Field amplitude is now LB-CONFIRMED, not just harness-confirmed.
- LPIPS WENT BACKWARDS (+0.048pp) and it carries the -40 coefficient. That is the video lam=0
  change doing exactly what it was designed to: it traded ~0.019 of LPIPS for ~0.10 of SSIM. Right
  trade, but chair and bonsai now ship with NO perceptual operator and NO field at all.
- Production harness has now called THREE consecutive rounds correctly (r27, r28, r29).

PROGRESSION r22 77.2318 -> r24 77.2528 -> r25 77.2954 -> r26 77.3281 -> r27 77.4024
         -> r28 77.5029 -> **r29 77.6644**.  Top-1 82.17, gap 4.51.

------------------------------------------------------------------------------------------------
BYTE BUDGET: THE 350 MiB CAP IS NOT A LEVER. AXIS CLOSED IN BOTH DIRECTIONS.        [28/07]
------------------------------------------------------------------------------------------------
r29 = 349,771,217 B = 333.57 MiB, headroom 16.43 MiB. Bytes in this submission can buy exactly one
thing -- encode fidelity -- and encode fidelity is already at its optimum.
COST SIDE (measured on the real r29 pixels, extrapolated from the true shipped byte totals):
  4:4:4 everywhere = 489.47 MiB, infeasible. Per scene ss2->ss0: towers +25.5..26.1 MiB (each
  ALONE exceeds the headroom), chair +11.21, bonsai +8.94, HCM0421 Q99->Q100 +6.64.
SCORE SIDE (HCM0181, REAL test GT, full r29 chain, n=60, vs shipped q100/4:2:0):
  4:2:2          -0.0035    4:4:4          -0.0076
  q98            +0.0020 +-0.0020 (noise, t=1.0, 32/60)   q96 -0.1106   q93 -0.6047
  chroma blur sigma 0.5 / 1.0 / 2.0:  -0.0200 / -0.0646 / -0.2787  (monotone -> DEAD)
=> We sit on a narrow q98..q100 plateau; below q98 it collapses. 4:2:0 beats 4:4:4 for the same
reason JPEG beats lossless PNG and lanczos4 beats sharper kernels: 96% of our missing HF is
INCOHERENT with GT, so buying fidelity buys back our own error. Explicit chroma low-pass is NOT the
mechanism (monotone negative) -- it is specifically the block-structured artifact.
**CORRECTION: the long-carried "q100/ss0 = +0.0094" is WRONG.** It came from a single-member PNG
chain with no restore and no field. On the shipped chain it is -0.0076. Independently reproduced by
the campaign's jpegalloc lens at -0.0049. HCM0421's stray Q99 is inside the flat region -> harmless,
do not spend 6.64 MiB "fixing" it.
BANKABLE: q98 is score-neutral at 25% fewer bytes, i.e. ~80 MiB is available for free if a future
operator ever inflates file size. Insurance, not score.

------------------------------------------------------------------------------------------------
THE ENSEMBLE HAS AN INTERIOR OPTIMUM AT k~10. WE WERE AT 8.                          [28/07]
------------------------------------------------------------------------------------------------
Production harness HCM0181, real test GT, n=60, FULL shipped chain, members added best-first by
solo PSNR (kcurve2.py -- member Laplacians hoisted; the first version recomputed 54 pyramids/view):
    k= 4  77.9190  |  k= 6  78.5059  |  k= 8  78.8165 (r29)  |  k=10  78.8653  <- ARGMAX
    k=12  78.8075  |  k=14  78.7380
  8->10 = +0.0488 +/- 0.0104 (4.7 sigma).  8->12 = -0.0091.  8->14 = -0.0785.
MECHANISM: PSNR rises MONOTONICALLY to k=14 (26.1849 -> 26.2448) while LPIPS TURNS OVER after k=10
(0.09338 -> 0.09659). Deeper pixel-averaging keeps cutting squared error while destroying
perceptual texture, and LPIPS carries -40. Same "our HF is incoherent with GT" physics as JPEG >
lossless PNG and lanczos4 > sharper kernels.
=> THIS RIGHT-SIZES THE CAMPAIGN'S HEADLINE "composition is the lever, +0.2030, worth 44x
weighting". That is true CLIMBING TOWARD the optimum (4->8 is +0.90). We already sit at 8. Six free
members per tower exist on disk; taking all six would LOSE score. Take exactly two.

MEMBER SELECTION: THE COLLAPSE GATE IS NOT A QUALITY GATE.
All 30 unused private tower members PASSED member_gate.py -- it only catches fog/collapse. Ranking
them GT-free against the shipped 8-member ensemble (cand_rank.py: consensus deviation + Laplacian
L0 HF energy):
    SHIPPED r2r9_ut7/ut42/seed101  dev 3.23-3.28  HF 1.037-1.039 | mip555 dev 2.49 HF 1.014
    r17_ema999   dev 2.79  HF 1.038  <- BETTER than every shipped UT member.  PICKED
    r2r8_ut42    dev 3.89  HF 1.013  <- credible second.                      PICKED
    r2r8_ut7     dev 3.94  HF 1.015
    s2g_champA/memB/memC  dev 7.35-7.41  HF 0.985-1.005  <- 2.3x the deviation of ANYTHING we ship
      and softer than the peer detail band in 14/15 tower-candidate pairs. An agent recommended
      these FIRST on the strength of the name "champA". Shipping them would have repeated the
      gsplatB6bilagrid failure (-0.7418). EXCLUDED.

------------------------------------------------------------------------------------------------
LOVO CALIBRATED AGAINST REAL TEST GT -- AND MY 10-30x HYPOTHESIS REFUTED                [28/07]
------------------------------------------------------------------------------------------------
HYPOTHESIS (mine): chair's field was killed by leave-one-view-out on TRAIN views, the same protocol
that got field AMPLITUDE wrong, so it might under-read by 10-30x and chair/bonsai could be carrying
a large uncorrected lens error. Tested head-to-head on HCM0181: same member (gsplatB9ut), same
median ds8 field, same metric, raw PNG both sides, ONLY the pose set differing.
    PROTOCOL A  LOVO, held-out TRAIN views, n=50:  none 78.8906 -> med_ds8_g1_rlan 79.8550 = +0.9644
    PROTOCOL B  real TEST poses + real TEST GT, n=60: none 75.2555 -> same field g1.0 = +1.4116
                     gain 1.15 +1.5003 | gain 1.30 +1.5586 <- argmax | gain 1.45 +1.5559
=> RATIO 1.46x, NOT 10-30x. REFUTED. LOVO reads field PRESENCE nearly right (+0.9644 against the
LB's +1.028/scene at R7); it gets AMPLITUDE wrong. Those are separable failures, so the r29 finding
stands and this extrapolation from it does not.
CONSEQUENCE (still useful): chair's LOVO +0.0299 projects to ~+0.0483 production = +0.0069 blended.
Chair genuinely has ~32x less misregistration than a tower: mean |d| 0.101 px vs 0.211 px, because
it is a handheld camera at 720x1280, not a drone with k1~+0.009. Field fitted, gain 1.30, --strict
clean. ALSO: gain 1.45 == 1.30 inside noise on a SECOND chain -> amplitude axis closed for good.

------------------------------------------------------------------------------------------------
GAUSSIAN-SMOOTH THE FIELD (sigma=1 on the ds8 grid): FREE +0.0106/tower                 [28/07]
------------------------------------------------------------------------------------------------
Full r29 chain, HCM0181, real test GT, n=60:
    plain x1.30 78.5186 | gauss1 x1.30 78.5293  +0.0106 +/- 0.0007, 59/60 WINS
                        | gauss2 x1.30 78.5307  +0.0120 +/- 0.0015, 52/60
sigma=2's extra +0.0014 +/- 0.0017 is noise and its win rate is worse -> sigma=1. Independently,
the chair LOVO preferred med_ds8_g1_rlan (+0.0300) over med_ds8_rlan (+0.0207) -- second scene,
same conclusion. Costs nothing, no bytes.

=> r30 = towers 8->10 members at TRUE uniform (+0.0349) + field gauss1 (+0.0076) + chair field
   (+0.0069) = ~+0.049 blended -> ~77.71. Video-scene member counts DELIBERATELY UNCHANGED: the
   k=10 optimum was measured WITH restore lam=1.0 putting texture back; chair/bonsai ship lam=0, so
   their over-smoothing penalty per member is LARGER and their optimum is probably LOWER. r28
   already cost a round by assuming tower behaviour transfers to the video scenes.

r30 BUILT (NOT SUBMITTED): /mnt/d/avv/submissions/sub_round30_k10.zip
  349,585,510 bytes = 333.39 MiB (cap 367,001,600 -> FITS, 16.6 MiB spare)
  386/386 exact, CRC OK, VERIFY PASSED. Changed: 5 towers + chair. bonsai BYTE-VERBATIM from r29.
  Members added per tower: r17/<T>_ut7_ema999 + r2r8/models/<T>_ut42, weights 6 1 1 1 1, k=10.
  Fields: /mnt/d/avv/fields_median_g1_g130/ (gauss1 x1.30) and fields_chair/chair_g1_g130.npy.
  Build script + full rationale: /home/bkai/.claude/jobs/1c9cf7e9/tmp/build_r30.sh
  Provenance: /mnt/d/avv/submissions/sub_round30_k10.PROVENANCE.txt
  BUILD NOTE: first attempt aborted on MY guard bug -- apply_field writes 60 PNGs PLUS
  field_applied.json, so `ls | wc -l` is 61 and `-eq 60` rejected a correct result. Fixed to count
  *.png, added a resume check, and VERIFIED THE PATCH CONTENT LANDED (not just that it parsed).

------------------------------------------------------------------------------------------------
ENCODE AXIS RE-OPENED: q98 / 4:4:4 IS +0.0504 AND CHEAPER. MY GRID MISSED THE CELL.     [28/07]
------------------------------------------------------------------------------------------------
I swept QUALITY at 4:2:0 and SUBSAMPLING at q100 and never crossed them, then concluded the axis was
closed. The winning cell was the one I skipped. A campaign agent found it; I re-measured
independently at k=10 + gauss1, n=60, real test GT and reproduced +0.0504 vs their +0.0507.
LADDER AT 4:4:4 -- a SHARP interior optimum:
   q100 +0.0147 (45/60) | q99 +0.0384 (55/60) | q98 +0.0504 (59/60, t=16.96) | q97 +0.0209 (50/60)
   q96 -0.0446 (1/60) | q94 -0.2953 (0/60)
CHEAPER TOO: 53.04 vs 59.30 MiB/60. Submission drops 333.39 -> 302.70 MiB. Rate-distortion win, not
a byte purchase -- so "the cap is not a lever" was RIGHT for the wrong reason: the win is free.
LPIPS-led (0.09341 -> 0.09132). 4:2:0 blurs chroma spatially and leaves block structure; q98 trims
incoherent HF in frequency. Different operations; the frequency one is what helps.
CORRECTION: "q100/ss2 beats lossless PNG" is FALSE on the production harness (PNG +0.0544). That
came from the retired eval-split proxy. PNG stays infeasible: 137.81 MiB/60 -> ~690 MiB of towers.

FIELD GAIN 1.50 REFUTED (campaign synthesis proposed it from a 30k-vs-60k field-class argument,
ratio 1.1486, 1.30x1.1486=1.493). At matched encode 1.50 is -0.036 vs 1.30; g1.50_q100ss2 = -0.0343
(t=-2.81, 19/60); 1.40 -0.007; 1.60 -0.085. THIRD independent measurement putting the optimum at 1.30.

=> SHIP sub_round30c_k10_q98ss0.zip (302.70 MiB, VERIFY PASSED). The other two r30 builds each have
   exactly one of the two settings wrong. Expected +0.06..+0.10 over r29 -> 77.72..77.77.

================================================================================================
r30c GRADED: **77.6595** (PSNR 26.628392  SSIM 87.2653  LPIPS 11.2429)   [28/07]  ** A LOSS **
================================================================================================
-0.0049 vs r29 (77.6644). PREDICTED +0.06..+0.10. MISSED BY ~0.08. r29 remains our best.
Decomposition closes exactly:
    0.6*(-0.009025) + 30*(-0.000431) - 40*(-0.000336)
  = -0.005415      -0.012930       +0.013440  = -0.004905  vs -0.0049 graded.

WHAT THE SPLIT SAYS: LPIPS DID improve (-0.0336pp) but only ~16% of what the encode predicted
(-0.209pp), while the encode's PSNR/SSIM COSTS appear to have transferred in full. An operator
whose whole value is a big LPIPS gain bought with PSNR/SSIM losses flips NEGATIVE the moment the
gain under-transfers. That is exactly what happened.

**THE METHODOLOGICAL ERROR, NAMED: EVERY NUMBER IN THIS ROUND CAME FROM ONE SCENE.**
production-harness.md already records the limitation -- HCM0181 is the ONLY public tower with a
real multi-member ensemble at test poses. So the k-curve (k=4..14) AND the encode ladder (q94..q100
x ss0/ss2) were both n=60 VIEWS of n=1 SCENE. The t=16.96 and 59/60 win rates measure VIEW-level
variance and say NOTHING about cross-scene transfer, and I read them as if they bounded it.
CONTRAST r29, which transferred at ~1x: its field gain was extended by the verifier to ALL FIVE
public towers (alpha* 1.34-1.43, |M|/|f| 1.26-1.39 on 5/5) BEFORE shipping. That is the difference.
=> NEW RULE: a single-scene harness result is a HYPOTHESIS, not a measurement. Cross-validate on
   >=3 public towers before shipping, or discount it hard. View-level SE is not transfer error.

SECOND ERROR: FIVE STACKED CHANGES => THE LOSS IS NOT ATTRIBUTABLE. I flagged this risk in the
provenance and shipped anyway. Cannot tell whether the members, the encode, or both cost us.
MITIGATION AVAILABLE FOR FREE: sub_round30_k10.zip is ALREADY BUILT AND VERIFIED and is exactly
r30c MINUS the encode change. Grading it splits the bundle perfectly:
    (r30  - r29 ) = members + gauss1 field + chair field
    (r30c - r30 ) = the encode, alone

------------------------------------------------------------------------------------------------
ROOT CAUSE OF THE r30c LOSS: THE HARNESS ENSEMBLE IS TWICE AS DIVERSE AS THE ONE WE SHIP  [28/07]
------------------------------------------------------------------------------------------------
Cross-validated the encode on SINGLE-MEMBER renders of all 5 public towers (encode_xscene.py),
same field construction, real test GT:
   q98/ss0 vs q100/ss2:  HCM0181 -0.0100 | HCM0193 -0.0038 | HCM0204 -0.0121 | hcm0031 -0.0132
                         hcm0034 -0.0081   5-scene mean -0.0094   scenes positive 0/5
**NEGATIVE ON HCM0181 ITSELF** -- the very scene where the k=10 ensemble gave +0.0504.
So the effect was never scene-specific; it is ENSEMBLE-DIVERSITY specific. JPEG artifacts
substitute for texture the pixel-mean destroys: a deep DIVERSE ensemble is over-smoothed so
artifacts help; a single render has its own texture so they hurt.
THE KILLER NUMBER (measured earlier by the combiner lens and ignored by me): our production pool's
heterogeneity is 2.20/255 vs the harness pool's 4.67/255 -- HALF. HCM0181's k=10 pool is 10 wildly
different recipes; ours is UT-family + 2 mip3d + an EMA + an old-generation member. Less
disagreement -> less over-smoothing -> less benefit from artifact substitution. Gain transferred at
~16% while the encode's PSNR/SSIM COSTS transferred in full => net negative.

**GENERAL RULE:** any operator whose value depends on HOW OVER-SMOOTHED THE ENSEMBLE MEAN IS will
OVER-READ on the production harness. That includes the encode, and it contaminates the k-curve --
the k~10 argmax was found on the diverse pool, so our homogeneous pool's optimum is probably LOWER,
which means even the member addition may not be worth its predicted +0.035. It does NOT affect
per-image operators independent of the pool: the lens field transferred at ~1x.
=> Before shipping a harness result: cross-validate on >=3 public towers AND ask whether the effect
   depends on ensemble diversity. A 59/60 win rate on ONE scene is not transfer evidence.

================================================================================================
r30 GRADED: **77.6614** (PSNR 26.644762  SSIM 87.325  LPIPS 11.3074)  -> PERFECT SPLIT  [28/07]
================================================================================================
Submitting r30 (r30c MINUS the encode) split the bundle EXACTLY. Both halves verified against
their own submetrics:
   r30  - r29  = members + gauss1 field + chair field
                 0.6(+0.007345) + 30(+0.000166) - 40(+0.000309) = **-0.0030**
   r30c - r30  = the encode q98/4:4:4, ALONE
                 0.6(-0.016370) + 30(-0.000597) - 40(-0.000645) = **-0.0019**
                                                          total  = -0.0049 = r30c - r29  OK
PREDICTED +0.050 and +0.050. BOTH NEGATIVE. r29 (77.6644) remains our best.

**THE MEMBER SIGNATURE PROVES THE DIVERSITY ACCOUNT.** 8->10 gave PSNR +0.0073 and LPIPS WORSE by
+0.0309pp. That is precisely the harness's PAST-THE-OPTIMUM signature: at k=12 the harness had PSNR
still climbing (26.2245 -> 26.2338) while LPIPS turned over (0.09348 -> 0.09501). We did not move
toward k=10; we moved PAST OUR OWN optimum.
WHY: homogeneous members carry less INDEPENDENT information, so our k=8 pool at 2.20/255 diversity
is already as over-smoothed as a DIVERSE pool at k~12. **r29's k=8 was already right.** The k-curve's
k~10 argmax is a property of HCM0181's 10-wildly-different-recipes pool, not of ours.
=> EFFECTIVE k != NOMINAL k. Scale by pool heterogeneity before transferring any k-dependent result.

ENCODE'S TRUE PRODUCTION VALUE: -0.0019, sitting between the single-member cross-scene mean
(-0.0094) and the diverse-k=10 harness (+0.0504), much nearer the former -- exactly as the
heterogeneity account predicts. Three measurements, one coherent story.

WHAT SURVIVES: the two FIELD changes (gauss1 smoothing, chair field). They are PER-IMAGE operators
independent of ensemble composition -- the category that demonstrably transfers at ~1x (the lens
field delivered +0.7345 on the LB). They were probably positive and buried under the members' drag.
BUT gauss1 was measured on HCM0181 ALONE, which is the exact error that cost the last two rounds,
so it is being cross-validated on 5 towers BEFORE shipping. Holding the rule I just wrote.

r31 PLAN (cheap -- pure re-warp + re-encode, all masters on disk):
  towers  /mnt/d/avv/r29/tower_ens/<T>/png_er  -> gauss1 x1.30 field -> q100/ss2
  chair   /mnt/d/avv/r30/video_ens/chair/png   -> q100/ss2
  bonsai  verbatim from r29
i.e. r29's exact members and encode, keeping ONLY the two field operators.

------------------------------------------------------------------------------------------------
FIELD SMOOTHING CROSS-VALIDATES 5/5 -- THE CONTRAST THAT DEFINES THE TRANSFER RULE     [28/07]
------------------------------------------------------------------------------------------------
gauss(sigma=1) vs plain, gain 1.30, single member, shipped encode, real test GT, all 5 public towers:
   HCM0181 +0.0095 (60/60) | HCM0193 +0.0082 (56/60) | HCM0204 +0.0073 (57/60)
   hcm0031 +0.0080 (47/50) | hcm0034 +0.0077 (55/60)
   5-SCENE MEAN +0.0081, POSITIVE 5/5, tight 0.0073-0.0095.   sigma=2: +0.0091, 5/5, wider spread.
CONTRAST the encode on the identical protocol: -0.0094 mean, 0/5 positive.
=> THE DISCRIMINATOR IS NOT "harness vs LB", IT IS **POOL-DEPENDENT vs PER-IMAGE**:
   PER-IMAGE (transfer ~1x): lens field (+0.7345 LB), field smoothing (5/5), resampling kernel.
   POOL-DEPENDENT (do NOT transfer): encode/artifact substitution, ensemble depth k, anything whose
   value scales with how over-smoothed the mean is. HCM0181's pool is 2x as diverse as ours.

r31 BUILT: sub_round31_fields.zip, 349,917,932 B = 333.71 MiB, 386/386 exact, VERIFY PASSED.
  = r29 members (k=8) + r29 encode (q100/ss2) + gauss1 field + chair field. bonsai verbatim.
  Expected +0.010 -> ~77.674. Nothing in it depends on ensemble diversity.

================================================================================================
r31 GRADED: **77.6804** (PSNR 26.645885  SSIM 87.3345  LPIPS 11.2687)  ** NEW BEST **   [28/07]
================================================================================================
+0.0160 over r29 (77.6644). Predicted +0.0127; delivered 126% of it. Decomposition closes exactly:
    0.6*(+0.008468) + 30*(+0.000261) - 40*(-0.000078)
  = +0.005081      +0.007830       +0.003120  = +0.016031  vs +0.0160 graded.
ALL THREE SUBMETRICS MOVED THE RIGHT WAY TOGETHER -- the pure-fidelity signature, same profile as
the r29 field-amplitude correction. That is what a per-image operator looks like when it lands.

**THE TRANSFER RULE IS NOW QUANTITATIVELY CONFIRMED:**
    PER-IMAGE      (r31: gauss1 field smoothing + chair field)   predicted +0.0127 -> +0.0160  126%
    POOL-DEPENDENT (r30/r30c: 2 extra members + q98/4:4:4 encode) predicted +0.100  -> -0.0049   -5%
Same harness, same chain, same session. The discriminator is not "harness vs LB", it is whether
the operator's value depends on HOW OVER-SMOOTHED THE ENSEMBLE MEAN IS. HCM0181's k=10 pool is
2x as diverse as ours (4.67 vs 2.20 /255), so pool-dependent effects over-read there and per-image
ones do not.

PROGRESSION r22 77.2318 -> r27 77.4024 -> r28 77.5029 -> r29 77.6644 -> (r30 77.6614, r30c 77.6595
diagnostic detours) -> **r31 77.6804**.  Top-1 82.17, gap 4.49.

NEXT: bonsai is the LAST scene with NO lens field. Fields are now the best-established member of
the transferable class (tower field +0.7345 LB; chair field just landed; gauss1 5/5 towers).
Flow cache built from r2r8/models/bonsai_ut42/train_png (124 pairs) vs 248 bonsai TRAIN photos --
Rule 10 clean. LOVO running with the same variant set the chair re-gate used, so the two scenes
are directly comparable.

------------------------------------------------------------------------------------------------
SIGMA SWEEP CLOSES THE LAST PER-IMAGE AXIS. BONSAI FIELD DEAD.                          [28/07]
------------------------------------------------------------------------------------------------
Field-smoothing sigma, 5 public towers, single member, gain 1.30, shipped encode, real test GT:
   g1 +0.0081 (5/5) | **g2 +0.0091 (5/5) <- argmax** | g3 +0.0029 (4/5) | g4 -0.0109 (0/5)
   g6 -0.05..-0.07  | g8 -0.13..-0.16
r31 shipped sigma=1. Moving to sigma=2 is worth +0.0010/tower = **+0.0007 blended** -- far below LB
noise, NOT worth a submission. The curve turns over hard past sigma=3, so nothing lies further out.

BONSAI LENS FIELD: DEAD. LOVO n=40, every variant <= no-field:
   none 48.6139 (best) | mean_ds8_cubic +0.0012 (noise) | med_ds8_g1_rlan -0.0176
   med_ds16_rlan -0.0394 | med_ds8_rlan -0.1248 | median_ds8 -0.1270
Matches the earlier independent finding (its geometry-based field earned +0.0006). CAVEAT: bonsai's
LOVO base is 48.6 vs chair's ~78, and the only available fit source is r2r8/models/bonsai_ut42 --
an OLDER generation on the one scene with a fog-collapse history. So a field fit there is partly
measuring the MODEL's errors, not the lens. Read as "do not ship", not "no lens error exists".

=> PER-IMAGE AXIS NOW EXHAUSTED: field amplitude (1.30), smoothing (sigma 1-2), resampling kernel
   (lanczos4), chair field (shipped), bonsai field (dead), encode (q100/ss2 correct FOR OUR POOL).
=> AND THE "TRAIN MORE MEMBERS" PATH IS CLOSED TOO, which is new: our homogeneous pool at k=8 is
   already PAST its optimum (r30 proved it on the LB). Adding same-family members HURTS. The only
   structural lever left is training a genuinely DIFFERENT-FAMILY member, which would raise pool
   heterogeneity (2.20 -> closer to the harness's 4.67) and would ALSO make the encode change turn
   positive -- but that needs retraining time we do not have.

FINAL POSITION: **r31 = 77.6804 is our best.** Top-1 82.17, gap 4.49, and that gap is primitive
quality. Remaining post-process levers are worth ~0.001 each.

------------------------------------------------------------------------------------------------
HETEROGENEITY GATE FAILS -> r34/r35 (train non-UT members) CANCELLED. + A SELF-CORRECTION [28/07]
------------------------------------------------------------------------------------------------
Question: at FIXED k=8 with MATCHED solo quality, does pool DIVERSITY pay? If yes, training
different-family members (~12h GPU) is justified; if no, that path is dead.
Built min- and max-diversity 8-subsets from HCM0181's top-14 by solo PSNR:
   DIV8  het 4.050/255  meanSolo 24.008 dB     HOMO8 het 3.486/255  meanSolo 24.050 dB
(DIV8 carries a 0.042 dB quality HANDICAP, so the test is conservative.)
   HOMO8  78.6678  PSNR 26.1268  SSIM 0.89276  LPIPS 0.09477
   DIV8   78.6883  PSNR 26.2405  SSIM 0.89482  LPIPS 0.09752   **+0.0204 +- 0.0330, 26/60 (t=0.62)**
Diversity buys PSNR (+0.114) and SSIM and gives it ALL BACK in LPIPS. NET NOISE, win rate BELOW
chance. => NOT worth 12h GPU. r34/r35 CANCELLED.

**SELF-CORRECTION.** Earlier today I wrote that our k=8 pool is "past its optimum", citing r30's
PSNR-up/LPIPS-down signature. That signature has TWO possible causes: (a) too much averaging, or
(b) the two added members were simply WORSE (r2r8_ut42 is an OLDER training generation). I asserted
(a). This gate shows diversity/effective-k is NOT the driver, which makes (b) the better
explanation. r30 most likely lost on MEMBER QUALITY, not ensemble depth. The two cannot be cleanly
separated from the data we have, and I should not have written it as settled. Memory corrected.
CONSEQUENCE: when adding a member, select on QUALITY and current training generation -- not on
novelty or decorrelation. Consistent with the k=4 result (corr(decorrelation, gain) = -0.985).

------------------------------------------------------------------------------------------------
GEOM (+0.74, the largest un-harvested residual component) CHECKED AND CLOSED           [28/07]
------------------------------------------------------------------------------------------------
Round-7 decomposition: LENS +0.40 / POSE +0.14 / GEOM +0.74. LENS was harvested (field +0.7345 LB,
plus the r29 amplitude fix +0.1615). POSE was killed (oracle ceiling only +0.14). GEOM was never
touched and is the one remaining component with a ceiling above +0.1. Checked today:
  1. **COLMAP IS NOT INSTALLED** (not on PATH, no /usr/local/bin/colmap). Dense/MVS seeding would
     require building it from source with CUDA -- hours, high failure risk, before any retraining.
  2. **The thesis is weak in OUR regime anyway.** Sparse init is already dense: HCM0421 171,304 /
     HCM0539 218,846 / chair 80,491 / bonsai 54,422 points. And we train with gsplat MCMC
     densifying to cap_max 8,000,000, so the init is ~2% of the final model and MCMC relocates
     aggressively. Dense seeding pays in VANILLA 3DGS where densification is timid; not here.
  3. Every other GEOM route is already dead: depth priors (hurt solo 3/3), 2DGS (backward-pass
     errors), 120k schedule (train +0.52 dB, test +0.00).
=> GEOM CLOSED. With it, every axis in the residual decomposition is now either harvested or dead.

================================================================================================
FINAL POSITION 2026-07-28: **r31 = 77.6804** (sub_round31_fields.zip). Session +0.1775 over r28.
================================================================================================
Recommend STOPPING. Full inventory of remaining levers and their measured value:
   field smoothing sigma 1->2            +0.0007   (below LB noise)
   train non-UT members (12h GPU)        noise     (gate +0.0204 +- 0.0330, 26/60, t=0.62)
   add any member                        <= 0      (r30 -0.0030)
   encode / bytes / chroma               <= 0      (0/5 towers)
   field amplitude / kernel / ds / est   0         (closed, 3x measured)
   bonsai field                          <= 0      (every variant loses)
   ensemble depth / weighting / family   <= 0      (closed)
   GEOM / dense seeding                  N/A       (no COLMAP; MCMC makes init ~irrelevant)
   video-scene lambda interior 0.25-0.5  UNKNOWN   (2/7 of score, NO test GT -> unmeasurable gamble)
Only the last row is open, and it is a coin flip with no way to measure it. r28 already cost a
round by assuming tower behaviour transfers to the video scenes.

## *** R32 BUILT + VERIFIED -- video-scene lambda, turned from a gamble into a measurement *** [28/07 13:40]

The last open row above ("video-scene lambda interior 0.25-0.5 UNKNOWN -- unmeasurable gamble")
is no longer unmeasurable. It does not need test GT.

THE MEASUREMENT (lambda_diag.py, GT-free).  The restoration gain is driven entirely by the
r-map, r = sqrt(1 + (k/(k-1)) * E(L0_i - L0_mean) / E(L0_mean)), which is computable from the
member renders alone. The quantity that matters is the mean finest-band energy BOOST that
lambda=1.0 would apply, i.e. mean(r-1) weighted by |L0(mean)|:

    scene            boost @ lam=1.0
    HCM towers            21.3% .. 33.6%      (lam=1.0 is the LB-PROVEN optimum here)
    chair                 64.5%               (3.0x the tower median)
    bonsai               119.2%               (4.4x the tower median)

So shipping lam=1.0 to the video scenes does NOT apply "the same operator" -- it applies an
operator 3-4x stronger than the one the leaderboard actually validated. That is why r28's
video lam=1.0 cost a round: not because video scenes are special, but because the SAME lambda
is a DIFFERENT amount of sharpening when the member pool disagrees more.

THE FIX is to match the delivered boost, not the parameter:
    chair   lam = 27 / 64.5  = 0.42  -> shipped 0.40
    bonsai  lam = 27 / 119.2 = 0.23  -> shipped 0.25
Both land inside the 0.25-0.5 interior the old note called a coin flip, and they get there from
a number rather than from a guess. This is the same TRANSFER-RULE logic that predicted r31 to
126% accuracy: normalize the operator to its per-image effect, not its nominal setting.

BUILD (build_r32.sh, assembled on top of r31 so the towers cannot move):
    chair   8-member mean -> restore(lam=0.40, k=8) -> chair_g1_g130 field -> q100/ss2
            field applied: mean|d| 0.0790 px, max|d| 0.6127 px
    bonsai  7-member mean -> restore(lam=0.25, k=7), NO field (field is dead on bonsai) -> q100/ss2
    towers  ALL 300 FILES CARRIED BYTE-VERBATIM from r31

VERIFY: CRC OK; 386/386 exact-name/exact-size; 5x60 towers @1320x989, bonsai 28 @1920x1080,
chair 58 @720x1280, all JPEG. scenes changed vs r31 = ['bonsai','chair'] and nothing else.
TOTAL 350,756,907 B = 334.51 MiB against the 367,001,600 B (350 MiB) cap -> FITS with 32 MiB spare.

    /mnt/d/avv/submissions/sub_round32_videolam.zip     -- BUILT, NOT SUBMITTED

RISK SHAPE: chair+bonsai are 2/7 of the score. The delta is bounded by how much the finest band
of two scenes can move. Upside if the boost-matching argument holds ~ +0.02; downside if video
scenes want lam=0 after all ~ -0.01 (r29/r31 shipped lam=0 there and won). Asymmetric, cheap,
and cleanly attributable because only the video scenes changed.

## *** FREEZE DAY: BONSAI IS THE BROKEN SCENE, AND THE RESTORE OPERATOR HAS A DEADBAND *** [28/07 15:xx]

### 1. PER-SCENE TRIAGE -- the deficit is CONCENTRATED, not diffuse (this was never measured before)
Scored all 7 private scenes on TRAIN views against TRAIN photos (legal GT; train->test is
slope 0.80 with a near-constant 2.03 dB gap, established at :1200), n=16 views/scene:

| scene | PSNR | SSIM | LPIPS | train score |
|---|---|---|---|---|
| HCM0421 | 27.802 | .9105 | .0768 | 80.924 |
| HCM0539 | 27.961 | .9161 | .0750 | 81.260 |
| HCM0540 | 28.432 | .9141 | .0752 | 81.472 |
| HCM0644 | 27.368 | .9164 | .0766 | 80.846 |
| HCM0674 | 28.387 | .9253 | .0749 | 81.797 |
| chair | 29.607 | .9122 | .0911 | 81.487 |
| **bonsai** | 27.941 | **.8707** | **.2047** | **74.697** |

bonsai is -5.66 below the 7-scene mean and its LPIPS is 2.7x every other scene. It reconciles
with the leaderboard: 6 scenes at ~9.3% + bonsai at ~20% averages to 10.8% vs our actual 11.22%.
BONSAI ALONE CARRIES THE 0.4-WEIGHTED LPIPS TERM. Radial power spectrum render/GT confirms it is
a reconstruction failure, not a sharpening deficit: bonsai mid 0.42 / high 0.19 / vhigh 0.23,
against HCM0421 1.02 / 0.90 / 0.82 and chair 0.92 / 0.61 / 0.38.

CAUSE: bonsai alone trains at 30k iters / 5M cap; chair and all five towers get 60k / 8M. From
r28_members.sh:17 -- "bonsai is only 30k/5M so it costs ~1.2h" -- i.e. a SCHEDULING shortcut, not
a quality decision. The 17/07 ladder that chose it was still climbing at the top (0.5M 69.571 ->
1M 70.298 -> 2M 70.660 -> 5M 71.156) with the cap BINDING ("Saved 5000000 gaussians"), and every
arm was fixed at 30k iters. 8M was never tested. The fog collapse it was guarding against was
resolved as MCMC CHURN (noise to 50k), which is orthogonal to cap and iters.
=> retrained at 8M/60k with capD's churn (refine_stop 15k, noise_stop 8k) UNCHANGED. r33_bonsai.

### 2. BONSAI LAMBDA IS ZERO -- swept on REAL held-out GT, not argued
28 held-out frames, k=6 members, full shipped encode. This is the measurement r32 could not make.
| lam | PSNR | SSIM | LPIPS | SCORE | d |
|---|---|---|---|---|---|
| **0.00** | 27.431 | .8560 | **.2546** | **71.955** | +0.000 |
| 0.25 (r32) | 27.429 | .8556 | .2649 | 71.531 | -0.424 |
| 0.50 | 27.426 | .8554 | .2682 | 71.390 | -0.565 |
| 1.00 | 27.415 | .8544 | .2830 | 70.760 | -1.195 |
| 2.00 | 27.378 | .8513 | .3079 | 69.650 | -2.304 |
MONOTONICALLY HARMFUL. bonsai's members disagree because the reconstruction FAILED, not because
averaging destroyed detail, so the r-map amplifies noise. r32's bonsai lam=0.25 was a -0.424
scene-pt (-0.061 LB) mistake. It cost ~nothing only because of finding 3.
Ensemble context (same split, no encode): 6-member mean 72.013 vs best single member 71.903 --
ensembling bonsai is roughly neutral, so a genuinely better single model can ship alone.

### 3. THE DEADBAND BUG (freeze-day fault audit, then verified independently)
energy_restore --apply reads ens_dir, which stores round(mean)*255 as uint8 = an EXACT INTEGER
per pixel. The write is (o*255+0.5).astype(uint8) = round(n+d) = n + round(d), so EVERY pixel
whose intended correction is |d| < 0.5 LSB gets ZERO change. No dither.
DESTROYED: 49% of the intended correction on chair, 79% on bonsai. Towers ship at 0.13-0.58 LSB
delivered (HCM0539 0.1297, HCM0421 0.2255, HCM0644 0.5849).
Public harness, same lambda, float chain vs shipped chain: lam=1.0 -0.0021 (1% lost),
lam=0.50 -0.0295 (18%), lam=0.25 -0.0419 (43%). The damage is worst at LOW amplitude -- exactly
where our private pool (half the harness disagreement) operates. The harness validated lam=1.0 at
0.627 LSB where the deadband is harmless; the private towers never ran in that regime.
FIX: rebuild the same mean in float32 from the dirs it was formed from, round ONCE at the end.
No re-rendering. png_ens is reproducible as round(float mean): bonsai k=7 BIT-EXACT (1.000000 --
sum(n_i)/7 can never hit a .5 tie), chair k=8 0.9715 and towers 4:1:1 0.9422, every mismatch
being exactly the +-1 LSB tie case where the float mean is still exactly recoverable.
Implemented as energy_restore --mean_from_dirs/--mean_weights, gated by asserting the rebuild
reproduces png_ens except at ties.

*** THE TRAP, AND IT IS THE REAL LESSON ***
The fix makes the operator STRONGER. Finding 2 shows the operator is HARMFUL on bonsai. Applying
the deadband fix everywhere would have amplified a -0.424 operator on the worst scene -- the
audit's "+0.009 on the video scenes" priced delivery, not sign. THE DEADBAND WAS ACCIDENTALLY
PROTECTING BONSAI. Ship the fix ONLY where lambda is independently known positive: the towers,
where r28 (lam=1.0) beat r27 (lam=0) by +0.1005 on the leaderboard. Expected +0.016.

### 4. ALSO CLOSED THIS ROUND
- Tower lambda: the harness optimum lam=1.0 was suspected to be a pool-dependent artifact (harness
  disagreement 2x ours). Measured delivered |L0|-weighted boost at lam=1.0: harness HCM0181 k=10
  7.12% vs private towers 5.85% = 1.22x, NOT the 4.5x the disagreement-squared story predicted.
  Boost-matching would want lam~1.22 and the harness prices lam=1.25 at -0.006. TOWER LAMBDA IS A
  WASH -- not a lever. (Ratio measured with one code path across pools; the older "4.67 vs 2.20
  /255" figures use a different disagreement statistic and are not comparable to these.)
- Rediscovered that images.bin ships test-frame poses + ~2.3k SIFT keypoints/img. ALREADY FOUND
  AND CLOSED on 16/07 at :1275 (+0.048 pts over the ship path, rejected). Left closed.
- r32 provenance sidecar written (it was missing); EXPERIMENTS.md:4234 "32 MiB spare" corrected to
  15.49 MiB (367,001,600 - 350,756,907 = 16,244,693 B).
- Laplacian nlev=5 is a numerical no-op vs nlev=1 (max diff 3e-05 LSB) at 5x the pyramid cost.

### 5. R32 GRADED 77.6907 -- NEW BEST (+0.0103 over r31), AND THE ATTRIBUTION IS NOT WHAT WE THOUGHT
PSNR 26.640357 SSIM 87.3191 LPIPS 11.2230. Decomposition closes exactly:
  LPIPS -0.000457 -> +0.01828   SSIM -0.000154 -> -0.00462   PSNR -0.005528 -> -0.00332
  total +0.01034 (observed +0.0103). The round traded PSNR+SSIM for LPIPS and won, correct at
  these weights. BUT the audit measured the shipped bytes: bonsai moved 0.0479 levels mean-abs,
  PSNR(r32,r31) 59.94 dB, finest-band +0.9% against a 27% target. Bonsai contributed ~0.
  THE ENTIRE +0.0103 CAME FROM CHAIR. Boost-matching was never really tested -- it shipped at a
  small fraction of nominal strength (finding 3 explains why).

## *** R33 BUILT + VERIFIED -- deadband fix (measured, not inferred) + bonsai lam=0 *** [28/07 ~17:0x]

sub_round33_deadband.zip: 386/386 exact, 356,304,408 B = 339.80 MiB, headroom 10.20 MiB,
VERIFY PASSED, CRC OK. Base r32 (77.6907). Scenes changed: 5 towers + bonsai. Chair VERBATIM.
NOT SUBMITTED.

### THE GATE THAT DECIDED IT -- public harness, REAL test GT, full shipped chain
Same lambda in both arms; the ONLY variable is whether the ensemble mean is pre-rounded to uint8
before restore. n=10 real HCM0181 test poses, k=8, mean -> restore -> lanczos field -> q100/ss2.
| lam | arm | PSNR | SSIM | LPIPS | SCORE | FLOAT-ROUNDED |
|---|---|---|---|---|---|---|
| 0.25 | ROUNDED | 25.531 | .8857 | .1032 | 77.7630 | |
| 0.25 | FLOAT | 25.532 | .8859 | .1030 | 77.7746 | **+0.0116** |
| 0.50 | ROUNDED | 25.519 | .8862 | .1014 | 77.8426 | |
| 0.50 | FLOAT | 25.520 | .8864 | .1012 | 77.8531 | **+0.0104** |
Positive at both amplitudes and ALL THREE submetrics improve -- a real fix, not a metric trade.
Our towers deliver 0.13-0.58 LSB so they live in exactly this low-amplitude regime.
=> 5 towers x ~0.011 / 7 = **+0.008 LB**. NOTE this is LOWER than the audit agent's +0.016
(which interpolated from -0.0295/-0.0419 loss figures); the number above is measured directly and
is the one to believe.

### CHAIN REPRODUCED BYTE-EXACT BEFORE CHANGING IT (the discipline that made this safe)
r31/r32 towers = r29 png_er -> apply_field(fields_median_g1_g130) -> JPEG, **q99 for HCM0421 and
q100 for the rest** (a byte-budget concession). Re-encoding r31's stored png at q99 reproduces the
shipped zip bytes 8/8; at q100 it does not. Only after that did I change one variable.
Float-mean rebuild reproduces png_ens **BIT-EXACTLY** -- worst tie-fraction 0.0000% over 60 frames
-- because ensemble_renders.py built the original mean in float32, so a float32 rebuild reproduces
its tie-breaking. (A float64 rebuild does NOT: it disagrees on 5.8% of pixels, all exact .5 ties.)

### CAVEAT FOUND EN ROUTE: the fix is not purely additive
Delta appears BEFORE the field: png_er new vs shipped is mean 0.16 LSB but max 21 with 0.35% of
pixels >1 LSB. Changing the base from round(mean) to the true float mean also perturbs the r-map
itself, because r = sqrt(1 + V/(Eb+1e-10)) is unstable and clamp-bound where Eb->0. The float base
is the CORRECT input so this is the operator behaving properly -- but it is why the harness A/B
was required rather than shipping on the "restores lost precision" argument.

### CHAIR: LEFT ALONE, AND THE SWEEP THAT SAYS WHY
Chair lambda swept on 58 REAL held-out frames (chair_eval, k=7, shipped encode):
  0.00 71.003 | 0.25 70.953 (-0.051) | 0.50 70.816 | 1.00 70.430 (-0.573) | 2.00 69.383
Monotonically harmful -- which CONTRADICTS the leaderboard, where r32's chair lam=0.40 gained
+0.0103 with the decomposition closing exactly. RESOLUTION: energy restoration is POOL-DEPENDENT
and chair_eval's 7 arms are different models from the 8 production chair members, so the sweep
measures the wrong pool; the LB measured the right one. Both agree on NOT INCREASING it, so:
chair ships byte-verbatim, no deadband fix, no lambda change. This is the transfer rule applied
in the direction that costs us a gain rather than buys one.

### BONSAI: lam 0.25 -> 0, and the trap that nearly shipped
Reverted to r29 png_ens (byte-identical to what r31 shipped, graded 3 rounds). Expected +0.008.
THE TRAP: the deadband fix makes the operator STRONGER, and the bonsai sweep shows it is
MONOTONICALLY HARMFUL there (0.00 71.955 -> 0.25 71.531 -> 1.00 70.760). Applying the "fix"
uniformly would have AMPLIFIED a -0.424 operator on our worst scene. The deadband was accidentally
protecting bonsai. An audit that prices DELIVERY without checking SIGN per scene will confidently
recommend this. Ship precision fixes ONLY where the operator's sign is independently known.

### TWO OF MY OWN ERRORS, CAUGHT BY ADVERSARIAL VERIFICATION BEFORE THEY COST ANYTHING
1. opacity_gate called torch.load without weights_only=False; torch 2.7 defaults it True and these
   ckpts carry numpy objects, so it RAISED -> exit 1 -> read as a fog collapse. Both 14:32 arms
   would have discarded themselves after ~4 GPU-h and logged "collapsed, discarding" -- a false
   negative wearing the exact costume of the real failure mode.
2. The first bonsai retrain changed FOUR variables (iters 30k->60k, lpips_from 12k->30k, cap
   5M->8M, +mip3d). 60k was already A/B'd on these 28 holes on 17/07 and LOST twice (pD 71.0776,
   pA 70.7211 vs shipped pC 71.3605); lpips_from 30k reverts a +0.20 win; the shipped bonsai
   members carry no mip3d and bonsai's mip3d member landed SOFTER than every peer. Only cap was
   untested. Relaunched as a clean single-variable test (30k/8M, churn and lpips_from unchanged).
3. Also corrected: my ship gate was 0.857 scene-pts too lenient -- the zip carries a 6-MEMBER
   bonsai ensemble at 72.013 raw, not the 71.156 single model the launch script compared to.

### ALSO IN r33
HCM0421 q99 -> q100. Affordable only because the deadband fix SHRINKS tower JPEGs ~0.54% (dithered
corrections compress better); that plus r32's 15.49 MiB headroom pays the +6.7 MiB.
Ladder fallback retained in the builder if q100-everywhere ever exceeds the cap.

EXPECTED r33 ~ 77.707 (77.69-77.73). Companion sub_round33a_bonsailam0.zip (bonsai change ONLY,
334.33 MiB) exists as the zero-tower-risk fallback.

## *** R35 GRADED 77.7106 -- NEW BEST (+0.0199), AND IT CALIBRATES THE GATE *** [30/07 ~02:0x]

r35 = r32 with bonsai only: 7 old members + 3 new `scale_reg=0.1` members. Towers and chair
byte-verbatim, so the LB delta is cleanly attributable to ONE scene.

    r32  PSNR 26.640357  SSIM 87.3191  LPIPS 11.2230  ->  77.6907
    r35  PSNR 26.649719  SSIM 87.3337  LPIPS 11.1985  ->  77.7106

    dPSNR +0.00936 dB x 0.6   = +0.0056
    dSSIM +0.000146   x 30    = +0.0044
    dLPIPS -0.000245  x (-40) = +0.0098
                        total = +0.0198   (observed +0.0199)

All three metrics up, none paying for another -- the signature of a real model change rather than
a trade-off shuffle. First gain after three straight losses (r33 -0.0149, r33a -0.0041,
r34 -0.0052), so 77.6907 is no longer a local optimum in every measured direction.

### THE CALIBRATION -- which local surface actually predicts the leaderboard

Scene gain = 0.0199 x 7 = +0.139. Against the three surfaces measured BEFORE submitting:

| surface | predicted scene gain | ratio to LB |
|---|---|---|
| diversity-matched, encoded **k=2 gate** | +0.1653 (t=2.12, 17/28) | **0.84** |
| single-arm eval-split A/B | +0.188 (2 replicates) | 0.74 |
| ensemble **mixsweep** | +0.309 | 0.45 |

The pre-registered gate was the only one close. Structural reason: the mixsweep varies ensemble
COMPOSITION, so it is partly a pool-dependent operator and decays like one; the k=2 gate holds
diversity fixed and varies only the training recipe, isolating the per-model effect. This is a
THIRD transfer class alongside the existing rule: per-image ~1x, **model-recipe ~0.85x**,
pool-dependent ~1/20. Forecast recipe changes from the k=2 gate x 0.85 / 7. Never from a mixsweep.

### SCALE_REG CURVE -- COMPLETE, UNIMODAL, AXIS CLOSED

All on the 28 held-out bonsai holes, bar = 71.799 (the corrected bar, not the stale 70.7965 that
several arm logs still print):

    scale_reg   0      70.7239   -1.075
                0.003  71.5086   -0.290
                0.01   71.6951   -0.104   (default)
                0.03   71.8718   +0.073
                0.1    71.9911 / 71.9816  +0.192 / +0.183   <- PEAK, two replicates
                0.3    71.7553   -0.044

`sr03` DID run -- queue3 relaunched it after queue2 died silently, so the earlier note that it was
never launched is wrong. 0.3 is past the peak. The COEFFICIENT axis is closed; only member COUNT
at 0.1 remains open. Same batch, same bar: `regstop15k` +0.081, `aniso001` -0.142, both under the
0.407 floor -> nulls.

### FASTGS ORIGINAL ON BONSAI -- TRACK A DEAD, DECISIVELY [30/07 01:09]

30k iters in 3m22s, 124,998 gaussians.

    FastGS    PSNR 26.5447  SSIM 0.8377  LPIPS 0.3244  ->  68.0797
    best arm  PSNR 26.9814  SSIM 0.8503  LPIPS 0.2448  ->  71.9029

Gap 3.82 decomposes as LPIPS -3.18, SSIM -0.38, PSNR -0.26: it is ENTIRELY perceptual. Part is
that FastGS carries no LPIPS loss, but the ceiling on retrofitting one is measurable -- our own
lam 0.01->0.1 moved LPIPS only 0.2601->0.2548, so 0->0.1 buys at most ~0.02 LPIPS = +0.8. We need
+4.7. Dead even on generous accounting. Its aggressive pruning starves the one already-starved scene.

### R36 BUILT + VERIFIED -- bonsai 10 -> 13 members [30/07 09:34]

Pure ADD: same 7 old, all SIX lam=0.1 production members (111/555/777 + 222/333/999, the latter
three recovered this morning after the prod queue was killed to hand GPU1 to UBS). k=13 passed to
the restorer, matching the true member count -- the ledger caught a k=8-for-9 mis-specification on
HCM0421 and this avoids repeating it. lam stays 0.25 (r33a shipped 0 and LOST).

    350,415,283 B = 334.18 MiB, 386/386 files, CRC OK, VERIFY PASSED
    scenes changed vs r35: ['bonsai']   towers + chair byte-verbatim

Expected +0.005..+0.015 only. There is NO eval measurement of 6-new compositions (the eval side
only ever got 3 lam=0.1 arms, and two share seed 42). Justification is that at fixed 3 new the
mixsweep is monotone in member count (k=5 72.2032 < k=6 72.2265 < k=8 72.2527), pure-new is worse
than mixed (3-new-only 72.0247), and ADD-not-REPLACE is LB-proven (r20 +0.0948). The spread across
all five mixes containing new members is only 0.15, so the signal is "include them", not the ratio
-- which is why no GPU time was spent on a finer composition sweep.

### IDEA LEDGER MINED -- 85 closed / 4 in flight / 25 untried [30/07 02:2x]

10 agents over all 4442 lines + logs + scripts + dirs + memory, 959 raw records, deduped.
Written to /mnt/d/avv/IDEA_LEDGER.md. Top untried: refit the lens field on ENSEMBLE-MEAN train
renders (+0.03..0.05, per-image so ~1x transfer) -- BUT its cost estimate is wrong, because every
production ckpt was deleted after render, so it needs retrains, not 2-4 GPU-h. Then: fix the
HCM0421 --k defect (20 min CPU), and GT-free harmful-member removal by LOO deviation (1h CPU).

Defects the ledger surfaced that change how older numbers must be read:
  1. The public harness pool contains a BYTE-IDENTICAL DUPLICATE member. De-duplicating is worth
     +0.0171 on the harness, and every published k-curve -- including the k=10 argmax that cost
     r30 -- is contaminated by it and was never recomputed. This competes with the standing
     "member quality, not depth" explanation for r30.
  2. `eval_score.py` DRIFTED after 17/07: byte-identical input scores 0.95 lower. Eval-split
     numbers across that boundary are incomparable.
  3. FABRICATION RISK x2: the 120k-iteration arm's curve file stops at iteration 5000 yet its
     closure line cites exp31b's warm-start numbers; and Charbonnier/Huber is listed twice as
     "tested and dead" with no arm, score, or table anywhere on disk.
  4. The 0.407 noise floor is a TRAINING-RUN floor. The LB is deterministic and has resolved
     0.0019-0.0030. Do not use 0.407 to dismiss a graded LB delta.
  5. Unexplained artefacts: data_supervision_probes (18 scripts, no result logs), cvclassic,
     an unrun ssaa_sweep. Any conclusion citing them is unsupported on disk.

### TOWER SCALE_REG SCREEN LAUNCHED [30/07 09:30]

The highest-leverage untested extension by a wide margin: scale_reg is worth +0.139 on the bonsai
scene, and towers are FIVE of seven scenes. Same rate would be 5 x 0.139 / 7 = +0.099 LB.
Two pre-registered reasons it may not transfer: (a) towers are not capacity-starved, so the
mechanism may be absent; (b) towers run 60k iters vs bonsai's 30k, and budget-matching says port
lam=0.05 while balance-matching says 0.1 -- and 0.3 was already past the peak on bonsai, so
guessing high costs. Therefore a 3-point SCREEN (0.01 / 0.05 / 0.1, single arms, shipped --ut
60k/8M/ema.999 recipe, scale_reg the only variable) rather than one guessed value. Single arms sit
under the 0.407 floor so NO ship decision comes from this; a peak earns the k=2 paired gate first.

## *** R36 GRADED 77.7230 -- NEW BEST (+0.0124). ENSEMBLE DEPTH IS LIVE, AND NEARLY SPENT *** [30/07]

r36 = r35 with bonsai 10 -> 13 members (same 7 old, 3 -> 6 new scale_reg=0.1). Towers and chair
byte-verbatim, so again one scene is cleanly attributable.

    r35  PSNR 26.649719  SSIM 87.3337  LPIPS 11.1985  ->  77.7106
    r36  PSNR 26.654971  SSIM 87.3474  LPIPS 11.1856  ->  77.7230

    dPSNR +0.005252 dB x 0.6   = +0.003151
    dSSIM +0.000137    x 30    = +0.004110
    dLPIPS -0.000129   x (-40) = +0.005160
                         total = +0.012421   (observed +0.0124)

Two graded wins in a row. Scene gain +0.0868. Return per 3-member step on bonsai:
+0.1393 (members 8-10) -> +0.0868 (members 11-13), **ratio 0.62**. Geometric continuation puts
ALL remaining bonsai ensemble depth at ~+0.019 LB (next 3 = +0.0077) for ~4-5 GPU-h a step.

**This settles the r30 question.** Adding CURRENT-generation good members pays; r30's k=8->10 loss
was the two stale members, not depth saturation. The duplicate-member contamination the ledger
found in the harness k-curve no longer blocks anything -- we have a direct LB measurement.

**It also bounds the ensemble path out of the bonsai problem: it does not exist.** Everything left
in averaging is +0.13 on the bonsai SCENE. The user's target is +6.1. The ceiling is the model.

## *** THE BONSAI DIAGNOSIS: IT IS NOT A WEAK RECONSTRUCTION, IT IS A TEXTURELESS ONE *** [30/07]

Best single arm per scene on its own eval split:

| scene | PSNR | SSIM | LPIPS | Score |
|---|---|---|---|---|
| **bonsai** | **26.9814** | **0.8503** | **0.2448** | 71.906 |
| HCM0181 tower | 24.2410 | 0.8480 | **0.1139** | 75.429 |
| chair | 25.1490 | 0.7942 | 0.2277 | 69.807 |

**bonsai has the BEST PSNR and the BEST SSIM of the three.** Its entire deficit is LPIPS, which is
2.15x the tower's. An earlier framing in this session -- "PSNR would need +10.2 dB, SSIM would need
to exceed 1.0, so only LPIPS is available" -- was arithmetically true for the single-metric
scenarios but badly misleading: it implied bonsai is weak everywhere. It is not. It is a model with
tower-class geometry and no texture.

Give bonsai the tower's LPIPS and change nothing else: **77.14**. Reach 78 at LPIPS 0.0924 with
PSNR/SSIM held; the requirement relaxes fast if they move at all:

    dPSNR +0.5 dB, dSSIM +0.01  ->  need LPIPS 0.1074
    dPSNR +1.0 dB, dSSIM +0.02  ->  need LPIPS 0.1224
    dPSNR +2.0 dB, dSSIM +0.04  ->  need LPIPS 0.1524   <- ABOVE the tower's 0.1139

So 78 = (28.98 dB, 0.890, 0.152). No term in that is beyond what another scene in this same
dataset already achieves. It is a hard target, not an impossible one.

### THE MECHANISM: the training images are soft and mutually inconsistent

Measured on the raw competition photos (all 248 bonsai train frames; 40-frame samples elsewhere,
where within-scene variance is negligible):

| | lapvar median | HF spectral fraction | within-scene p90/p10 |
|---|---|---|---|
| bonsai | **19.5e-4** | 0.029 | **7.7x** |
| chair | 358e-4 | 0.058 | 4.1x |
| HCM0421 / HCM0644 | 424 / 463e-4 | 0.103 / 0.112 | 1.33 / 1.49x |

bonsai's photos are ~22x softer than the drone photos AND vary 7.7x among themselves. A model
fitting mutually inconsistent blur converges to the blur-average: low-frequency content is right
(PSNR 26.98, SSIM 0.8503 -- both best in class) and high-frequency texture is absent (LPIPS 0.2448).
That is exactly the observed metric signature, and it explains the in-sample floor (0.2065 of the
0.2448 held-out) that no amount of extra optimisation has moved -- 60k iters is WORSE than 30k
(70.72 vs 71.16).

### THE EXPLOITABLE STRUCTURE

Every scene is a sequential capture with the test frames as INTERIOR HOLES: towers are consecutive
DJI stills (index gap 1), chair is stride 5, bonsai is stride 10 over frame_000000..002750 with all
275 union gaps exactly 10. And per-frame sharpness is strongly autocorrelated in time:

    bonsai  Pearson r +0.70 @ lag 10   -> predict a hole's log-lapvar from its +-10 neighbours: R^2 0.585
    chair   Pearson r +0.87 @ lag  5   -> R^2 0.826

So **the blur of a held-out frame is predictable from train photos alone** -- legal under Rule 10.
Two attacks follow: a per-frame render-time operator (transfers ~1.0x) and, with far more headroom,
factoring the blur out during training (per-view kernel, BAD-Gaussians style) so the gaussians stay
sharp. Sharpness does NOT correlate with camera speed measured over the 10-frame stride
(Spearman +0.05), but that stride is 1/3 s and cannot resolve intra-exposure motion, so this
neither confirms nor rules out motion blur; autofocus breathing fits equally well.

CONFOUND STILL OPEN: the 22x could be CONTENT (indoor wall vs steel lattice) rather than BLUR.
The workflow's adversarial critic is running a check that separates them. If content explains it,
most of this direction is misdirected.

Workflow `bonsai-to-78` (wf_37060394-78d, 9 agents, 3 phases, CPU-only) launched 09:52.

## *** THE BLUR THESIS IS DEAD. THE DRIVER IS TRAINING-VIEW COVERAGE *** [30/07 11:xx]

Workflow `bonsai-to-78` (4 agents, CPU-only) returned. It killed the thesis I had spent the
morning building, and replaced it with a better one.

### What was refuted, with numbers

**Per-frame and global render-time SHARPENING — dead by a factor of 25 to 120.**
Oracle (GT-peeking) ceiling **+0.24**; honest legal value **+0.05**; against a +6.01 target. The
grid has no interior optimum — monotone decreasing in both sigma and alpha out to -5.47. The
MSE-optimal radial filter *gains* PSNR +0.33 dB and SSIM +0.004 and still nets **-0.71**, because
LPIPS moves +0.0257 against it. **The 0.4-weighted LPIPS term vetoes any global frequency
re-weighting however well fitted.** And the helpful direction is NEGATIVE alpha — mild blur. The
render is marginally OVER-crunchy, not under-sharp. Its missing high-frequency content is missing
STRUCTURE, not missing GAIN: amplitude a filter can restore, structure it cannot.
This is also the first time ANY sharpening operator has been measured on a bonsai render — the
entire prior "sharpening family is dead" body of evidence was HCM0181 tower measurements, and two
of its four numbers have no source on disk.

**Per-frame BLUR prediction and compensation — dead.** The legal predictor works (R^2 0.62-0.75 for
a hole's log-lapvar from its +-10 train neighbours) and is worthless: the render already reproduces
65% of the predictable blur variance, and partial corr(GT sharpness, render sharpness | predicted
sharpness) = **-0.038**. The predictor knows exactly and only what the render already knows.
`blur_bound.py` had already implemented this mechanism and scored -0.006, with median fitted
a = -0.71 — it wanted to blur the render FURTHER.

**The blur thesis itself was a confound.** `lapvar` correlates **+0.786 with median scene depth** —
distant views of a room carry more detail per pixel. So "sharp frames score badly" was "far frames
score badly".
    score ~ gt_lapvar                     Spearman -0.468 (p=0.012)
    score ~ gt_lapvar | frame index                -0.015 (p=0.94)
    score ~ contrast-normalised sharpness          +0.154 (p=0.43)   <- the clean axis is NULL
The 22x cross-scene softness gap vs the towers is real and stands; what is refuted is that
within-bonsai blur variation drives the within-bonsai score spread.

**The MIRROR hypothesis — real feature, not the dominant error.** The scene genuinely is a bonsai on
a glass table with a black display panel lying on it (I had never once looked at an image before
today; the round2 memory had recorded "glossy black glass table — mirror reflections" on 16/07 and
I re-derived it from scratch). But tile error tracks GT texture at Spearman **+0.757**, and the flat
mirror surface sits in the LOWEST-texture, LOWEST-error quintile (render/GT HF ratio 0.60-0.78
there, vs 0.219 in the top texture quintile). A planar mirror's virtual image is a valid static 3D
scene and gsplat already builds it behind the plane. The mirror hurts only by DOUBLING the amount
of fine texture in frame — through the texture channel, not a specular failure mode.
Caveat: no mirror mask was built, so the region was never isolated; residual-after-texture analysis
was inconclusive (R^2 0.57, residuals scattered, no clustering on the table).

### What actually drives it: TRAINING-VIEW COVERAGE

The 8 worst holes are **exactly the first 8 holes** (frames 10,120,190,260,440,510,630,710 — a fast
far-field sweep at the start of the capture). Clean separation: 8th-worst 66.00, 9th-worst 68.82.

    first8 mean 62.0041      last20 mean 75.9861      gap 13.98
    d_mean5 (mean dist to 5 nearest train cams)   first8 0.842  vs last20 0.279   3.0x
    train cams within 0.5 units                   first8 1.5    vs last20 8.35    5.6x
    score ~ log(d_mean5):  R^2 = 0.679,  Spearman -0.831 (p<1e-4).  Adding lapvar: +0.096 only.

The real private test set has the same structure — its first 10 test frames are 80, 90, 280, 320,
370, 380, 530, 550, 740, 880, the same sparse early segment. So the intervention transfers.

**Honest ceiling arithmetic:**
    first8 -> median          75.43   (57% of the gap)
    first8 -> last20 mean     75.99   (66%)     = +0.57 LB
    first8 -> best frame 81.98  77.70 (95%)     STILL NOT 78
So 78 needs BOTH the coverage fix AND ~+2 on the well-covered 20. The second is the near-uniform
mid/high-frequency deficit: bonsai reproduces 0.987/0.703/0.447/0.330/0.432 of GT band energy
DC->Nyquist (noise-corrected) against a tower's 0.994/0.938/0.874/0.878/1.420. bonsai GT sensor
noise is 0.62/255, so it is not noise. LPIPS is spatially DIFFUSE (worst 10% of 128px tiles carry
13.9%, Gini 0.124); the only real structure is vertical — bottom third 39.9% of LPIPS mass.

### RECORD CORRECTION: the 17/07 scorer drift is real and bigger than cited

Measured at **-0.947 / -1.149 on byte-identical input**. Consequence: the 8M-cap kill is **-1.713**
vs the corrected bar, not the -0.7101 the record cites, and the whole 17/07 capacity ladder
(0.5M 69.571 -> 5M 71.156, the one still quoted as "still climbing") sits ~1.0 above today's scale
and cannot be compared to any new arm until re-based. A re-basing job is queued (CPU, ~25 min).

## *** BOTH GPU JOBS LOST TO A PROCESS EXIT *** [30/07 ~11:37]

The previous Claude Code process exited and took every child with it:
`sr001` (tower screen, 2h07 in, no ckpt) and UBS-6D cap5M (**10h30 in**, no ckpt, nothing on disk
past its 00:59 init files). Both were launched as plain background jobs of that shell. Combined
loss ~12.5 GPU-hours with zero output. UBS is NOT being restarted — its pre-registered kill bar of
>=72.80 was independently judged unreachable, and it had already blocked GPU1 for ten hours.

Fix: every queue from here is launched with `setsid nohup ... < /dev/null & disown`, giving it its
own session so it survives the parent. Verified by checking the SID differs from the launcher's.

Also caught before it cost anything: `arm(){ local TAG=$1 SR=$2 M=$OUT/$TAG` silently expands
`$TAG` to EMPTY — bash does not see an earlier assignment inside the same `local` statement.
The output went to `$OUT/` and the first arm's `.DONE` marker would have skipped arms 2 and 3.
Split into two `local` statements. (The original script used `$OUT/$1`, which is why it worked; my
rewrite introduced the bug.)

## RELAUNCHED, DETACHED [30/07 11:5x]

GPU0 `/mnt/d/avv/r43_bonsai/` — six single-variable arms on scale_reg=0.1 seed 42, reference
sr01 = 71.9911 (matched paired baseline, NOT the 71.799 default-scale_reg bar), ~1h35m each,
each reporting **first8 / last20 separately** because a scene average hides the 13.98-point gap:
  churn_r25n15, churn_r25n25 — refine_stop/noise_stop. This axis has exactly TWO points on it:
    capD's 15k/8k, chosen mid-incident on 17/07 and never revisited in 13 days, and "standard"
    (= the fog collapse). Same class of untouched defensive default that just paid +0.139 via scale_reg.
  opreg003, opreg03 — opacity_reg has NEVER been swept on bonsai, and only ~1.3M of the 5,000,000
    saved gaussians have opacity > 0.05: 74% of the budget is invisible.
  minop001, minop002 — the MCMC relocation threshold, same waste attacked from the recycling side.
    chair collapsed at 0.02, so minop002 is the deliberate risk arm of the batch.
Capacity is deliberately absent: 16 GB cards, and 8M is now -1.713 re-based.

GPU1 `/mnt/d/avv/r41_towersr/` — tower scale_reg screen restarted, 0.01/0.05/0.1, ~2h45m each.

Workflow `bonsai-coverage` (wf_40a2b652-786, 5 agents, CPU-only) running: coverage intervention
design, the structure-deficit direction (incl. whether densification can be made error-guided
rather than opacity-guided, and what difix_run.py actually is), the scorer re-basing, plus an
adversarial critic whose load-bearing job is to test whether "sparse coverage" is itself just
"frame index" in new clothes — the exact confound that killed the blur thesis three hours ago.

## COVERAGE SURVIVES THE CONFOUND TEST -- BUT IT IS NOT THE WHOLE STORY [30/07 13:2x]

A3 reported score ~ log(d_mean5) at R^2 0.679 but never partialled it against frame index -- the
exact omission that let the blur thesis stand for three hours. Ran it on A3's own bonsai_geom.csv
(28 holes, quadratic control in frame index, dof 24):

    score ~ log d_mean5        | frame_idx   r -0.524  t -3.02  p 0.003   SURVIVES
    score ~ log d_nearest      | frame_idx   r -0.499  t -2.82  p 0.005   SURVIVES
    score ~ log ang_nearest    | frame_idx   r -0.488  t -2.74  p 0.006   SURVIVES
    score ~ n_within_0.5       | frame_idx   r +0.288           p 0.141   dies
    score ~ log temporal_basel | frame_idx   r -0.227           p 0.253   dies
    score ~ med_scene_depth    | frame_idx   r -0.060           p 0.768   dies
    score ~ log baseline/depth | frame_idx   r -0.234           p 0.239   dies
    (for contrast, the blur thesis: score ~ lapvar | frame_idx = -0.015, p 0.94)

Camera coverage is a REAL independent driver -- three related distance metrics all survive at
p<0.01, and it is continuous distance that matters, not the hard count within a radius.

**But the reverse test is stronger: score ~ frame_index | log d_mean5 = +0.744, t +5.46.**
Frame index keeps MORE unique variance than coverage does. Coverage is *a* driver, not *the*
driver, and A3's "coverage is the primary axis" overstates it.

### Seven candidate causes for the residual, all killed

Decomposing the first8 PSNR deficit using A3's cause.csv (best global affine colour transform and
best 2D shift, both fitted per frame ON the GT, i.e. generous oracles):

                            first8   last20      gap
    PSNR raw                 22.60    28.68   -6.08 dB
      + best colour fix      +0.07    +0.25              (helps the GOOD frames more)
      + best 2D shift fix    +0.02    +0.04
    after BOTH               22.67    28.94   -6.27 dB   <- the gap gets WIDER
    SSIM raw                 0.704    0.905   -0.201
      + colour fix          -0.0006  +0.0002

**The first8 deficit is irreducible by any photometric or registration correction.** Not exposure,
not misregistration, not depth, not blur. And none of the removable defects explains the residual
frame-index effect (colour-fix gain p 0.95, shift-fix gain p 0.39, |shift| p 0.77).

SfM initialisation density is also ruled out -- and in the opposite direction to the guess:
    SfM points inside the frustum:  first8 37,536   last20 30,281   (first8 sees 24% MORE)
    score ~ log n_sfm | frame_index = r +0.234, p 0.24  -- dies
(bonsai has only 54,422 points total, "glass table kills SIFT", but they are not concentrated away
from the bad holes.)

Joint model score ~ 1 + idx + idx^2 + log d_mean5 + log n_sfm reaches R^2 0.872, but the idx terms
carry most of it. **Something about the first third of this capture is badly reconstructed for a
reason not yet identified**, after testing coverage, angular baseline, temporal baseline, scene
depth, sharpness, exposure, registration and SfM density.

Practical consequence: the coverage-weighted-loss experiment is justified (p=0.003), but should be
expected to capture PART of the 13.98-point first8 gap, not all of it. Pre-register the threshold
on the first8 mean, and do not read a scene average.

## WORKFLOW LOST TO QUOTA [30/07 ~12:5x]

`bonsai-coverage` (wf_40a2b652-786) returned {design:[], critic:null, plan:null} -- all 5 agents
failed with "session limit". 133,808 subagent tokens spent, zero results. The coverage-vs-frame-index
test above was then done inline instead, which is where the p=0.003 and the seven kills come from.

## *** THE CHURN WINDOW IS LIVE, AND THE GAIN LANDS EXACTLY ON THE SPARSE HOLES *** [30/07 17:2x]

bonsai eval split, 28 holes, reference sr01 = 71.9911 (scale_reg=0.1 seed 42, matched paired
baseline). Every arm changes ONE variable and reports the two populations separately.

| arm | SCORE | d vs ref | first8 | last20 | N opacity>0.05 |
|---|---|---|---|---|---|
| ref sr01 (refine 15k / noise 8k) | 71.9911 | -- | 62.0041 | 75.9861 | ~1.30M |
| churn_r25n15 | 72.1273 | **+0.1362** | 62.9190 | 75.8107 | **1.95M** |
| **churn_r25n25** | **72.2931** | **+0.3020** | **63.2695** | 75.9025 | **1.94M** |
| opreg003 (opacity_reg 0.003) | 71.6799 | -0.3112 | 61.6938 | 75.6743 | 1.80M |

**Three things line up that did not have to.**

1. **Monotone in the churn window.** 15k/8k -> 25k/15k -> 25k/25k gives 0 -> +0.136 -> +0.302.
   Two arms, one direction. A single arm at +0.302 sits under the 0.407 training-noise floor and
   is not evidence alone; two arms on a monotone trend with a mechanism and a control is.

2. **The mechanism is visible in the census.** Effective gaussians (opacity > 0.05) go
   **1.30M -> 1.95M**, a 50% increase, against an unchanged 5,000,000 cap. The "74% of the budget
   is invisible" finding was real and the churn window was what was suppressing it. MCMC needs
   time to relocate dead gaussians into useful ones; freezing topology at 15k of 30k denied it that.

3. **The gain is ENTIRELY in the sparse-coverage population**, which was identified independently,
   before this ran, by a partial correlation:
        first8  62.0041 -> 63.2695   **+1.2654**
        last20  75.9861 -> 75.9025    -0.0836
        8/28 x 1.2654 + 20/28 x (-0.0836) = +0.3018   (observed +0.3020)
   Diagnosis and intervention agree on WHICH FRAMES, not just on the average. That is much stronger
   than a scene-averaged delta of the same size.

`opacity_reg` moves the other way at 0.003 (-0.311, and first8 -0.31), so it is not a generic
"more gaussians is better" effect -- it is specifically the relocation schedule.

Forecast if it holds: model-recipe class transfers ~0.85x, so scene +0.257 -> **LB +0.037**, about
3x r36. Requires the diversity-matched encoded k=2 gate before shipping. NOT gated yet.

### What this reopens: the 60k kill is confounded

The record kills 60k iters for bonsai (70.72 vs 30k's 71.16). But that arm ran under the STARVED
churn schedule. 25k/30k is 83% of the run; the shipped tower recipe uses 50k/60k = 83%. So
churn_r25n25 IS the tower ratio at 30k, and **bonsai at 60k with the tower's churn schedule has
never been run**. If longer training was only losing because topology froze at 25k, that kill is
an artefact of the defect just fixed. Queued as `long60k` (~3h10).

## TOWER SCALE_REG SCREEN -- DOES NOT TRANSFER. AXIS CLOSED. [30/07 17:19]

HCM0421 eval split, 40 holes, shipped tower recipe, scale_reg the only variable:

    sr001  scale_reg 0.01 (control)  PSNR 25.1107  SSIM 0.8403  LPIPS 0.1132  **75.7468**
    sr005  scale_reg 0.05            PSNR 24.9229  SSIM 0.8356  LPIPS 0.1185  **75.2812**  -0.466
    sr01   scale_reg 0.1             running

Monotone DOWN, and -0.466 is above the 0.407 floor, so it is a real loss rather than noise. There
is no peak, so the pre-registered k=2 gate is never reached and no ship decision arises.
**The +0.139-per-scene bonsai win does not port to the towers**, and the pre-registered reason (a)
was right: towers are not capacity-starved, so the mechanism that pays on bonsai is absent there.
This closes the axis and saves the ~14 GPU-h of production retrains it would have triggered.
Note the asymmetry it implies: bonsai responds to regularisation and schedule changes that towers
do not, because bonsai is the only scene where the gaussian budget is the binding constraint.

## Q1 COMPLETE -- churn is the only winner, and the census was NOT the mechanism [30/07 21:5x]

| arm | SCORE | d vs ref | first8 | last20 | N opacity>0.05 |
|---|---|---|---|---|---|
| ref sr01 (refine 15k / noise 8k) | 71.9911 | -- | 62.0041 | 75.9861 | ~1.30M |
| churn_r25n15 | 72.1273 | +0.1362 | 62.9190 | 75.8107 | 1.95M |
| **churn_r25n25** | **72.2931** | **+0.3020** | **63.2695** | 75.9025 | 1.94M |
| minop001 (min_opacity 0.01) | 71.8204 | -0.1707 | 62.5494 | 75.5288 | 1.29M |
| opreg003 (opacity_reg 0.003) | 71.6799 | -0.3112 | 61.6938 | 75.6743 | 1.80M |
| opreg03 (opacity_reg 0.03) | 70.6868 | **-1.3043** | 60.8592 | 74.6179 | **0.66M** |

**`opacity_reg` is unimodal with its peak exactly at the existing default 0.01** -- both directions
lose, 0.03 catastrophically (N collapses to 662,680, half the default). Axis CLOSED, current value
already optimal. **`min_opacity` 0.01 loses -0.171** with N unchanged, so raising the relocation
threshold recycles nothing useful. Both "recover the invisible 74%" knobs FAIL.

### CORRECTION to this morning's mechanism claim

I wrote that the census jump 1.30M -> 1.95M *was* the mechanism behind the churn win. That is not
supported. **opreg003 is a direct counterexample: 1.80M visible gaussians, 38% more than the
reference, and it scores 0.31 WORSE.** Visible-gaussian count does not determine score. What the
churn window buys is relocation TIME -- placement quality -- and the count increase co-occurs
rather than causes. The distinction matters because it predicts that any other route to a higher
count (which is what both opacity knobs are) will not pay, and that is exactly what happened.

### The asymmetry is unique to the winner

The three losing arms lose roughly EVENLY across both populations (opreg003 -0.31/-0.31,
minop001 -0.45/-0.46, opreg03 -1.14/-1.37). Only churn is asymmetric:
    churn_r25n25   first8 **+1.2654**   last20 -0.0836
It specifically repairs the sparse-coverage holes and costs a little on the dense ones. That is a
targeted effect matching a diagnosis made before the arm ran, not a general quality shift.

Honest magnitude: +0.302 scene x 0.85 / 7 = **+0.037 LB**, roughly 3x r36 and the best single-arm
recipe result this campaign has had on bonsai -- but it is 5% of the +6.01 needed for 78.

### Queued 21:58, both detached
GPU0 (frees ~22:15): `churn_r25n25_s101` -- THE GATE ARM. churn_r25n25 is a measurement, not
something shippable; the pre-registered diversity-matched encoded k=2 paired gate needs two members
per side and the reference side already has both (sr01 s42, sr01_s101 s101). This single run is the
only thing between the result and a ship decision. Then `long60k`.
GPU1 (frees ~23:00 when tower sr01 ends): `churn_r29n29` (boundary check -- the churn axis is still
climbing at its last sampled point), then `long45k`.

## *** GATE PASSED, AND long45k IS THE BIGGEST BONSAI ARM OF THE CAMPAIGN *** [31/07 ~04:00]

### The pre-registered encoded k=2 paired gate -- PASSED

Diversity-matched, 2 seeds per side (42, 101), shipped ensemble rule, exact ship JPEG encode,
per-frame paired. This is the only surface calibrated against the LB (0.84x).

    ref_churn15k8k  k=2 encoded  PSNR 27.1980 SSIM 0.8512 LPIPS 0.2529  SCORE 71.7394
    churn_r25n25    k=2 encoded  PSNR 27.1957 SSIM 0.8577 LPIPS 0.2462  SCORE 72.2023
    PAIRED delta +0.4630   sd 0.8928   **t = +2.744**   wins 17/28
    forecast: scene +0.3935  ->  **LB TOTAL +0.0562**   (r36 was +0.0124, r35 +0.0199)

That is 4.5x r36 and 2.8x r35 -- the largest forecast gain since the round-2 dataset landed.

### The full bonsai table (ref sr01 = 71.9911 / first8 62.0041 / last20 75.9861)

| arm | SCORE | d | first8 | d | last20 | d | N op>0.05 |
|---|---|---|---|---|---|---|---|
| churn_r25n15 (30k) | 72.1273 | +0.136 | 62.9190 | +0.91 | 75.8107 | -0.18 | 1.95M |
| churn_r25n25 s42 | 72.2931 | +0.302 | 63.2695 | +1.27 | 75.9025 | -0.08 | 1.94M |
| churn_r25n25 s101 | 72.3335 | +0.342 | 63.3274 | +1.32 | 75.9359 | -0.05 | 1.94M |
| churn_r29n29 (30k) | 72.1304 | +0.139 | 63.5033 | +1.50 | 75.5813 | -0.40 | 2.31M |
| **long45k** | **72.7028** | **+0.712** | **64.7180** | **+2.71** | 75.8967 | -0.09 | 2.33M |
| long60k | 72.2482 | +0.257 | **65.0460** | **+3.04** | 75.1291 | -0.86 | 2.39M |
| minop001 | 71.8204 | -0.171 | 62.5494 | | 75.5288 | | 1.29M |
| minop002 | 71.4286 | -0.563 | 62.1929 | | 75.1229 | | 1.10M |
| opreg003 | 71.6799 | -0.311 | 61.6938 | | 75.6743 | | 1.80M |
| opreg03 | 70.6868 | -1.304 | 60.8592 | | 74.6179 | | 0.66M |

### THE 60k KILL WAS AN ARTEFACT OF THE STARVED CHURN SCHEDULE

The record killed 60k iters at 70.72 (-1.44 vs its bar). Re-run with a proportional churn window,
**long60k scores +0.257 ABOVE the reference**. A 1.7-point swing on the same nominal setting.
The conclusion was never about length; it was about topology freezing at 25k.

### THE LENGTH AXIS IS A TRADE-OFF BETWEEN THE TWO POPULATIONS

    iters / churn        first8      last20      net
    30k  25k/25k         +1.27       -0.08      +0.302
    30k  29k/29k         +1.50       -0.40      +0.139
    **45k  38k/38k       +2.71       -0.09      +0.712**
    60k  50k/50k         +3.04       -0.86      +0.257

**first8 improves MONOTONICALLY with churn budget; last20 collapses past 45k.** The sparse-coverage
holes need more optimisation to resolve under-constrained geometry; the well-covered holes start
over-fitting. 45k is the top of the trade-off. Census rises monotonically too
(1.30M -> 1.94M -> 2.31M -> 2.33M -> 2.39M), confirming again that N is a symptom of churn budget
and not itself the driver (opreg003 had 1.80M and lost 0.31).

### THIS RESCUES THE USER'S PER-REGION ENSEMBLE IDEA, IN A DIFFERENT FORM

Per-TILE selection among homogeneous members is dead (measured 31/07: oracle 27.44 dB, HONEST
fit-on-half/apply-to-half **25.72 dB, -0.69 BELOW the pixel mean**; per-tile winner agrees between
halves on 12.7% of tiles vs 10.0% chance -- pure noise). But per-FRAME selection between
SYSTEMATICALLY different recipes, keyed on a statistic computable from POSES ALONE, is a different
proposition: long60k's first8 (65.0460) with churn_r25n25_s101's last20 (75.9359) gives
**72.8245**, above long45k's 72.7028. Caveat: the assignment rule is 1 bit fitted on the same 28
holes, so it needs its own gate before it means anything -- but the population split itself was
defined by d_mean5 before any of these arms ran.

### Also closed this round
`opacity_reg` is unimodal with its peak at the existing default 0.01 (0.003 -0.311, 0.03 -1.304).
`min_opacity` loses in both directions tested (0.01 -0.171, 0.02 -0.563). Both "recover the
invisible 74%" knobs fail; only the churn schedule pays.

## THE GEOM LEVER IS RUNNING -- first time since it was named on 14/07 [31/07 10:07]

Depth-Anything-V2 maps precomputed on CPU overnight (220 train_sub + 248 full, weights were
already in the HF cache from an earlier abandoned attempt). `--depth_prior` matches rendered
expected depth to the frozen mono-depth via a scale/shift-invariant Pearson correlation; its own
help text says it regularises geometry "where SfM is sparse (glass table / low texture)".
GPU1: `long45k_dp05`, `long45k_dp15` -- clean single-variable A/B against long45k's 72.7028.
GPU0: `long45k_s101` (gate arm for the new best), `long50k` (locate the length peak).

## *** long45k GATE PASSED -- FORECAST +0.1098 LB, 8.9x r36 *** [31/07 17:0x]

    ref_churn15k8k  k=2 encoded  71.7394   first8 62.2120  last20 75.5503
    long45k         k=2 encoded  72.6437   first8 65.6443  last20 75.4434
    PAIRED +0.9043   sd 2.0617   **t = +2.321**   wins 16/28
    forecast scene +0.7687  ->  **LB TOTAL +0.10981**

Noisier than the churn gate (sd 2.06 vs 0.89) because the effect is concentrated in 8 of 28 holes,
but the mean is twice as large and it clears t=2.3.

### LENGTH CURVE COMPLETE -- unimodal, peak at 45k

    30k   +0.302 / +0.342  (2 seeds, mean +0.322)
    45k   +0.712 / +0.806  (2 seeds, mean +0.759)   <- PEAK
    50k   +0.523
    60k   +0.257

CORRECTION to yesterday's note: I wrote that first8 improves MONOTONICALLY with training length.
With more points that is wrong -- 45k_s101 +3.41, 50k +2.52, 60k +3.04 -- first8 PLATEAUS around
+2.5..+3.4 past 45k and the apparent monotonicity was three single-seed points. The clean monotone
relationship is with the DEPTH PRIOR (below), not with length. The net-score curve is unimodal and
that part stands.

### *** THE DEPTH PRIOR IS A PURE DIAL BETWEEN THE TWO POPULATIONS *** [31/07 15:48]

Base long45k, depth_prior the only variable, frozen Depth-Anything-V2, scale/shift-invariant Pearson:

| dp | SCORE | d | first8 | last20 | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|
| 0 | **72.7028** | +0.712 | 64.7180 | **75.8967** | 26.9613 | 0.8634 | 0.2344 |
| 0.002 | 72.4841 | +0.493 | 65.2131 | 75.3924 | 26.6920 | 0.8650 | 0.2370 |
| 0.01 | 72.2819 | +0.291 | 66.0625 | 74.7697 | 26.2774 | 0.8715 | 0.2407 |
| 0.05 | 70.5229 | -1.468 | **66.3579** | 72.1889 | 24.5016 | 0.8695 | 0.2566 |

**Every column is monotone in the weight.** first8 UP 64.72 -> 66.36. last20 DOWN 75.90 -> 72.19.
SSIM UP 0.8634 -> 0.8715. PSNR DOWN 26.96 -> 24.50. There is no interior optimum: the prior is a
continuous dial that trades the well-covered holes for the sparse ones.

**first8 66.3579 is the best ever measured** (+4.35 over the reference; the best non-depth arm
reached +3.41). The geometry hypothesis is now confirmed BY INTERVENTION, not just by correlation:
adding geometric constraint fixes exactly the population that the partial-correlation analysis
identified, and nothing else has come close there.

The pre-registered discriminator is answered: at dp=0.002 PSNR recovers to 26.69 (vs 26.96 at
dp=0), so this is OVER-REGULARISATION with a working implementation, not a bug.

**THE PRIZE, if the prior is applied SELECTIVELY** (full weight on sparse-coverage train views,
zero on dense ones): first8 66.3579 with last20 75.8967 gives **73.171**, i.e. +1.18 over the
reference and +0.47 over long45k -> forecast **LB +0.143**. That is the next build.

This is the third time in three days that a globally-applied setting turned out to be the wrong
GRANULARITY rather than the wrong value: the churn window, the training length, and now the depth
prior. All three surfaced only after the 28 holes were split into their two populations.

## GPU IDLE 15:48 -> 17:36 (1h48) -- my fault again

Q3B2 finished at 15:48 and I had set no waiter on it, exactly the failure logged on 30/07.
Production launched at 17:36: `r45_prod`, six long45k members on the FULL 248-image train set
(seeds 42/202/404 on GPU0, 101/303/505 on GPU1), ~2h50 each, first pair ~20:30, last pair ~02:10.
These are what r37 gets built from.

## *** r37 BUILT + VERIFIED -- forecast +0.1098 LB, the largest of the campaign *** [31/07 22:07]

r37 = r36 with bonsai only: 13 carried members + the 4 new `long45k` production members = 17.
Towers and chair byte-verbatim.

    350,572,727 B = 334.33 MiB, headroom 15.67 MiB, 386/386 files, CRC OK, VERIFY PASSED
    scenes changed vs r36: ['bonsai']
    restore lam=0.25, k=17 (true member count)

Production members (full 248-image train set, 45k iters): s42 N=2,235,833 | s101 N=2,225,312 |
s202 | s303 N=2,233,087. All 28/28 pngs.

**The gate that authorised it** -- diversity-matched, encoded, k=2 paired, 2 seeds per side:

    ref_churn15k8k  71.7394   first8 62.2120  last20 75.5503
    long45k         72.6437   first8 65.6443  last20 75.4434
    PAIRED +0.9043  sd 2.0617  t = +2.321  16/28 wins
    forecast scene +0.7687 -> **LB TOTAL +0.1098**   (r36 +0.0124, r35 +0.0199)

**Composition was measured, not argued.** Encoded sweep on the 28 holes:

    3 old only      72.0247      2 new only              72.6437
    2 mid + 2 new   73.0490      1 old + 2 new           73.0522
    3 old + 2 new   73.0613      2 old + 2 new           73.0984
    3 old + 2 mid + 2 new (everything)                   73.1034   <- best

Every mix containing old members beats pure-new, so **ADD-not-REPLACE survives a 0.76 quality
gap** and the ~0.15 member-quality rule does not apply at that range. The top five mixes span
only 0.055, which is why no GPU time was spent tuning the ratio. NOT SUBMITTED.

## SELECTIVE DEPTH PRIOR -- FAILS ITS PRE-REGISTERED GATE [01/08 00:02]

Pre-registered: `first8 > 65.5` AND `last20 > 75.5` simultaneously. Result:

| arm | overall | first8 | last20 |
|---|---|---|---|
| long45k (no depth) | **72.7028** | 64.7180 | **75.8967** |
| dp 0.05 UNIFORM | 70.5229 | 66.3579 | 72.1889 |
| dp 0.05 SELECTIVE P=1 | 70.6333 | **66.6655** | 72.2204 |

first8 66.6655 is the best ever measured. last20 fails by 3.3. **FAIL.**

**The redistribution changed almost nothing** -- against uniform at the same coefficient it bought
+0.31 first8 and +0.03 last20. The trade-off curve is unmoved.

**The design flaw is mine and it is diagnosable.** I normalised the per-view weights to MEAN 1
precisely to isolate "granularity" from "magnitude" -- methodologically clean, but it *guarantees*
the well-covered views keep most of the prior (median weight 0.823 at P=1). Protecting last20
requires the dense views to receive ~zero, i.e. a MASK, not a power law: the very
constant-total constraint that made the experiment clean is what made it ineffective.

`dpsel_P2` (w 0.007-3.87, median 0.483) and `dpsel_P2_dp02` (dense views ~0.00014 effective, close
to a true mask) were still in flight when work was paused. If neither lifts last20 above 75.5,
the depth axis is closed and long45k without a depth prior remains the best recipe.

## WEIGHTS: 50 OF 62 CHECKPOINTS WERE DESTROYED BY OUR OWN PIPELINE [31/07 21:4x]

The organiser now requires the winning weights to be submitted. Inventory:

    bonsai   3/13 survive     chair 7/8 survive     towers 2/41 survive     TOTAL 12/62 (19%)

Cause: every production script ends `rm -f $M/ckpt.pt` after the model has rendered. A checkpoint
is 1.1-1.8 GB and 62 of them need ~90 GB the machine did not have. What was kept instead is each
model's rendered PNG output -- which is what every downstream stage actually consumes, so the
submission remains bit-reproducible without any checkpoint (verified: bonsai rebuilt from its 13
member archives is **28/28 byte-identical** to the shipped zip).

Also corrected here: the campaign record said "31 members". The true count is **62 distinct
trained models** -- each tower's k=8 is `r22/png_ens` (w4) + `r25_mip3d` + `r28_members`, and
`r22` is itself a mean of an `r20`/`r21` base of 5-6 models plus `r22_seed101`.

**Fixed going forward:** `q4_dpsel.sh` now archives `ckpt.pt` to `/mnt/d/avv/WEIGHTS/` before
deleting the working copy, and a preserver raced the deleter to save the two r37 production
members. Four checkpoints are now retained: `bonsai_long45k_s202`, `bonsai_long45k_s303`,
`bonsai_dpsel_P1`, plus whatever the in-flight arms archive.

Disk: 24 stale `ckpt.pt` files were deleted from `output/` (set1/public scenes) and `r2r8/`
(verified absent from the r36 provenance chain), freeing 26 GB. Manifest at
`/mnt/d/avv/WEIGHTS/deleted_ckpts.txt`. No `test_png` was touched.

## DELIVERY PACKAGE FOR THE ORGANISER [01/08 00:0x]

`REPRODUCE_r36.md` (repo root) documents environment, pipeline, exact per-scene parameters and
pitfalls. Verified rather than asserted: 10/10 script paths exist, 32/32 documented flags exist in
the scripts they are documented for, and the bonsai path reproduces the shipped zip byte-exactly.

Two scripts had to be vendored into the repo (`energy_restore.py`, `lapfuse.py`) -- they were
living in a scratch directory and would not have reached anyone. Two documentation errors were
caught before shipping: `gsplat` ships a **pure-python wheel** and JIT-compiles kernels on FIRST
USE (so `nvcc` is a RUNTIME requirement -- `pip install` succeeds without it, then training fails),
and the claim "renders for all 62 models" was false because `r20`/`r21` retained only the
already-averaged `png_ens`.

## *** DEPTH AXIS CLOSED -- the cost is GLOBAL, so no per-view weighting can separate it *** [01/08 00:39]

Full curve, base long45k, 28 holes, reference 71.9911:

| arm | overall | first8 | last20 |
|---|---|---|---|
| long45k, no depth prior | **72.7028** | 64.7180 | **75.8967** |
| dp 0.002 uniform | 72.4841 | 65.2131 | 75.3924 |
| dp 0.01 uniform | 72.2819 | 66.0625 | 74.7697 |
| dp 0.05 uniform | 70.5229 | 66.3579 | 72.1889 |
| dp 0.05 selective P=1 (w 0.102-3.31, med 0.823) | 70.6333 | 66.6655 | 72.2204 |
| dp 0.05 selective P=2 (w 0.007-3.87, med 0.483) | 70.3316 | **66.9488** | 71.6848 |

first8 66.9488 is the best ever measured (+4.93 over the reference; the best non-depth arm reached
+3.41). The geometric prior really does fix the sparse-coverage population -- that part of the
diagnosis is confirmed three times over.

**But selectivity does not decouple the populations.** Going P=1 -> P=2 raises first8 by +0.28 and
lowers last20 by -0.54 -- the same shape as raising the uniform coefficient. Redistributing the
budget behaves like *increasing* it, not like targeting it.

**Why, and this is the finding that closes the axis:** the prior's damage to last20 is not caused
by the prior being *applied to* well-covered views. It is caused by the prior distorting the
GLOBAL geometry, which every view then renders through. The gaussians are shared. A per-view
loss weighting cannot separate a benefit and a cost that live in the same parameters.
My mean-1 normalisation was suspected as the culprit after P=1; P=2 rules that out, because
halving the median weight moved last20 the *wrong* way.

Every point on the curve is below long45k. **Depth prior: CLOSED.** `long45k` with no depth prior
(72.7028, gate-passed at +0.9043 paired, forecast LB +0.1098) remains the best bonsai recipe.

`dpsel_P2_dp02` (dp 0.02, P=2) was still running at 30.9k/45k when work was paused. PRE-REGISTERED
PREDICTION, recorded before it lands: the curve is monotone in effective coefficient, so it should
fall between dp 0.01 uniform (72.2819) and long45k (72.7028) and will not beat long45k. If it does
beat it, the monotonicity claim above is wrong and the axis must be reopened.

## PAUSED [01/08 00:39]

Per user instruction, work is stopped pending organiser direction. State at pause:
- **r37 built and verified**, NOT submitted. 334.33 MiB, 386/386, forecast LB +0.1098.
- **Delivery package**: `/home/bkai/VAR2026_BTS_postverification_r36.zip`, 26.12 GiB, 4,942 files,
  CRC OK, SHA256 `9e2599f6f5c0242e817ea76584b00fdf44d1ae6754b3e03befe1637a53e3a7c3`.
  Split alternatives kept: `r36_core.zip` (351 MiB), `r36_renders.zip` (6.63 GiB),
  `r36_weights.zip` (19.12 GiB).
- **Weights now retained**, 4 archived: `bonsai_long45k_s202`, `bonsai_long45k_s303`,
  `bonsai_dpsel_P1`, `bonsai_dpsel_P2`. The `rm -f ckpt.pt` habit is fixed in the queue scripts.
- GPU0 idle by design; GPU1 finishing `dpsel_P2_dp02`. Nothing new queued.
