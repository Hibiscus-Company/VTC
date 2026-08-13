# CANDIDATE EXECUTION PLAN — everything concluded 20–21/07
Built 22/07 00:50 GMT+7. Freeze = 28/07. Best submission so far = r16 = 77.09550.

## Per-run GPU cost (measured, train+render+eval on eval-split)
- chair  = 2.0h  | bonsai = 1.5h | tower (HCM0181 set1 eval-split) = 2.0h
- 2 GPUs. Baselines CACHED (no re-run): chair 69.37, bonsai 71.90, HCM0181 tower (absgrad=0 gen tonight).

## OPTIMIZATION PRINCIPLES (how the arrangement minimizes wall-clock)
1. GPU-hours are the bottleneck, not my coding time — each structural change is 1-3h of my
   time but yields 1.5-5.5h of GPU work, so I implement the NEXT candidate WHILE the current
   one trains. GPUs never idle waiting for code.
2. Baseline cached once per scene, reused for every A/B on that scene (~saves 5+ GPU-h).
3. PRIMARY-SCENE-FIRST: test each candidate only on the scene its mechanism targets; expand a
   winner to other scenes only after it wins. Avoids a full candidate x scene matrix.
4. Cheapest-first within each wave: pure-config (zero code) tests are gap-fillers that launch
   whenever a GPU frees and no structural impl is ready — a dud dies for ~2 GPU-h, not a slot a
   structural test needs.
5. Scene-duration balancing: pair chair/tower (2h) against bonsai (1.5h)+filler so both lanes drain evenly.

## PHASE 0 — running tonight (booked until ~04:45)
- pose_opt v3 (LR 1e-5 + warmup): bonsai done ~02:15, chair ~02:45
- absgrad A/B on HCM0181 tower: done ~04:15 (GPU1) / ~04:45 (GPU0 = absgrad=0 tower baseline)

## PHASE 1 — pure-config A/Bs, zero implementation (~04:45 -> ~11:30, 22/07)
Each is a flag change on an existing eval-split with a cached baseline. ~13 GPU-h / 2 lanes ≈ 6.5h wall.
| # | candidate | scene | flags | GPU |
|1| init_clip on towers (audit: 21.9% pts >2x hull) | tower | --init_clip 2.0 | 2.0h |
| 2| cap_max low (we run 47-99x SfM pts vs heuristic 2-15x; 27-47% dead gaussians) | chair | --cap_max 2000000 | 2.0h |
| 3| cap_max low | bonsai | --cap_max 1500000 | 1.5h |
| 4| noise cooldown (chair CHA has only 16% noise-free tail vs bonsai 73%) | chair | --noise_stop 40000 | 2.0h |
| 5| brightness normalization (deterministic per-frame gain, NOT learned — unlike killed affine) | bonsai | preprocessed GT | 1.5h |
| 6| brightness normalization | chair | preprocessed GT | 2.0h |
| 7| SSIM window widen (if fused_ssim exposes it; else small code) | chair | (tbd) | 2.0h |

## PHASE 2 — structural code changes (implement-ahead, ~11:30 22/07 -> 24/07)
Impl times are MY time, overlap Phase-1/earlier-Phase-2 GPU runs. ~35 GPU-h / 2 ≈ 17-18h wall.
Ordered by (expected value / implementation cost), highest first:
| # | candidate | scene(s) | impl | GPU |
| 8| Mip-Splatting 3D smoothing filter (2x recommended; never regresses at native res) | tower+bonsai+chair | ~2-3h | 5.5h |
| 9| scale/anisotropy reg (max-scale-ratio cap; NOT the killed opacity/scale-reg-lowering) | chair+bonsai | ~1h | 3.5h |
| 10| sky-dome init seeding (WildGaussians; sky has ~0 SfM pts -> floaters) | tower | ~1h | 2.0h |
| 11| dead-gaussian min_opacity prune (27-47% at opacity<0.05; reclaim budget) | chair+tower | ~0.5h | 4.0h |
| 12| EFA-GS low-freq-come-first (expand under-optimized gaussians; thin legs) | chair | ~2h | 2.0h |
| 13| Pixel-GS density control (pixel-coverage-weighted; thin lattice braces) | tower | ~3h | 2.0h |
| 14| glossy-region down-weight (mask static reflective table; NOT transient masking) | bonsai | ~2h | 1.5h |
| 15| planarity reg on table (fix reflection geometry warp; diagnose first) | bonsai | ~1.5h | 1.5h |
| 16| foliage normal/depth consistency (stabilize thin-branch geometry) | bonsai | ~2h | 1.5h |
| 17| EMA of gaussian params (denoise oscillating late-training MCMC trajectory) | chair+bonsai | ~2h | 3.5h |
| 18| depth-seeded carpet init (generic pretrained depth net -> seed pts on low-texture floor) | chair | ~3h | 2.0h |
| 19| rolling-shutter render (native gsplat; per-scanline pose for handheld pans) | chair+bonsai | ~3-4h | 3.5h |

## DECISION RULE (user reframe 22/07): candidates = FAMILY BRANCHES, not replacements
A winning idea does NOT replace the recipe -- it becomes its own ensemble FAMILY BRANCH, developed
then pixel-mean-averaged into the final ensemble, exactly like set1's FastGS-family + UT-family.
This LOWERS the keep-bar dramatically:
- KEEP as ensemble member if eval-split score is within ~0.15 of baseline AND method-decorrelated
  (a genuinely different mechanism, not a seed rerun). Decorrelated members ADD via pixel-mean even
  at parity (set1 B4warm: -0.33 single -> +0.05 as a member). See [[ensemble-strategy]].
- KEEP as replacement only if it beats baseline by a structural margin (transfers ~0.7-1.4x).
- EVAL PROTOCOL per candidate: after single-model eval-split score, if within ~0.3 of baseline,
  ALSO compute the 2-member pixel-mean ensemble (candidate eval renders + best existing member's)
  and score THAT -- cheap CPU step, reuses existing renders. A candidate that is -0.2 alone but
  +0.1 as a member is a KEEPER. This is the real decision number, not the single-model score.
- LENS FIELD is post-render, applied to the ENSEMBLE MEAN (run_dataset.sh: ensemble_renders ->
  png_ens -> apply_field -> png), tower scenes only. New tower family branches just join the
  pre-field average; field logic unchanged. Video scenes: no field, members average directly.

## PHASE 3 — develop winning branches + ensemble + submissions (25-27/07)
For each KEEP (winner OR decorrelated near-miss per rule above): retrain on FULL train, render
test_poses, ADD to its scene's ensemble family (tower: pre-field; video: direct mean), build +
verify a candidate zip. Develop strong branches further (2nd seed of the winning method, etc.)
for intra-family decorrelation. Also: 5th tower seed (proven 1/N, banked-cheap).
STANDING DIRECTIVE (user 22/07): on EACH round of potential submission, BUILD THE ZIP and TELL
THE USER (they submit manually + report grades). Never auto-submit. One clean swap per zip where
possible so LB delta x7/scenes-changed = exact per-scene gain.

## PHASE 4 — freeze buffer (28/07): final per-scene argmax, verify best zip, submit, STOP.

## RUNTIME SUMMARY
- Total eval-split testing ≈ 46-48 GPU-h ≈ 23-24h wall-clock floor on 2 GPUs.
- With implementation overlapping compute and config-fillers preventing idle: ~2-2.5 days of
  testing (through ~24/07 midday), leaving 3.5 days for scaling winners + ensembling + freeze buffer.
- Comfortably fits before 28/07 with room for winners to be scaled and re-ensembled.

## DEAD — not scheduled (concluded 20-21/07)
eps2d, B3 detail-inject, texture_weight, app_affine (x2), RobustNeRF hard-mask, DWTGS,
Charbonnier/Huber, DINOv2/SpotLessSplats masking, SfM point filtering, cross-pol/matte-spray,
COLMAP re-matching, random-bg aug, opacity/scale-reg lowering, uncertainty_weight (failed both).
pose_opt v1/v2 crashed; v3 running -> if it fails too, pose_opt is dead.
