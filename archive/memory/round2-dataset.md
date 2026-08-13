---
name: round2-dataset
description: "private_set2 (16/07 upgrade) — 7 scenes incl. 2 indoor videos, all-native path, chair obs-scale 1.5, test-keypoint gray zone"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
---

Round-2 data (landed 2026-07-16): `/mnt/d/avv/data/phase1/private_set2`, 7 scenes.
5 HCM towers (240 train/60 test, SIMPLE_RADIAL k1≈+0.009, 1320×989, same difficulty class as set1)
+ 2 INDOOR VIDEOS: `bonsai` (248/28, PINHOLE 1920×1080, bonsai on glossy black glass table — mirror
reflections, auto-exposure drift 63 levels, blurry frames VoL med 474) and `chair` (205/58, PINHOLE
720×1280 PORTRAIT phone video, DoF blur, drift 49). Video test frames are interleaved holdouts
(every ~10th/5th frame, 3–4° from nearest train view); exposure drift is temporally smooth →
per-test-frame appearance interpolation from train neighbors is legal (frame index only).

- NO negative-k1 scenes; native render path everywhere, no warp scenes.
- Train images now at camera resolution (set1 shipped 4×). README scale: towers 1/4, chair 1/1.5, bonsai 1/1.
- chair COLMAP obs are at 1.5× delivered res — preflight fixed (obs-scale candidates now incl. 1.5); poses fine (0.63 px).
- images.bin contains TEST frames (poses bit-exact vs csv) WITH full triangulated keypoints, plus
  11–98 undelivered extra frames per tower. Set1 identical (verified). Gray zone RECHECKED and
  DEFUSED (16/07, public HCM0181): full test-KP per-image field = +0.60 scorePart, but the legal
  DIS field ([[lens-field-correction]]) already gives +1.16 and subsumes it — residual exploit
  value +0.05. Per-image translation dead (0.018px — BA poses rigidly perfect). Not using test KPs.
  BONUS: train-KP global field (+0.57, no renders needed) = legal cross-check of the DIS field
  (direction agrees, corr 0.8; DIS 2.3× larger — sees render-side systematics too).
- Pipeline compatible as-is (train_gsplat accepts SIMPLE_PINHOLE, render W/H from csv, portrait OK).
- Round-2 submission = private_set2 ONLY (user confirmed): 386 imgs ≈ 315 MB PNG.
- SET2 LEADERBOARD (released 17/07): TOP-1 = 80.56 (vs set1 top 86.12 → set2 is harder, as organizers said). Target = 80.56.
  Our scores: r10 72.485 → r10b 76.105 → r11 76.610 → r12 76.649 → r15 76.95340 (BEST, gap 3.61).
  BEST zip = sub_round15_3seedtowers_videoaa.zip (3-seed UT towers ut7+42+13 + field, AA videos).
  r15 all-tower swap proved 3rd seed = +0.232/tower (eval increment +0.17 transferred 1.4×).
  COMPOSITION changes transfer ≥1× (3rd confirmation); recipe nudges still ~0.3×.
  eps2d (sharper video) DEAD: lower eps2d worse on both video scenes — GT is genuinely blurry.
  r16 GRADED (20/07): 77.09550 NEW BEST (+0.142 vs r15, gap 3.465). Towers 3->4 seed + chair/
  bonsai 2->3 seed changed together (not isolable). Landed inside projected 77.05-77.15 band —
  3rd straight composition-move calibration hit. Diminishing 1/N returns visible: this combined
  2-mechanism step netted only +0.142 vs the +0.232 from the single 2->3 tower step in r15.
  BEST zip = sub_round16_4seedtower_3seedvideo.zip. Freeze 28/07 (8d out); target band 77.0-77.3
  ALREADY REACHED — seed-stacking alone won't clear it further, need a fresh lever.
  KEY RULE (empirical): STRUCTURAL changes (rendering-mode, collapse fix) transfer ~0.7× to LB;
  recipe nudges only ~0.3×. Video scenes: drop --ut (k1=0 → unlocks gsplat antialiased mode) = +0.49/scene.
  Gates family REGRESSED on set2 towers (-0.60, stale weights); shelved.
- BONSAI COLLAPSE FIX: cause was MCMC CHURN (noise/relocate to 50k) on view-inconsistent glossy-glass,
  NOT the cap. Winner recipe capD = cap 5M, refine_stop 15k, noise_stop 8k (30k iters). Eval-split
  model selection ([[eval-split-method]]) picked it and the LB delivered the predicted jump. bonsai
  still weakest scene (~70 vs ~77 others); LPIPS is its next lever.
- Full EDA in /mnt/d/avv/EXPERIMENTS.md ("ROUND-2 DATASET EDA").
