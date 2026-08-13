---
name: bonsai-capacity-starvation
description: "bonsai is the one broken scene — but the 2026-07-28 capacity diagnosis was WRONG; the measured driver is sparse training-view coverage in the first third of the capture"
metadata:
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-07-30T05:48:20.175Z
---

bonsai carries the whole LPIPS deficit: on TRAIN views (identical protocol, n=16/scene) the five
towers score 80.8–81.8, chair 81.5, **bonsai 74.7**, and its LPIPS is 2.7× every other scene.
That part still holds. The *cause* recorded on 28/07 does not.

## THE 28/07 DIAGNOSIS WAS WRONG — do not repeat it

It said: bonsai alone trains at 30k/5M while chair and every tower get 60k/8M, so restore 60k/8M.
Both halves are now measured and both LOSE:

    60k iters   70.72   vs 30k's 71.16
    8M cap      −1.713  vs the corrected bar   (the record's −0.7101 predates the scorer re-base)

Also refuted: the "17/07 ladder was still climbing at 5M" argument. `eval_score.py` DRIFTED after
17/07 — byte-identical input scores **−0.947/−1.149** lower today — so the whole ladder
(0.5M 69.571 → 5M 71.156) sits ~1.0 above today's scale and cannot be compared to anything new.

## WHAT ACTUALLY DRIVES IT (measured 30/07, eval split, 28 holes)

**Sparse training-view coverage, concentrated in the first third of the capture.** The 8 worst
holes are exactly the first 8 (frames 10–710, a fast far-field sweep). Clean separation.

    first8 62.00   last20 75.99   gap 13.98
    mean distance to the 5 nearest train cams:  first8 0.842 vs 0.279   (3.0×)
    train cams within 0.5 units:                first8 1.5   vs 8.35    (5.6×)
    score ~ log(d_mean5):  R² = 0.679,  Spearman −0.831 (p<1e-4)

The private test set has the same shape — its first 10 test frames are 80…880 — so a fix transfers.
Ceiling: lifting first8 to last20's mean gives 75.99 (66% of the gap to 78); lifting them to the
best single frame still only reaches 77.70. **78 also needs ~+2 on the well-covered 20**, which is
the separate near-uniform mid/high-frequency deficit (bonsai reproduces 33–45% of GT band energy
where a tower reaches 87–88%; sensor noise is only 0.62/255, so it is not noise).

## DEAD ENDS, with numbers — do not re-propose

- **Any render-time sharpening / unsharp / deconvolution / radial frequency matching.** Oracle
  ceiling +0.24, honest +0.05, grid monotone with no interior optimum. The MSE-optimal radial
  filter gains PSNR +0.33 dB and SSIM +0.004 and still nets **−0.71** — the 0.4-weighted LPIPS term
  vetoes any global frequency re-weighting. The helpful direction is mild BLUR: the render is
  over-crunchy, and its missing HF is missing *structure*, not *gain*.
- **Per-frame blur prediction.** The legal predictor is good (R² 0.62–0.75) and useless: partial
  corr(GT sharpness, render sharpness | prediction) = −0.038 — it knows only what the render knows.
- **The blur thesis itself.** `lapvar` is a depth proxy (+0.786 with scene depth); score ~ lapvar
  partialled on frame index = −0.015 (p=0.94); the contrast-normalised sharpness axis is NULL
  (+0.154, p=0.43). The 22× cross-scene softness vs towers is real; the within-scene effect is not.
- **Mirrors.** The scene IS a bonsai on a glass table with a black display panel on it, but tile
  error tracks GT texture (+0.757) and the flat mirror region is the LOWEST-error quintile. A
  planar mirror's virtual image is a valid static 3D scene and gsplat already builds it.

**How to apply:** the 28/07 lesson (diagnose per scene before optimising a scene-averaged metric)
was right and is what found bonsai. The new lesson is one level up: **a per-scene diagnosis can
itself be confounded.** Both the capacity story and the blur story survived because nobody
partialled the candidate cause against capture position. Before believing any per-frame
correlation on a video scene, partial it against frame index.

Related: [[gate-calibration]], [[round2-dataset]], [[eval-split-method]], [[ensemble-strategy]].
