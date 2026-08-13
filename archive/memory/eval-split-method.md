---
name: eval-split-method
description: Per-scene model selection via a carefully-picked train/eval split that mimics the hidden test; validated on the LB
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
---

Per-scene recipe selection loop (user's method, LB-validated 17/07): split TRAIN → train-sub +
eval → score recipe candidates on eval → retrain winner on FULL train → render test → submit.
Rule-10 legal: eval GT = held-out TRAIN photos, eval poses from images.bin; no test pixel touched.

CAREFUL SPLIT (not random — one rule, both regimes): eval = ISOLATED, evenly-spaced frames along
the capture sequence (never adjacent). Works because the hidden test is itself an interleaved
holdout of the same capture, so mirroring its structure reproduces its novel-view difficulty.
- video scenes (bonsai/chair, `frame_NNNNNN`): grid-holes at the video stride.
- drone scenes (`DJI_*`): every-k by filename/capture order.
Validated on public HCM0181 (we own its test GT): isolated-every-k eval gap 0.121 vs test 0.116,
p75 0.176 vs 0.168 — matches median AND tail; random has too-fat a tail; arcs/widening overshoot.
Tools: `scripts/make_eval_split.py` (both regimes), `scripts/eval_score.py` (exact comp metric).

Guards: keep eval representative (done via the isolated rule), try a HANDFUL of candidates not a
big sweep (eval-overfit), calibrate the eval→test gap on the PUBLIC set (drone → same regime as
private towers). This is also the NAME-FREE router — every scene, seen or unseen, self-selects its
recipe from its own eval holes; no hardcoding by scene name. See [[round2-dataset]].

PROVEN: picked bonsai's capD recipe (eval 71.16); round10b LB jumped +3.62 exactly as predicted.
CALIBRATED (r12, 18/07): ranking transfers, but small-recipe-nudge MAGNITUDE compresses ~0.3×
(eval +0.94 scene-pts → test +0.28; full-train baseline closes part of what the candidate
exploited on train_sub). Rule: an eval win needs ≥ +0.5 scene-pts to be worth a submission slot;
big structural wins (collapse fix +25) transfer near-1:1.
