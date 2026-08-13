---
name: gate-calibration
description: The diversity-matched encoded k=2 gate predicts LB scene gain at ~0.84x; raw ensemble mixsweep inflates ~2.2x
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-07-29T19:22:26.086Z
---

r35 (2026-07-30) closed the loop on how well each local surface predicts the leaderboard.
Only bonsai changed (7 old members + 3 new `scale_reg=0.1` members); towers and chair were
byte-verbatim, so the LB delta is cleanly attributable to one scene.

LB: 77.6907 -> **77.7106**, +0.0199 on the 7-scene mean = **+0.139 on the bonsai scene**.
All three metrics moved the right way (PSNR +0.0094 dB, SSIM +0.000146, LPIPS -0.000245),
and the ΔScore arithmetic reproduces +0.0198 vs +0.0199 observed.

**Predicted vs realised, same change, three surfaces:**

| surface | predicted scene gain | ratio to LB |
|---|---|---|
| diversity-matched, encoded **k=2 gate** | +0.1653 (t=2.12, 17/28) | **0.84** |
| single-arm eval-split A/B | +0.188 (mean of 2 replicates) | 0.74 |
| ensemble **mixsweep** (5 old vs 5 old+3 new) | +0.309 | 0.45 |

**Why:** the mixsweep changes ensemble COMPOSITION, so it is partly a pool-dependent operator and
decays like one. The k=2 gate holds diversity fixed and varies only the model recipe, so it
isolates the per-model effect, which transfers nearly intact.

**How to apply:** for a TRAINING-RECIPE change, gate on the diversity-matched encoded k=2 test and
multiply by ~0.85 to forecast the LB scene gain; then divide by 7. Do NOT forecast from a raw
mixsweep — it overpredicts ~2.2x. This is a third transfer class alongside
[[production-harness]]: per-image ~1x, model-recipe ~0.85x, pool-dependent ~1/20.

Also settled: `scale_reg` is unimodal on bonsai with a peak at 0.1, measured on the eval split
(0 -> 70.72, 0.003 -> 71.51, 0.01 -> 71.80 bar, 0.03 -> 71.87, **0.1 -> 71.99/71.98**, 0.3 -> 71.76).
The coefficient axis is closed; only member COUNT at 0.1 is still open. Caveat before porting to
other scenes: bonsai runs 30k iters while chair and every tower run 60k, so an identical
coefficient applies ~2x the total regularisation there — port the budget, not the number.
See [[bonsai-capacity-starvation]].
