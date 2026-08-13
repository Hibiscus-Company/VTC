---
name: ensemble-strategy
description: "Audit-confirmed render-ensemble rules for the NVS competition (member quality, averaging math, tooling)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-07-30T03:09:07.705Z
---

Render-ensemble rules, numerically confirmed by audit round 3 (2026-07-13) on HCM0181 with 6 members:

- Pixel-MEAN of independent models' renders; accumulate float32, single `*255+0.5` round (uint8 `//N` costs −0.009). Average in sRGB (linear-RGB −0.012). Median rejected (−0.08). Score-weighted mean worthless.
- **Member quality rule:** only ensemble members within ~0.15 pts of the best single model. A −0.97 member measured NEGATIVE (−0.09); a −0.28 member flat.
- **Config-jitter generates members** (metric_gate 0/1/2/3 all ≈ equal singles, decorrelated). Same-config reruns do NOT decorrelate: `safe_state()` seeds all RNGs to 0.
- Gains: mean2 +0.61, mean3 +0.78, mean4 +0.88 on HCM0181; full-set realizes ~85% of that. 5th member ≈ +0.05 (diminishing 1/N).

## ADDING MEMBERS AT k=8 LOST ON THE LB — but the cause is member QUALITY, not depth

**r30 shipped towers at k=8→10 and graded −0.0030** (PSNR +0.0073, LPIPS worse +0.0309pp).
The added members were `r17_ut7_ema999` and `r2r8_ut42` — the latter an OLDER training
generation. **Do not repeat that swap.**

**Two explanations fit that signature and I initially picked the wrong one.** (a) too much
averaging — past the k-optimum; (b) the two added members were simply worse. A later gate
settles it toward (b): at FIXED k=8 with matched solo quality, a max-diversity 8-subset beat a
min-diversity one by only **+0.0204 ± 0.0330, 26/60 wins** (t=0.62 — noise), gaining PSNR
+0.114 and giving it all back in LPIPS +0.00275. So *diversity/effective-k is not the driver*;
member quality is. This is consistent with the k=4 finding that member QUALITY predicts gain and
decorrelation does not (corr −0.985).

**r36 (2026-07-30) CONFIRMS (b) on the LB and prices the depth axis.** bonsai went 10 → 13
members (same 7 old, 3 → 6 new `scale_reg=0.1` members), nothing else changed, and graded
77.7106 → **77.7230, +0.0124** = +0.0868 on the bonsai scene, all three metrics up. So depth
DOES pay when the added members are current-generation and good — r30's k=10 loss was the two
stale members, exactly as (b) said. Measured return per 3-member step on bonsai:
+0.1393 (r35, members 8–10) → **+0.0868 (r36, members 11–13), ratio 0.62**. Geometric
continuation puts everything remaining in bonsai ensemble depth at only ~**+0.019 LB total**
(next 3 members +0.0077), at ~4–5 GPU-h per step. Depth is a real but nearly-exhausted axis;
it cannot move a scene by points, only by hundredths.

**Consequences for planning:**
- Training genuinely DIFFERENT-FAMILY members to raise heterogeneity is **NOT worth it** — the
  gate measured that effect at noise. That path was costed at ~12 h GPU and cancelled.
- If adding a member, select on QUALITY, not novelty, and prefer the current training
  generation. An older-generation member drags the mean.
- The harness k-curve (k=8 78.8165 / k=10 78.8653 / k=12 78.8075 / k=14 78.7380) is a property
  of ITS 10-different-recipe pool; our private pool disagrees at 2.20/255 vs its 4.67/255, so do
  not transfer its argmax directly. See [[production-harness]].

**Selection caveat:** `member_gate.py` catches COLLAPSE, not mediocrity. All 30 unused private
members passed it, including a set deviating 7.4/255 from consensus vs 2.5–3.3 for everything
shipped. Rank candidates by consensus deviation + HF energy before trusting the gate.
- Prefer lossless PNG archives (`--png_dir`) as ensemble sources (~+0.005 vs q100 JPEG).
- Private-8 member gates: A=gate1 (trained, sub_round1_champ), B=gate5 stock (queue13), C→gate2, D→gate0.
- Repo tools (committed 2026-07-13): `ensemble_renders.py` (rules baked in), `build_submission_zip.py` (single-generation JPEG from PNG archives, quality ladder to 350MB, CSV/CRC self-verify). Never re-encode an existing JPEG for submission.
[[nvs-competition-setup]] [[fastgs-env-and-build]]
