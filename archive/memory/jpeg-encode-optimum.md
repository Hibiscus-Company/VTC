---
name: jpeg-encode-optimum
description: "DO NOT ship q98/4:4:4 — it lost on the leaderboard. Its harness gain came from ensemble heterogeneity we do not have, not from the encode"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-07-28T04:04:52.034Z
---

**SHIPPED AND LOST.** r30c used `quality=98, subsampling=0` and graded **77.6595 vs r29's
77.6644 (-0.0049)** where the harness predicted +0.05. Keep `quality=100, subsampling=2`.

## What the harness said, and why it was wrong

On HCM0181's **k=10 ensemble**, q98/4:4:4 measured **+0.0504** (t=16.96, 59/60), with a clean
interior optimum (q99 +0.0384, q97 +0.0209, q96 -0.0446). All real — and all useless.

Cross-validated afterwards on **single-member renders** of all five public towers:

    HCM0181 -0.0100 | HCM0193 -0.0038 | HCM0204 -0.0121 | hcm0031 -0.0132 | hcm0034 -0.0081
    5-scene mean -0.0094, scenes positive 0/5   (NEGATIVE on HCM0181 ITSELF)

The effect was never scene-specific — it was **ensemble-depth/heterogeneity specific**. JPEG
artifacts substitute for texture the pixel-mean destroys; a deep, DIVERSE ensemble is
over-smoothed so artifacts help, a single render already has its own texture so they hurt.

**The killer number: our private pool's heterogeneity is 2.20/255 against the harness pool's
4.67/255 — HALF.** The harness's k=10 pool is 10 wildly different recipes; ours is UT-family +
2 mip3d + an EMA + an old-generation member. Less disagreement → less over-smoothing → less
benefit from artifact substitution. The gain transferred at ~16%, and since the encode's
PSNR/SSIM COSTS transferred in full, the operator flipped negative.

## The general rule this buys

**Any operator whose value depends on how over-smoothed the ensemble mean is will OVER-READ on
the production harness**, because HCM0181's pool is twice as diverse as anything we ship. That
includes the encode, and it contaminates the k-curve (the k≈10 optimum was found on the diverse
pool; a homogeneous pool's optimum is probably lower). It does NOT affect operators that act on
a single image independent of the pool — the lens field transferred at ~1x.

**Before shipping any harness result: (1) cross-validate on >=3 public towers, (2) ask whether
the effect depends on ensemble diversity, and if so discount it hard.** A view-level SE or a
59/60 win rate on ONE scene is not transfer evidence. Contrast [[lens-field-correction]], which
was extended to all five towers before shipping and transferred at full strength.

Related: [[production-harness]], [[submission-size-cap]]
