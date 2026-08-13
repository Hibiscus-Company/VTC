---
name: production-harness
description: public_set test GT is a production-regime scoring harness that is submetric-predictive of the leaderboard; use it instead of the eval-split proxy
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-07-28T04:56:31.652Z
---

`/mnt/d/avv/data/phase1/public_set/*/test/images` holds real test GT for 5 drone-tower scenes
whose models were trained on 100% of their train photos, and
`/mnt/d/avv/output/<scene>_*/test_poses_renders_png` holds renders at those real test poses.
Full training density + real test poses + real test GT = **production regime by construction**,
and different scenes from the graded private_set2, so zero leakage.

**Why it matters:** the old eval-split proxy trains on only 75–83% of photos, so its renders are
noisier than production. Every *cleanup-class* intervention measures too well there and inverts
in production. Confirmed casualties: JPEG encode "fix" (+0.158 proxy → −0.026 LB), EMA (+0.484 →
+0.0028, 170×), and a learned restoration U-Net. Measure on the harness and that bias disappears.

**Calibration (2026-07-27, r27):** harness predicted LPIPS −0.12 / SSIM +0.064 / PSNR +0.019;
leaderboard delivered −0.105 / +0.066 / +0.021. All three inside 15%, total +0.0743 vs +0.077
predicted. It is predictive at the **submetric** level, not just in sign.

**It overturns TRAIN-SIDE conclusions — treat those as untrustworthy for anything touching
novel-view geometry.** Three times now: (1) video-scene energy-restore lambda, which train-side
said was flat near the top and the LB scored NET NEGATIVE; (2) lens-field amplitude, where our
leave-one-view-out protocol confidently reported optimum gain 1.00 and the truth is 1.30
(see [[lens-field-correction]]). The pattern: any protocol that holds out TRAIN views cannot see
a defect the model absorbed into its geometry at train poses.

**Always score through the FULL shipped chain** (operator → field → JPEG round-trip), never on
raw PNG or a partial chain. The encode has flipped the sign of a result more than once.

## THE TRANSFER RULE — learned by losing r30 and r30c (2026-07-28)

**Sort every candidate into one of two classes before booking its harness delta.**

**PER-IMAGE** operators — act on one image, independent of the member pool. These transfer at
**~1×**: the lens field (+0.7345 on the LB), gaussian field smoothing (**5/5 public towers**,
+0.0073…+0.0095), the resampling kernel.

**POOL-DEPENDENT** operators — value scales with how *over-smoothed* the ensemble mean is.
These **do not transfer**, because **HCM0181's k=10 pool is twice as diverse as anything we
ship (4.67/255 vs our 2.20/255)**, so it over-smooths far more and benefits far more:
- q98/4:4:4 measured **+0.0504** on HCM0181's k=10 ensemble and **−0.0094 mean, 0/5 positive**
  on single-member renders — *negative on HCM0181 itself*. LB: −0.0019. See
  [[jpeg-encode-optimum]].
- The k-curve's k≈10 argmax is likewise a property of that diverse pool. Our homogeneous k=8 is
  already as over-smoothed as a diverse k≈12, so **effective k ≠ nominal k**; going 8→10 on the
  LB produced the textbook past-the-optimum signature (PSNR +0.0073, LPIPS worse +0.0309pp).

**Limitation that causes this:** only HCM0181 has a real multi-member ensemble at test poses;
the other four are single renders. So any k>1 result is **n=60 views of n=1 scene** — a
view-level SE or a 59/60 win rate is *not* transfer evidence. **Cross-validate on ≥3 public
towers before shipping.** Field warps and encodes can be cross-validated on single renders even
though ensembles cannot; do that.

**Also:** apply a haircut for estimator/pooling changes that scale with sample count — r26's
median field under-delivered at 31% because private fields pool 120 views against the harness's
60.

Related: [[eval-split-method]], [[submission-size-cap]], [[audit-practice]]
