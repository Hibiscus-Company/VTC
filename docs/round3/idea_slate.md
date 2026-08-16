# Round-3 Idea Slate (adversarially refereed)

> Drafted 2026-08-15, then attacked by three independent reviewers (feasibility /
> competitive-edge / record-consistency). What survives is below, with the referees'
> corrections baked in. Ideas the record killed are listed at the end WITH the numbers —
> do not resurrect them without new evidence.

## Build-now tier (before the data drops)

1. **Cell-partition platform on gsplat** — VastGaussian-style airspace-aware cells,
   overlap-expand→trim→merge, plus the eval harness. **Holdout rule: isolated
   evenly-spaced frames (every-k), NEVER contiguous segments** — Rounds 1–2 measured
   that contiguous/arc holdouts overshoot test difficulty while isolated-every-k
   matched real test GT (gap 0.121 vs 0.116). Mirror the actual test CSV structure at
   drop. Seam metric folded into the merge acceptance tests.
2. **Shared-trunk fork ensemble as the default production shape** — train one trunk per
   cell/scene, fork 2–3 members over the last ~10–25% with **config jitter** (seed
   jitter alone does not decorrelate — measured), pixel-mean in float32, round once.
   Member-quality rule: within ~0.15 of the best single, current generation only.
   (Also the SoccerNet-2026 winning recipe.)
3. **Harness arsenal port** — encoded k=2 gate procedure, zip verifier, quality-per-byte
   ladder: port the CODE day-0, but treat every carried CONSTANT (0.84× gate transfer,
   metric formula, 350 MiB cap) as a prior to recalibrate on the first Round-3 probes.
   JPEG ladder engages only when the cap actually binds. Never re-encode a JPEG.
4. **Capacity/churn tuning per cell-CLUSTER, not per cell** — cluster cells by
   coverage/density statistics, sweep 2–3 representatives per cluster, transfer within
   the cluster. (Per-cell sweeps = 100+ GPU-nights; infeasible.) On the H200 the
   >8M-gaussian capacity axis is measurable for the first time — it was never measured
   in Rounds 1–2 (the 16M attempt OOM'd on 16 GB after 13 h).

## Day-1 data-drop tier (see `day1_protocol.md` for the runnable checklist)

5. **Rig forensics = the decision node.** Camera model/k1 → **3DGUT default ON unless
   pure pinhole** (the Rounds 1–2 +0.9 was earned on SMALL k1 ≈ +0.009; do not demand
   "meaningful" distortion). Exposure-vs-capture-order → appearance gate. Sky fraction
   in test frusta → sky policy only if >10–15%. Pose reprojection sanity → trust
   organiser geometry by default. Test-CSV structure → holdout design.
6. **Lens-field refit** — scheduled unconditionally in production round 1: render train
   poses, fit the DIS field on TRAIN photos, warp renders. Assumes nothing about rig
   continuity (it measures whatever residual exists; zero residual = zero cost).
   Re-verify the ×1.30 amplitude gain on the new rig before shipping it.

## Gated contingencies (build only when their trigger fires)

- **Per-image appearance embeddings** — only if measured inter-pass exposure drift is
  clearly super-LSB. Test policy: nearest-pose training embedding (timestamps may not
  exist). Rounds 1–2 killed this class 4 ways on a photometrically stable rig.
- **Sky/background model** — only if sky exceeds ~10–15% of test-frustum pixels
  (sky dome measured NEUTRAL on the towers).
- **Per-image view-class loss up-weighting** (the SoccerNet one-liner) — only where the
  harness shows a residual sparse-coverage deficit AFTER the churn/schedule dial.
- **HF energy restoration** — near-refuted for a shared-trunk pool (low disagreement +
  uint8 deadband + the LPIPS veto on frequency re-weighting). Three rules if ever used:
  float-accumulate and round once at final encode; measure delivered amplitude in LSB;
  sweep the operator's sign per scene family and apply only where independently positive.

## Killed by the referees / the record — do not re-propose

- **Depth priors as a training loss** (uniform OR "selective"): REFUTED. Rounds 1–2
  measured depth losing to churn/length tuning even in sparse-coverage conditions — a
  pure dial between view populations with global geometry damage. Replacement:
  observation-triggered post-hoc view-consistency floater culling (no training loss).
- **Coverage-weighted training loss**: same dial by construction; replaced by the
  gated per-image up-weighting above.
- **Re-running SfM by default**: organiser poses were correct to 0.018 px; a re-solve
  must be aligned back to the organiser frame and can only add error. Cheap
  reprojection sanity check only; GLOMAP re-SfM is a contingency for measurably bad or
  absent poses. NEVER re-solve test poses.
