# CONSULT BRIEF #3 — NVS competition, stuck at 76.65 vs top-1 80.56, need a way out of the +0.04/round loop

You are consulted cold; this document is self-contained. Be maximally skeptical and concrete.
We don't need encouragement — we need either (a) a lever we haven't tried that can plausibly move
≥1 point, or (b) a well-argued verdict that the remaining gap is not reachable with our compute,
so we should optimize rank defense instead.

## 1. The competition

Per-scene novel-view synthesis of 7 scenes from drone/phone captures ("BTS Digital Twin", VAR 2026).
For each scene we get: `train/images` (~205–248 photos), `train/sparse/0` (COLMAP: cameras.bin,
images.bin, points3D.bin), and `test/test_poses.csv` (poses + intrinsics of 28–60 withheld test
images we must render). Submission = rendered images for ALL 386 test poses, zip ≤ 350MB.
System refuses to grade partial submissions.

**Metric** (per scene, then averaged over 7 scenes):
`score = 100·[0.4·(1−LPIPS_vgg) + 0.3·SSIM + 0.3·min(PSNR/50, 1)]`
LPIPS carries 40%. Sub-metric exchange rates per scene (÷7 for the mean):
ΔLPIPS −0.01 → +0.4 scene-pts; ΔPSNR +1dB → +0.6; ΔSSIM +0.01 → +0.3.

**Rules (verbatim hard line, organizer-confirmed):** no external data or imagery of these scenes;
no inferring/looking at test GT; no manual per-image editing. Pretrained VGG/AlexNet as loss or
metric IS allowed. Everything else legal. (A COLMAP quirk ships test-frame SIFT keypoints inside
images.bin — we measured exploiting them = +0.05 pts and rejected on compliance; do not resurrect.)

**Deadline:** 2026-07-30 (12 days). **Compute:** 2× RTX 5070 Ti (16GB). Submissions: ~2-3/day OK.

## 2. The scenes (private_set2 — this is the graded set)

| scene | imgs train/test | camera | content | est. our scene score |
|---|---|---|---|---|
| HCM0421/0539/0540/0644/0674 | 240/60 | SIMPLE_RADIAL k1≈+0.009, 1320×989 | dense urban rooftops, drone orbit around a telecom mast | ~79.5 avg (inferred) |
| bonsai | 248/28 | PINHOLE 1920×1080 | INDOOR video: bonsai on a GLOSSY BLACK GLASS table (mirror reflections), auto-exposure drift 63/255, blurry frames | ~70–71 |
| chair | 205/58 | PINHOLE 720×1280 portrait | INDOOR phone video: office chair (fine mesh backrest) + tissue box on carpet, DoF/motion blur, drift 49/255 | ~68–69 |

Test poses: towers = held-out orbit views (median 0.12 normalized dist / 8–11° from nearest train
view); videos = interleaved single-frame holdouts (every ~5th/10th video frame, 3–4° from flanking
train frames). Train images at delivered (camera) resolution — no higher-res source available.

## 3. Leaderboard trajectory (ours) and the loop we're stuck in

| round | change | LB score |
|---|---|---|
| r10 | 2-seed 3DGS(UT) 30k/5M + lens field; bonsai TRAINING COLLAPSED (fog) | 72.485 |
| r10b | bonsai collapse fixed (cap kept, MCMC churn stopped early) | 76.105 (+3.62) |
| r11 | towers+chair upgraded to 60k iters / 8M gaussians | 76.610 (+0.50) |
| r12 | video scenes re-tuned per-scene (longer LPIPS phase) | 76.649 (+0.04) |

**Top-1 = 80.56.** Gap = 3.91. Top-1 jumped there in one step; 2nd place is near us (exact value
unknown to us). Sub-metrics of r12: PSNR 26.14, SSIM 0.860, LPIPS 0.121 (means over 7 scenes).

The loop: our per-scene eval harness (below) reliably finds +0.2–0.7 eval-point recipe
improvements, but they compress ~0.3× on the real test (measured r12: eval predicted +0.94
scene-pts total, LB delivered +0.28). So each cycle = ~+0.04–0.1. That pace closes 0.5–1.0 of
the 3.91 gap by deadline. We need a different class of move.

## 4. Our method stack (all LB-validated)

- **Base:** gsplat MCMC 3DGS with UT (3DGUT: distorted-space rasterization, k1 from COLMAP),
  2 seeds per scene, pixel-mean ensemble. 60k iters, cap 8M gaussians, L1+0.2·(1−SSIM), VGG-LPIPS
  loss term λ0.1 in the final 10k. Loader: per-frame AE soak. SfM init.
- **Lens-field correction (towers only):** COLMAP SIMPLE_RADIAL single-k1 under-parameterizes the
  real lens; renders are misregistered from photos by a fixed sub-px field. We fit it as the mean
  dense DIS flow over TRAIN renders vs TRAIN photos and warp the test renders by it.
  Held-out train x-val: +0.11..+0.44 dB per tower. On video scenes the fitted field fails x-val
  (bonsai +0.003 junk from moving glass reflections; chair −0.13 harmful) → videos ship unwarped.
- **Per-scene eval-split model selection** (the harness): split train into train-sub + eval where
  eval = isolated holes mimicking the test structure (video: grid holes at the video stride;
  drone: every-k by capture order — validated to reproduce true test difficulty on a public scene
  where we own test GT). Score candidate recipes on eval with the exact competition metric, ship
  the winner retrained on full train. This found the bonsai collapse fix (+25 scene pts, transferred
  1:1) and the video recipe wins (+0.5–0.7 eval, transferred 0.3×).
- **Bonsai special recipe:** the scene COLLAPSES (5M near-transparent micro-gaussians, opacity→0)
  if MCMC noise/relocation churn runs past ~10k iters — glossy-glass view-inconsistent supervision.
  Fix: cap 5M, refine_stop 15k, noise_stop 8k, 30k iters. 60k iters makes it WORSE (26.43 vs 26.80
  eval PSNR). This is a hard scene-intrinsic ceiling symptom.
- **FastGS gate family (IN FLIGHT NOW):** 3 extra members/scene from a second 3DGS codebase
  (importance-gated densification, gate ∈ {1,5,2}), family-weighted ensemble
  (UT 0.3×2 + gates 0.133×3). On set1 this family effect measured +0.5–0.9. Lands tonight →
  round13. Set2 has no negative-k1 scenes so gates are fully supervised everywhere.

## 5. The graveyard (measured dead — do NOT re-propose without a NEW mechanism)

| idea | evidence |
|---|---|
| Capacity ↑ (16M gaussians) | 5M→8M = +0.085 dB; 16M OOM/impractical |
| Longer training on videos | bonsai 60k < 30k (−0.44 eval); chair 60k helps only with LPIPS-phase change |
| Pure-L2 / metric-exact loss | worse than L1+SSIM (grad vanishes near target); 3 confirmations |
| Regularization removal | no train-fit change |
| Per-frame exposure/appearance (PPISP) on videos | bonsai −0.13, chair +0.06 eval — dead despite 25% drift |
| Frame interpolation (classical flow) for video holdouts | 6–8 dB BELOW renders at 3–4°; occlusion/perspective kill it; not seam-limited |
| Photo-reuse / IBR paste | 0/290 views (set1, 11.8°); interp result confirms for 3–4° too |
| Per-image pose refinement | +0.14 dB oracle ceiling; BA poses rigidly near-perfect (0.018px median) |
| Rolling shutter correction | cos=+0.089 (random) |
| SH clamp / supersampling / MVS seeding | monotone worse / null / null |
| k2 refit + retrain | breaks pose contract (test poses live in the original BA frame) |
| Test-keypoint per-image warp | +0.05 over ship path; compliance-risky; rejected |
| JPEG/packaging tricks | three independent nulls; quality ladder already optimal |
| Mirror-aware anything for bonsai | untried as such, but the collapse fix + short training is what survived; reflections move view-dependently and defeated both DIS fields and interp |

## 6. Questions for you (ranked by what we most need)

1. **Where is +1 to +4 points hiding, if anywhere?** Given the metric decomposition (LPIPS 40%),
   our per-scene estimates (towers 79.5 avg, bonsai ~70, chair ~68), and 2×16GB/12 days: what
   method class do you nominate, with a mechanism for WHY it beats the graveyard? Examples of the
   kind of thing we can't evaluate ourselves without a hint: post-render perceptual enhancement
   (legal? a pretrained enhancer beyond VGG/Alex is NOT clearly allowed — treat as needing organizer
   approval and rank accordingly), depth-regularized or anti-aliased splatting variants (Mip-Splatting,
   StopThePop, gsplat's antialiased mode), 2DGS/GOF surfels for the glass table, ray-traced
   reflection modeling for bonsai, per-scene SH-degree/appearance variants, distillation from the
   ensemble into one sharper model, test-time per-view optimization that only uses train data
   (e.g., render-space sharpening tuned on held-out train views — legal since no test GT).
2. **Is the top-1 gap structural?** They jumped +8 in one submission (72.5→80.6 equivalent).
   Given the data (videos with blur/reflections; towers dense urban), what single change plausibly
   produces +8 for them, and can we reverse-engineer the class of that change? (Their sub-metrics
   are not visible to us.)
3. **The eval→test 0.3× compression** for recipe nudges: our diagnosis is train_sub(72–89%) vs
   full-train baseline shift. Do you read it differently (eval-set overfit? isolated-hole bias?),
   and does your reading change what experiments are worth running?
4. **The video scenes** (bonsai LPIPS 0.25, chair 0.24 vs towers ~0.10): given mirror glass +
   DoF blur GT, is there a known-good method for view-inconsistent content under THIS metric —
   or is ~70 near the content ceiling and the towers are where the remaining points actually are?
5. **Sanity-check the gates bet** (in flight): on set1 the family ensemble added +0.5–0.9. Any
   reason it should NOT transfer to set2's towers (same camera family, no crop ring)?
6. If your verdict is "gap unreachable": what is the optimal 12-day plan to defend/maximize rank
   (2nd place economics), given ~+0.1/cycle grind reliability?

Answer format: ranked list of moves with (expected pts, GPU-days, risk, kill-test that decides it
in <1 day). Argue mechanisms, not vibes. Assume we will actually run your kill-tests.
