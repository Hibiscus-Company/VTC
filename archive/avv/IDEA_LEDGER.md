# IDEA LEDGER — campaign private_set2 (mined 2026-07-30 02:2x)

Mined from all 4442 lines of EXPERIMENTS.md + run logs + scripts + dirs + memory + BRIEFING
by 10 independent agents (959 raw records), deduped by one synthesis agent.

CORRECTION APPLIED BY HAND: the miners' snapshot predates the r35 grade.
r35 GRADED 77.7106 (+0.0199 over r32) = NEW BEST. The ledger's 'inflight' note telling us
to rebuild bonsai at lam=0 before submitting is WRONG: r33a already shipped lam=0 and LOST -0.0041.

## CLOSED (85)  KILLED=49  SHIPPED=36

| # | idea | cat | status | delta | note |
|---|---|---|---|---|---|
| 1 | bonsai MCMC churn-stop (capD: cap 5M, refine_stop 15k, noise_stop 8k, 30k) | training-recipe | SHIPPED | +3.6206 LB (r10 72.4845 -> r10b 76.1051); implied bonsai scene +25.3 | Fog collapse was MCMC churn on glass, not capacity cap. |
| 2 | Multi-model pixel-mean render ensembling (the 1/N lever) | ensemble | SHIPPED | +1.82 LB at R2 (74.348 -> 76.166); +0.65 first public measurement | Only lever class that never transferred below 1x in 12 rounds. |
| 3 | 3DGUT distorted-space unscented-transform training (--ut) as base method | model-architecture | SHIPPED | +0.90 LB at R5 as one w=0.5 member; +0.67 mean solo on 5 public scenes | Training on virgin distorted pixels removes the undistort resample generation. |
| 4 | Lens displacement field: fit DIS flow on train photos, warp renders | post-process | SHIPPED | +0.7345 LB at R7 (77.6629 -> 78.3974); PSNR +0.556 dB, SSIM +1.31 | Largest post-process win; per-image operator, no retraining, SSIM paid most. |
| 5 | UT recipe scale-up: 30k/5M -> 60k/8M | training-recipe | SHIPPED | +0.5046 LB (round11 76.6097); +0.68/model on public (75.21 -> 75.58 -> 75.90) | Capacity and schedule additive; cap was genuinely binding at 5M. |
| 6 | Drop --ut on pinhole video scenes to unlock gsplat antialiased mode | model-architecture | SHIPPED | +0.486 per video scene on LB (r14b 76.7878); eval +0.54 bonsai / +0.83 chair | k1=0 means UT was pure cost; antialiased mode is forbidden with UT. |
| 7 | Third gate member / mean3 composition | ensemble | SHIPPED | +0.23 LB at R3 (76.166 -> 76.4005) | Early confirmation that composition depth transfers at full strength. |
| 8 | R29 bundle: 7 new members + lens-field gain 1.30 + video lam=0 | ensemble | SHIPPED | +0.1615 LB (77.5029 -> 77.6644), largest late-campaign jump; decomposition closes exactly | Field gain 1.30 corrects the ~30% train-absorption undershoot. |
| 9 | Third tower UT seed (r15) | ensemble | SHIPPED | +0.16563 LB (76.7878 -> 76.9534) = +0.232 per tower, zero extra GPU | Eval 2->3 increment transferred at 1.4x; seeds are non-redundant. |
| 10 | Fourth tower seed + third video seed (r16) | ensemble | SHIPPED | +0.1421 LB (76.9534 -> 77.0955) | Landed inside projection; two axes moved so not individually isolable. |
| 11 | Energy restoration: disagreement-map-driven finest-Laplacian-band boost | post-process | SHIPPED | +0.1005 LB at r28; +0.1883 on production harness at lam=1.0 | Gain grows with ensemble depth: anti-cleanup signature, not proxy inflation. |
| 12 | ADD-not-REPLACE ensemble composition (r20) | ensemble | SHIPPED | +0.0948 LB (77.0955 -> 77.1903); 36x the r17/r18 single-slot swap nulls | Even maximally-correlated EMA twins add; swapping leaves decorrelation on the table. |
| 13 | Field warp resample kernel INTER_CUBIC -> INTER_LANCZOS4 (r27) | post-process | SHIPPED | +0.0743 LB; +0.1095 on production harness, 5/5 towers, one flag, zero GPU | Bicubic destroyed 5% of detail per warped frame. |
| 14 | Mip-Splatting 3D filter baked in DURING training (mip3d), added at w=0.2 | model-architecture | SHIPPED | +0.0426 LB at r25; proxy +0.6596 HCM0181 / +0.4572 HCM0421, all three metrics up | Only late fidelity-signature model gain; ~0.1x transfer to LB. |
| 15 | r12 video recipe fix (chair lpearly + bonsai pC) | training-recipe | SHIPPED | +0.039 LB (76.6097 -> 76.6490) vs eval-predicted +0.134 | Calibrated recipe nudges at ~0.3x eval-to-LB transfer. |
| 16 | Lens field estimator: per-pixel median over the flow stack (r26) | post-process | SHIPPED | +0.0327 LB; LOVO +0.0625/+0.1082, only 31% of harness prediction | Under-delivered because private fields pool 120 views versus harness 60. |
| 17 | Bonsai ensemble 3 -> 6 members (r24) | ensemble | SHIPPED | +0.0210 LB = +0.147 on the bonsai scene; predicted +0.04..0.06 | Video composition marginal is flatter than towers; now equally saturated. |
| 18 | r21 composition floor bank (5/5 tower EMA + chair 7th member) | ensemble | SHIPPED | +0.0163 LB (77.1903 -> 77.2066); all three submetrics up | Completing a proven composition, no new method. |
| 19 | Gaussian field smoothing sigma=1 + chair lens field (r31) | post-process | SHIPPED | +0.0160 LB (77.6644 -> 77.6804); predicted +0.0127 = 126% transfer | Canonical per-image operator; established the ~1x transfer rule. |
| 20 | Boost-matched video energy lambdas (chair 0.40, bonsai 0.25) — r32 | post-process | SHIPPED | +0.0103 LB (77.6907, campaign best); audit shows chair carried all of it | Bonsai half was a measured null; chair worth +0.0063. |
| 21 | Seed-101 as new equal-weight tower member (r22) | ensemble | SHIPPED | LB 77.2318, became the r23-r25 reference baseline; proxy 1->2 add +0.636 | Fresh healthy seed, native render path verified aligned. |
| 22 | Submission cap is 350 MiB (367,001,600 B), not 350e6 | infra | SHIPPED | +21.9 MiB headroom (4.5x); LB-confirmed when the 357,078,727 B r28 zip graded fine | verify_zip had enforced the decimal reading since round one. |
| 23 | chair lpips_from moved earlier at 60k (lpearly) | training-recipe | SHIPPED | +0.74 eval-split (68.536 vs 67.794), all three axes up | LPIPS-phase texture lever transferred from bonsai to chair. |
| 24 | bonsai pC_lpearly (lpips_from 12k, lambda 0.1, 30k) | training-recipe | SHIPPED | +0.20 eval-split (71.3605 vs capD 71.1564); best of four perceptual arms | Longer LPIPS phase at standard weight; 60k and lambda 0.2 both lost. |
| 25 | FastGS ladder: grad_abs 0.00015 + VGG-LPIPS finetune + metric_gate 1 | training-recipe | SHIPPED | g15 +0.136, lpips-ft +0.78, gate1 +0.28 on public; champion 74.7969 -> LB 74.348 | Track A base, superseded entirely by gsplat MCMC and 3DGUT. |
| 26 | EMA of parameters in the frozen-topology window (decay .999 towers / .99 chair) | training-recipe | SHIPPED | eval +0.484 tower / +0.435 chair, but LB single-slot swap only +0.0028 (170x over-read) | Strongest per-model recipe lever; nearly worthless as a correlated swap. |
| 27 | bonsai scale_reg 0.1 members (only axis to pass the paired A/A gate) | training-recipe | SHIPPED | eval ladder 0 -> 70.72 up to 0.1 -> 71.93; gate +0.1653, t=2.12, 17/28; mixture k=8 72.2527 | Mixing new scale_reg members with old beats pure replacement. |
| 28 | Family-weighted ensemble (w_UT 0.5-0.6) — later refuted as pool-dependent | ensemble | SHIPPED | +0.23 local at R5/R6; production re-derivation says uniform is the interior optimum (-0.0435 for any tilt at k=8) | Family tilt only paid across two complementary families; sign flips between pools. |
| 29 | Production harness (public_set, real test poses and GT) as decision surface | eval-method | SHIPPED | r27 predicted every submetric within 15%; harness 3->4 marginal +0.0919 vs LB +0.142 | Replaced the eval-split proxy for all post-processing decisions. |
| 30 | THE TRANSFER RULE: per-image ~1x, pool-dependent ~1/20 | eval-method | SHIPPED | per-image r31 predicted +0.0127 -> +0.0160; pool-dependent r30/r30c predicted +0.100 -> -0.0049 | Discriminator is whether value depends on ensemble over-smoothing. |
| 31 | Isolated-hole eval-split + name-free per-scene recipe router | eval-method | SHIPPED | eval gap 0.121 vs real test 0.116; predicted the +3.62 round10b jump | Eval holes must be isolated, evenly spaced, never adjacent. |
| 32 | Score algebra dScore = -40dLPIPS + 30dSSIM + 0.6dPSNR, psnr_max=50 | eval-method | SHIPPED | reconstructs graded rounds exactly (r25 +0.0426, r32 +0.01034) | LPIPS dominates; PSNR term near-saturated at +0.006 per dB. |
| 33 | Single-variable submissions with byte-verbatim carry of unchanged scenes | infra | SHIPPED | r30/r30c split a failed bundle exactly: -0.0030 members + -0.0019 encode = -0.0049 | Every LB delta attributable to exactly one change. |
| 34 | Ship infra: verify_zip, member_gate, preflight, provenance sidecars, claim pools | infra | SHIPPED | caught wrong-res dirs, fog collapse, silent scene drops, oversize zips, r32 null-op bonsai | Measuring shipped bytes rather than intent reversed a whole round's attribution. |
| 35 | Ensemble numerics: float32 accumulate, single round-half-up, sRGB, PNG sources | ensemble | SHIPPED | uint8 //N costs -0.009, linear-RGB -0.012, median -0.08, JPEG sources -0.03 | Averaging in sRGB code space with one final rounding is optimal. |
| 36 | Encode: JPEG q100 / 4:2:0 / progressive, single-generation from PNG archives | encode | SHIPPED | q99-vs-q100 |effect| <= 0.001 on 290 real GT pairs; ladder optimum, q96 -0.11, q93 -0.60 | Narrow q98-q100 plateau; 4:2:0 beats 4:4:4 on our pool. |
| 37 | Bonsai capacity 5M -> 8M and 60k iterations | training-recipe | KILLED | 8M -0.7101 vs 5M on 28 held-out holes; 60k arms 71.08 / 70.72 vs 71.36 | 8M has MORE live gaussians and still loses; capacity axis closed. |
| 38 | Depth-guided IBR / photo-reuse / frame interpolation (all scenes, both regimes) | post-process | KILLED | -4.4 to -7.4 on towers; chair video probe 41.71 vs render 77.66 (13.7 dB deficit at 3.57 deg) | Translation destroys the paste; 0/290 views where a photo wins. |
| 39 | min_opacity 0.02 (raised MCMC relocation threshold) | training-recipe | KILLED | chair 26.9870 vs 69.37 (PSNR 10.1); tower +0.037 neutral | Chair's low-opacity gaussians are load-bearing; aggressive relocation churn-collapses. |
| 40 | Mip-Splatting 3D filter applied POST-HOC to trained checkpoints | post-process | KILLED | chair 69.8051 -> 29.36 at scale 0.2 (PSNR 8.8); math unit-tested correct | MCMC models are full of sub-pixel spikes; energy compensation blacks out. |
| 41 | init_clip (drop far / mirror-world init points) | data-pose | KILLED | bonsai K2 42.89 vs 71.36; tower init_clip2 -0.12 | Poisoned init was never the bonsai collapse mechanism. |
| 42 | pose_opt (per-view learnable SE3 residual), three variants | data-pose | KILLED | chair 45.96 (v1) / 47.36 (v2 smoothed) / 59.47 (v3 low-lr+warmup) vs 69.37 | Gaussian count exploded 80k->2.97M; low LR helped but still catastrophic. |
| 43 | Difix3D+ pretrained restorer, zero-shot and fine-tuned | post-process | KILLED | tower 82.01 -> 66.86 (s1.0) / 80.60 (s0.25); LPIPS worse at every strength; FT never produced a checkpoint in 6h | Our renders sit far above the LPIPS-0.33 artifact regime Difix trained on. |
| 44 | Bilateral grid appearance model (--bilagrid) | training-recipe | KILLED | -7.6 solo (66.885, PSNR 17.4); -0.7418 as one member at 1/5 weight | Nothing anchors grids to identity; upstream masks it with GT-fitted correction. |
| 45 | Test-time supersampling / SSAA (1.5x-3x, Lanczos and box) | post-process | KILLED | -0.88 / -3.10 / -2.38 / -6.73, monotone with factor | GT has a sharp native-resolution frequency response the model already fits. |
| 46 | Appearance correction family: per-image affine, PPISP, exposure/WB oracle | training-recipe | KILLED | affine -0.40/-0.53/-0.91; PPISP -0.13..-0.68; perfect per-image gain oracle only +0.09..+0.19 dB | Killed from four directions; the whole prize is under 0.1 dB. |
| 47 | FastGS gate family as set2 ensemble members (r13) | ensemble | KILLED | -0.5961 LB = -0.835 per tower; recovered exactly by r14b (+0.595) | Stale 0.4 family weight on 30k gates against 60k/8M UT members. |
| 48 | Monocular depth prior (Depth-Anything-V2 scale-invariant Pearson) | training-recipe | KILLED | solo chair -0.65, bonsai -1.49, tower -0.79; ensemble-add +0.01..+0.08 (noise) | Our multi-view geometry is more accurate than mono depth; prior fights it. |
| 49 | LEGS-inspired Sobel texture-weighted L1 (--texture_weight) | training-recipe | KILLED | chair -0.68 (tw1.0) and -1.51 (tw3.0); bonsai +0.16 / -0.04 | Monotone dose-dependent regression; Sobel on blurry GT upweights blur gradients. |
| 50 | Learned per-view uncertainty weighting (Kendall-Gal Laplace NLL) | training-recipe | KILLED | chair -1.40, bonsai -0.35 | Confidence field drifted below 1 broadly; loss went negative. |
| 51 | Learned restoration head (residual U-Net on held-out render/GT pairs) | post-process | KILLED | same-scene +0.3128 production harness, but cross-scene -0.24/-0.15/-0.17 PNG and -0.39/-0.30/-0.34 after JPEG | Memorises one scene's texture statistics; cross-scene is the only legal route. |
| 52 | Metric-exact loss (0.4 LPIPS + 0.3 DSSIM + 0.02606 ln MSE) and pure-L2 | training-recipe | KILLED | arm B -0.94 (74.96 vs 75.899), arm A -0.27; fidfit pure-L2 arms all <= standard | L1's constant-magnitude gradient beats MSE, which vanishes near target. |
| 53 | LPIPS loss weight raised (0.1 -> 0.3 -> 0.5) and WD-R perceptual loss | training-recipe | KILLED | 75.4293 -> 75.3301 -> 75.2661 monotone down; WD-R nets +0.10 by its own paper table for 2.8x cost | LPIPS term saturated at 0.1; LPIPS and SSIM are near zero-sum. |
| 54 | r33 uint8 deadband fix (float-mean rebuild) on towers + HCM0421 q100 | post-process | KILLED | -0.0149 LB despite +0.0116/+0.0104 harness A/B at lam 0.25/0.50 | Deadband destroyed 43-79% of corrections but fixing delivery lost anyway. |
| 55 | Video energy lambda pushed up (chair 0.60 / bonsai 0.50) — r34 | post-process | KILLED | -0.0052 LB (77.6855); +0.015 dB PSNR bought +0.021pp LPIPS | Brackets r32 from above; sharpening axis exhausted. |
| 56 | Bonsai energy lambda pulled to zero (r33a) | post-process | KILLED | -0.0041 LB (77.6867), despite eval sweep saying lam=0.25 costs -0.424 scene-pts | Brackets r32 from below; the proxy sweep was sign-inverted. |
| 57 | Tower ensemble depth k=8 -> k=10 (r30) | ensemble | KILLED | -0.0030 LB; harness k-curve argmax said k=10 at 4.7 sigma | Reattributed to member QUALITY: added members were an older training generation. |
| 58 | JPEG q98 at 4:4:4 (r30c) | encode | KILLED | harness +0.0504 (t=16.96, 59/60) but LB -0.0019 and cross-scene -0.0094 mean, 0/5 | JPEG artifacts substitute for texture only in a diverse over-smoothed pool. |
| 59 | JPEG keep_rgb / chroma-space encode changes (r23) | encode | KILLED | -0.0259 LB; proxy predicted +1.01 on bonsai; PSNR/SSIM up but LPIPS +0.0662 | GT-free self-distortion was the wrong measurement; encode axis closed. |
| 60 | Ensemble heterogeneity / diversity at fixed k (DIV8 vs HOMO8) | ensemble | KILLED | +0.0204 +- 0.0330, 26/60 wins, t=0.62; PSNR +0.114 dB given back entirely in LPIPS | Cancelled r34/r35 non-UT member training; diversity buys nothing at fixed k. |
| 61 | Alternative combiners: median, trim, huber, tukey, vecmed, patchsel, agreement, inverse-variance | ensemble | KILLED | all <= pixel mean (median -0.0375, vecmed -0.2000, patchsel -0.9534); winsor only +0.008 | Robust aggregators pay only when the pool contains a genuine outlier. |
| 62 | Non-uniform ensemble weights (Gram/LS-optimal, two-stage, family tilt) | ensemble | KILLED | LS weights come out near-uniform; 2-fold CV -0.001 to -0.0430; two-stage -0.0166 | Uniform is the measured interior optimum for a same-generation pool. |
| 63 | Align-then-merge (DIS sub-pixel member alignment before averaging) | ensemble | KILLED | +0.0505 isolated but only +0.0077 stacked on energy restoration; global translation +0.0001 | Substitute for energy restoration; both recover the same level-0 deficit. |
| 64 | Sharpening family: unsharp, PSF deblur, radial MTF, spectral matching, contrast LUT | post-process | KILLED | unsharp -0.14/-0.50; PSF K7 -0.3606 (0/20); MTF -0.05; MSE-optimal global unsharp -0.618 | HF amplitude already optimal in every variance decile; 96% of missing HF is incoherent. |
| 65 | Flat-region band attenuation (mu operator) | post-process | KILLED | monotonically negative: -0.0331 / -0.1040 / -0.2948 / -0.6051; LPIPS itself worse | Falsified the premise that the mean's flat-region HF is mostly noise. |
| 66 | Per-view / per-image lens fields and 3D-lift oracle | post-process | KILLED | spatial hold-out -1.7497 vs global +0.4423; bonsai per-image -2.15 to -4.44 | The oracle fits each view's own flow-noise realisation; no legal predictor reaches it. |
| 67 | Low-frequency and global-colour residual oracles | post-process | KILLED | oracle +0.4553 pooled, realizable LOO -0.0154 (1/5); DC oracle +0.0596, LOVO -0.0010 | Only 5-8% of the LF residual is view-consistent; the rest is unestimable. |
| 68 | Bonsai lens field (every estimator variant) | post-process | KILLED | LOVO n=40 all <= no-field: median_ds8 -0.1270, best variant +0.0012; train-KP geometry +0.0006 | Glass-table reflections move with viewpoint, so no fixed field exists. |
| 69 | Long schedule: 120k iters, warm-start tail extension, noise_stop tuning | training-recipe | KILLED | train +0.52 dB, test +0.00; exp20 vs exp19 exact tie 74.5067 vs 74.5072 | Tail not binding; parameter LRs converge in 10k Adam steps. |
| 70 | cap_max 16M / 12M capacity ceiling | training-recipe | KILLED | OOM at ~11k/60k after 13h; prior 5M->8M rung gave only +0.085 dB | Capacity lever closed on evidence as well as on memory. |
| 71 | 2DGS surfel primitive as a decorrelated member | model-architecture | KILLED | never scored: contiguity backward error, chair 4h50m no ckpt, bonsai CUDA crash, 3.7x slower | DefaultStrategy has no cap_max so surfel count exploded to OOM. |
| 72 | Densification knobs: absgrad, aniso_reg, sky_dome, opacity_reg, eps2d, final_prune, SH clamp | training-recipe | KILLED | all within +-0.15 or negative; absgrad proved a silent no-op under MCMCStrategy; SH clamp monotone harmful | Entire densification and regulariser axis dead in the interpolation regime. |
| 73 | Rolling shutter on drone towers (D13 diagnostic) | data-pose | KILLED | row-linear flow slope 0.324 px over full height; velocity alignment cos +0.089, directions uniformly random | Residual is not velocity-coupled, so RS cannot explain the GEOM term. |
| 74 | BAD-Gaussians / DeblurGS motion-blur forward model | model-architecture | KILLED | free gate: corr(inter-frame motion, render-GT L1) = -0.008, p=0.93 | Chair blur is defocus, not motion; killed before building. |
| 75 | H-B per-frame blur and exposure MATCHING bound | post-process | KILLED | chair +0.094, gain-only +0.013, bonsai -0.006 (eval-split) | Blur is not smooth in frame index; post-hoc matching cannot recover it. |
| 76 | Pose refinement (global, per-image, keypoint translation) and k2/k1 refits | data-pose | KILLED | oracle ceiling +0.233 dB = +0.14 pts; measured -0.02 dB; median per-test-image KP shift 0.018 px | BA poses are rigidly near-perfect; refits desynchronize test_poses.csv. |
| 77 | images.bin test-frame keypoint exploit (compliance gray zone) | data-pose | KILLED | +0.048 increment over the legal DIS ship path, bounded <=0.15 even tuned | Rejected on compliance; legal DIS field already subsumes it. |
| 78 | MVS / COLMAP dense seeding for the GEOM residual | data-pose | KILLED | never run; sparse init already 54k-219k pts = ~2% of an 8M MCMC model; oracle ceiling was +0.74 | Dense seeding pays in timid vanilla 3DGS, not aggressive MCMC relocation. |
| 79 | Literature triage rejected on mechanism: MH-3DGS, DWTGS, RobustNeRF, NeRFLiX, NeRF-W, sparse-view regularizers | model-architecture | KILLED | unmeasured; each closed by abstract fetch or regime argument before GPU spend | Wrong regime: our test poses are dense near-trajectory interpolation. |
| 80 | PSNR ceiling impossibility proof | eval-method | KILLED | RETRACTED: ceiling moved 85.55 -> 85.882 as PSNR went 25.91 -> 26.47 | Structural argument expired at R7; the 4-7 dB practical gap remains. |
| 81 | Top-1 submetric forensics | eval-method | KILLED | other teams' submetrics not visible; +9.2 discrete jump unresolved | Model-class gap: they beat our TRAIN fit on held-out views. |
| 82 | Screen-tier proxy (1/4 iterations) for recipe decisions | eval-method | KILLED | produced an outright ranking SIGN FLIP on tower EMA decay vs full-length | 4x throughput but rankings untrustworthy; late-failure modes invisible. |
| 83 | Single-slot member swaps as LB experiments (r17, r18) | eval-method | KILLED | +0.0024 and +0.0028 = unmeasurable; CRC-verified as real byte changes | Bundle confirmed changes into one zip; fragments waste submission rounds. |
| 84 | Per-test-pose local finetune (PPFT) | training-recipe | KILLED | +0.09 inside a +-0.2-0.3 noise band at n=6; full version 64h for 480 images | Metrics disagree internally and it is compute-infeasible. |
| 85 | Byte-allocation optimisation: knapsack lambda, custom quant tables, chroma precomp, PNG | encode | KILLED | knapsack retracted by the MiB correction; all quant tables -0.009 to -0.094; PNG +0.05 but 2.6x the cap | Bytes stopped binding after the MiB correction; allocation buys nothing. |

## IN FLIGHT (4)

### r35 zip: bonsai rebuilt as 7 existing + 3 scale_reg=0.1 members
- state: built 334.36 MiB, VERIFY PASSED, NOT SUBMITTED; scale_reg gate predicted ~+0.022 LB before transfer discount
- Blocker: it still ships bonsai energy lam=0.25, which its own 28-frame held-out sweep measured at -0.424 scene-pts and which r33a/r32 bracket as worth only +0.0041. Rebuild bonsai at lam=0 before submitting, or the composition gain and the operator loss cancel.

### bonsai 30k/8M single-variable retrain
- state: relaunched clean after a four-variable version was caught pre-GPU; a later record grades 8M at -0.7101 vs 5M on the 28 holes
- Treat as effectively closed: the capacity hypothesis lost on its own eval holes and 8M has MORE live gaussians than 5M, so there is no unexplained upside left.

### UBS-6D (6D spatial+angular Beta gaussians) in an isolated env
- state: reached training start on bonsai cap5M after two failed env builds; no score ever printed; kill bar was >= +1.0 over 71.799
- A different-primitive moonshot on the one scene with an in-sample fitting floor; the +1.0 bar is unreachable and the heterogeneity gate says a decorrelated member is worth ~0 anyway.

### 120k-iteration exploration rung
- state: curve file stops at iteration 5000; the closure line cites train +0.52 / test +0.00 which belongs to exp31b's warm-start tail, not a 120k run
- Either finish it or mark it dead honestly; right now the log claims a result it never measured.

## UNTRIED (25)

### WORTH_RUNNING (3)

**Refit the lens field on ENSEMBLE-MEAN train renders instead of one member's**  _[post-process]_
- gain: +0.03..+0.05 LB. Basis: self-estimate +0.05-0.1 dB per scene; per-image operator so ~1x transfer; 6 fielded scenes x (0.6*0.075 dB + SSIM share)/7. Below the 0.407 A/B noise floor, so validate on >=3 public towers on the production harness, not by a single LB round.
- cost: 2-4 GPU-h (train renders for k members x 6 scenes at ~5 min/scene/model) + 1h CPU refit
- why untried: Deferred to R8 in July, never revisited across five refits

**Fix the energy_restore --k mis-specification on HCM0421 (k=8 passed, true member count 9)**  _[post-process]_
- gain: +0.000..+0.010 LB. Basis: r-map mis-scaled 2-4% on 1 of 7 scenes; the lambda curve near optimum is flat (harness lam 0.75 +0.1756 vs 1.0 +0.1883), so the correction is small but its sign is known non-negative on towers.
- cost: ~20 min CPU (rebuild one scene's mean, re-apply, re-encode)
- why untried: Logged as an open defect in the briefing, never rebuilt

**GT-free harmful-member removal via LOO deviation predictor**  _[ensemble]_
- gain: +0.000..+0.020 LB. Basis: removing the worst member is +0.0372 on the public pool with corr(LOO_delta, deviation) = -0.901; but member value is pool-dependent so apply the /20 haircut, giving ~+0.002 expected, with upside if a genuinely stale-generation member is found (r30 showed one bad member costs -0.003 to -0.05).
- cost: ~1h CPU, zero GPU (renders already on disk)
- why untried: Identified as the one lead with positive evidence, then the campaign froze

### MARGINAL (7)

**Probe the flat-region / sky half of the LPIPS budget with operators other than attenuation**  _[post-process]_
- gain: +0.00..+0.05 LB at ~20% odds. Basis: 34.7% of pixels far from edges carry 39% of LPIPS and sky has the highest LPIPS density (1.377), and LPIPS carries the -40 weight; but every operator that touched flat regions (attenuation -0.03..-0.61, geoflat arms all <= baseline) has lost, so the prior is bad and only the size of the untouched budget justifies a cheap look.
- cost: 4-8h CPU on cached harness state (pyramids hoisted, ~6-20s/arm)
- why untried: Only one operator class tried; it falsified its own premise and the axis was dropped

**Motion-blur K-render row-weighted composite probe**  _[post-process]_
- gain: +0.00..+0.05 LB at ~15% odds. Basis: the gate that killed the class measured ~15 px motion in both best- and worst-fit frames, which is exactly the signature of a uniformly-present blur it cannot see; but the independent blur surrogate says blur explains only 23.3% of bonsai LPIPS and every sharpening/deblur operator has lost.
- cost: ~2h, existing checkpoints, no retrain
- why untried: Self-flagged that the Farneback gate tested variance and is blind to uniform blur

**mip3d member ensemble-weight sweep (w_mip currently 0.2)**  _[ensemble]_
- gain: +0.00..+0.02 LB. Basis: the r25 postmortem blamed the w=0.2 dilution for delivering +0.0426 of a predicted +0.10-0.25; but weighting is pool-dependent (flips sign between pools A and B) so haircut hard, and the production re-derivation says uniform is the interior optimum.
- cost: <1h CPU, renders already on disk
- why untried: Contradictory records: one says swept on HCM0421, one says shipped by assumption

**DENSER shared-trunk ensemble (branch members from one checkpoint)**  _[ensemble]_
- gain: +0.00..+0.02 LB. Basis: composition is the only never-negative lever class, and branching yields more members per GPU-hour than independent seeds; but branched members are maximally correlated and our depth curve is saturated (8->10 lost -0.0030, r21 marginal +0.0163), so the extra members land on the flat part.
- cost: ~8-12 GPU-h plus integration
- why untried: Surfaced in the final lit scan two days before freeze

**Deblur-consistent TRAINING (in-training blur forward model, not post-hoc)**  _[training-recipe]_
- gain: +0.00..+0.10 LB at ~25% odds (advisor said +0.1-0.3 blended at 30%). Basis: chair has a 7.6x within-scene sharpness spread so the post-hoc fit received inconsistent supervision; but the motion gate, the blur surrogate (23% of LPIPS), and every eps2d/sharpening arm all point the other way, and any win is a recipe change discounted ~0.3x.
- cost: ~20-30 GPU-h (implementation + 2 video scenes x seeds + eval)
- why untried: Advisor argued H-B's flat post-hoc bound did not kill it; never scoped

**Post-freeze code hygiene: PNG-over-JPEG source ordering, apply_field re-apply stamp, field meta dims**  _[infra]_
- gain: 0 expected, but removes a double-warp / wrong-source-dir tail risk that has already nearly regressed bonsai once (mean|px| 1.69 from a stale dir).
- cost: ~1h CPU
- why untried: Deliberately deferred as too risky to touch mid-flight

**q98 encode as free byte insurance**  _[encode]_
- gain: 0.000 score, frees ~80 MiB. Basis: q98 measured +0.0020 +- 0.0020 (t=1.0) on the production harness at 25% fewer bytes. Only worth invoking if a future operator inflates file size.
- cost: ~15 min CPU
- why untried: Banked as insurance, never needed once the MiB cap freed 21.9 MiB

### DEAD_ON_ARRIVAL (15)

**Rolling-shutter forward model on the two video scenes**  _[model-architecture]_
- gain: negative expected. Basis: RS requires with_ut=classic, which forfeits the measured +0.486/scene antialiased win = -0.139 blended before RS delivers anything; the only RS diagnostic we own (D13) found the residual is not velocity-coupled at all.
- cost: ~10 GPU-h
- why untried: Confirmed available in gsplat 1.5.3 three times, never run

**Additional seeds / 5th-6th tower member, seed-202 productionization**  _[ensemble]_
- gain: +0.00..+0.02 for ~25 GPU-h. Basis: k=8 -> k=10 already lost -0.0030 on the LB and the marginal add is measured at +0.016 and falling; only justified if the added members are current-generation and pass cand_rank on deviation.
- cost: ~5 GPU-h/member x 5 towers = 25 GPU-h
- why untried: Depth saturated before the seeds were promoted

**Per-scene hyperparameter optimisation (idea 9)**  _[training-recipe]_
- gain: +0.00..+0.03 for ~40 GPU-h. Basis: the +0.2-0.4 estimate is set1-era and pre-dates the finding that recipe nudges transfer at 0.3x and that every individual recipe knob (cap, noise, opacity, aniso, absgrad, eps2d, loss) is dead on its own eval holes.
- cost: ~40 GPU-h
- why untried: Ranked 4th repeatedly, never had 40 spare GPU-hours

**Specular deferred rendering for bonsai's glass table (3DGS-DR / GaussianShader / PR-ENDO)**  _[model-architecture]_
- gain: ~0. Basis: bonsai is the only scene with nominal headroom (69.96 vs towers 78.74) but 84.4% of its held-out LPIPS is an in-sample fitting floor it cannot fit even on views it fully supervises, so no rendering-model change reaches the deficit; and it is a 1/7-weight scene, so even +2 scene-pts is +0.29 blended for days of work.
- cost: 3-5 days build + 10 GPU-h
- why untried: Architecturally heavy for one scene worth 1/7 of the score

**Perceptual-GS capacity allocation ported to MCMC relocation weights**  _[training-recipe]_
- gain: ~0 vs a +0.1-0.3 lit-scan guess with no surface. Basis: every capacity and allocation experiment we ran (cap 0.5M-16M, min_opacity, init_clip, sky_dome, importance_vis_norm, gaussian-vs-texture frequency matching showing gaussians already 20x finer than the texture) says allocation is not the bottleneck.
- cost: ~15 GPU-h plus integration
- why untried: Lit-scan candidate #3, surfaced at freeze

**Other pretrained NVS restorers (GSFix3D / GSFixer / GFix / BAGS)**  _[post-process]_
- gain: negative. Basis: Difix zero-shot was destructive at every strength (-1.4 to -15.1) because our renders sit at LPIPS 0.06 versus the 0.33-class degradation these models train on; our own learned restorer confirmed the same distribution-shift failure cross-scene.
- cost: 1-2 days integration each
- why untried: Only Difix was ever fetched; its failure closed the class

**RIFE / FILM learned frame interpolation on video scenes**  _[post-process]_
- gain: negative. Basis: the classical DIS midpoint probe scored 50.62 (chair) and 59.54 (bonsai) versus renders at 77 and 71 — a 6-8 dB deficit that is a registration/parallax problem, not a seam problem a better interpolator fixes.
- cost: 1-2 days + approval
- why untried: Gated on Rule-10 approval; the classical proxy killed the case first

**Gate-family re-sweep at tiny weight (w_gates 0.1 / 0.2)**  _[ensemble]_
- gain: ~0 at best. Basis: gates cost -0.835/tower at w=0.4; scaling to 0.1 asymptotes to zero, not to positive, and the heterogeneity gate says a foreign family at fixed k is worth +0.02 +- 0.03 (t=0.62).
- cost: ~2h CPU
- why untried: Deadline economics said no after r13's regression

**Fit the lens field on HELD-OUT train views (train on 90%, fit on the unseen 10%)**  _[post-process]_
- gain: ~+0.005. Basis: it targets the last ~5% of the oracle the train fit misses, and the absorption bias it removes has ALREADY been captured by the 1.30 gain scalar (+0.1615 LB), so it is now double-counting.
- cost: ~30 GPU-h (retrain at 90% data)
- why untried: Would need 90%-data retrains of every model

**Untouched method families: Scaffold-GS / Octree-GS / LightGaussian / EAGLES; GOF / PGSR / StopThePop / SpecGaussian**  _[model-architecture]_
- gain: ~0. Basis: the compression/anchor family optimises MODEL size, but our binding constraint was the 350 MiB submission-IMAGE budget, so there is no lever; the surface/sorting family would only enter as a decorrelated member, and the heterogeneity gate prices diversity at fixed k at noise while 2DGS (our only surface attempt) never produced a checkpoint.
- cost: 3-7 days each
- why untried: Zero mentions in 4442 log lines — never even triaged

**Wave-2 densification variants: EFA-GS, Pixel-GS, glossy mask, planarity, foliage consistency**  _[training-recipe]_
- gain: ~0. Basis: same closed densification axis as Perceptual-GS; Pixel-GS was already on the strategy audit's kill-list without a test, and the screen tier that would have screened them produced a ranking sign-flip and is itself retired.
- cost: ~4 GPU-h each
- why untried: Queued to a screen-tier pool that was overtaken by the EMA pivot

**Cheap loss/aug leftovers: larger SSIM window, Charbonnier/Huber, random background, SH 3->4, training-time supersampling**  _[training-recipe]_
- gain: ~0. Basis: the loss avenue was closed end-to-end (metric loss, texture weight, uncertainty, pure-L2, LPIPS weight all <= baseline) and attributed to the ~27 dB fitting wall, so reshaping the loss cannot help; SH is load-bearing (clamping is monotone harmful) and training supersampling is VRAM-infeasible. Each also needs a full 7-scene retrain to ship.
- cost: ~6 GPU-h each to test, ~50 GPU-h to ship
- why untried: Each dismissed on argument, then folded into 'loss avenue closed'

**MIP_PLAN (A): 2D opacity compensation in the FastGS rasterizer**  _[model-architecture]_
- gain: ~0. Basis: the FastGS family is permanently shelved (gates cost -0.835/tower on set2), and the equivalent 2D mip already ships free via gsplat's antialiased mode (+0.486/scene).
- cost: ~90-140 LoC CUDA + 6 GPU-h
- why untried: Spec corrected and ready, then FastGS was abandoned for gsplat

**IBR leftovers: DIS-flow-corrected paste arm, IBR directly in distorted space**  _[post-process]_
- gain: negative. Basis: IBR loses monotonically with photo fraction (-4.4 at the tightest gate) and D12 found 0/290 views where a train photo beats our render; halving the resamples cannot close a 13-15 dB deficit.
- cost: ~3h
- why untried: Implemented but deprioritized when IBR closed

**Full-resolution originals (images.bin keypoints at 5280x3956)**  _[data-pose]_
- gain: n/a — out of bounds regardless of magnitude.
- cost: n/a
- why untried: User hard line, explicitly not to be crossed

## NOTES — contradictions, fabricated-result risks, gaps (14)

1. STATUS CONFLICT (resolved by surface): video energy lambda. The chair and bonsai eval-split sweeps on real held-out frames say lam>0 is monotonically harmful (bonsai lam=0.25 = -0.424 scene-pts; chair lam=0.50 = -0.188). The LEADERBOARD says the opposite in both directions: r32 (chair 0.40 + bonsai 0.25) = +0.0103 and is the incumbent best, r33a (bonsai back to 0) = -0.0041, r34 (both raised) = -0.0052. The proxy pools are built from DIFFERENT models than the production members, so a pool-dependent operator inverts. LB wins; the eval sweeps must not be cited as kills.

2. STATUS CONFLICT: mip3d ensemble weight. One record has the sweep SHIPPED with w_mip=0.2 chosen on the HCM0421 proxy; another has it PROPOSED_UNTRIED with r25 shipping 0.2 by assumption and the postmortem blaming 'the w=0.2 dilution'. Unresolved — listed as untried/MARGINAL because the postmortem language implies no sweep result was ever read.

3. FABRICATED-RESULT RISK: the 120k-iteration arm. Its curve file stops at iteration 5000, yet the closure line cites 'train +0.52 dB, test +0.00' — those are exp31b's warm-start tail-extension numbers, not a 120k run. The long-schedule axis may be closed on a borrowed number.

4. FABRICATED-RESULT RISK: Charbonnier/Huber loss is listed twice as 'tested and dead' in summary prose, but no arm, score, or table for it exists anywhere in the record. It was closed by first-principles argument only.

5. SELF-OVERTURNED: ensemble diversity. The fixed-k ladder says a foreign-family add beats a same-family add (+0.31 vs +0.16); the k=6->8 refutation says the opposite (add_same_UT +0.1213 beat every foreign add); the heterogeneity gate at fixed k=8 reads +0.0204 +- 0.0330, t=0.62, 26/60 wins. Family value is a function of base pool depth and is not a stable lever. Later self-correction attributes the r30 loss to member QUALITY, not depth or diversity.

6. SELF-OVERTURNED (twice): encode. 'q100/ss2 beats lossless PNG' was measured FALSE on the production harness (PNG +0.0544, but infeasible at ~690 MiB of towers). 'q100/ss0 = +0.0094' was re-measured at -0.0076. Both original figures came from retired single-member proxy chains. Any encode claim older than the production harness should be treated as void.

7. SELF-OVERTURNED: the D5 registration cave. First read as +2.13 dB and called dead; re-scored as +2.89 SCORE points because the gain is SSIM-heavy; then re-scoped again as bounding REGISTRATION of this render, not model quality — dense flow only moves existing pixels and cannot sharpen a soft cable. Three different readings of one number.

8. SELF-OVERTURNED: 'LPIPS and SSIM are zero-sum'. True for judge-directed moves (sharpen, blur, WD-R, restoration head) and false for fidelity moves (lens field, mip3d both pay on all three metrics at once). The zero-sum framing was actively pushing the campaign toward LPIPS-only tricks.

9. NOISE-FLOOR INCONSISTENCY: the stated 0.407-point 1-vs-1 A/B noise floor is a TRAINING-RUN floor (seed spread, matched configs). The leaderboard itself is deterministic and has resolved differences of 0.0019-0.0030. Do not use 0.407 to dismiss a graded LB delta, and do not use LB precision to justify shipping a harness result below 0.407 without cross-scene validation on >=3 towers.

10. GAP: no record of whether r35 was ever submitted, and no grade for it. The log ends with r32 as the graded best (77.6907) bracketed by three losses (r33 -0.0149, r33a -0.0041, r34 -0.0052). Treat 77.6907 as a local optimum in four measured directions.

11. GAP: eval_score.py drifted — byte-identical input scores 0.95 lower after 17/07. Every eval-split number recorded before that date is incomparable to anything after it, and several early FastGS/Track-B deltas in this ledger straddle the boundary.

12. GAP: the public harness pool contains a byte-identical duplicate member; de-duplicating is worth +0.0171 on the harness. Every published k-curve number (including the k=10 argmax that cost r30) is contaminated by it and was never recomputed.

13. GAP: bonsai. It is the only scene with nominal headroom (~70 vs towers ~78.7, chair ~80.2), and the campaign spent heavily on it, but 84.4% of its held-out LPIPS is an in-sample fitting floor — it cannot fit views it fully supervises. Capacity (8M -0.71), iterations (60k -0.28), init clipping, depth priors, lens fields, and energy restoration all failed there. No remaining lever in this ledger addresses a fitting floor.

14. GAP: several artefact directories were never explained — data_supervision_probes (18 scripts, no docstrings, no result logs), cvclassic (burst/dof/neigh/fuse/chroma scripts with cached fields but no per-script numbers), and an unrun ssaa_sweep. Any conclusion attributed to those probes is unsupported by an on-disk result.
