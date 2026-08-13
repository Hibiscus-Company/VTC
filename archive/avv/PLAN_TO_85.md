# PATH TO 85+ — built 23/07 03:xx, freeze 28/07 (5 days left)

User request: schedule a detailed, step-by-step plan toward score 85+, explicitly acknowledging
it may be impossible — "what point of not giving it a try." This is that plan: an honest reality
check first (grounded in OUR OWN measured evidence, not guesswork), then the maximal, GPU-
saturating attempt within the time left.

## PART 0 — THE MATH, AGAINST WHAT WE'VE ACTUALLY MEASURED

Score S = 100·[0.4(1−LPIPS_vgg) + 0.3·SSIM + 0.3·PSNR/50]. Each term caps out fast:
- LPIPS→0.03 (excellent): contributes 0.388 of the 0.4 max.
- SSIM→0.96 (excellent): contributes 0.288 of the 0.3 max.
- PSNR needs **29.0 dB** just to make the arithmetic reach 0.85 total, *given* the other two are
  simultaneously near-best-possible. At 85.94 (set-1's top-1) the requirement was 30.57 dB.

**Our best-ever simultaneous numbers, one round, set-1 R7:** LPIPS .0983, SSIM .8816, PSNR
26.47 dB → score 78.397 (verified, this is a real submitted result, not a projection).
Plug those personal-bests into the formula and you get exactly what we got: **there is no
hidden slack** — the formula reproduces our result to 5 decimals every time we've checked.

**To reach 85 we do not need one more trick. We need to simultaneously beat our own all-time
best LPIPS, all-time best SSIM, AND all-time best PSNR — each by a wide margin — in the SAME
round, on scenes that are on average harder than set-1's** (2 of set-2's 7 scenes are blur-
limited handheld video, not drone stills).

## PART 0.5 — WHAT OUR OWN FORENSICS ALREADY PROVED (set-1, exhaustive, don't re-derive)

This is not speculation — it's the result of ~2 weeks of dedicated audits (EXPERIMENTS.md,
audit rounds 7-13). Load-bearing findings, each independently confirmed:

1. **Appearance/exposure gap: dead.** Oracle per-test-image color correction (the max ANY
   appearance method could ever buy) = +0.09 dB. Not the lever.
2. **Photo-reuse / IBR: dead.** 11.8° mean parallax between nearest train and test poses;
   0/290 nearest-train-photo substitutions beat our own render. Too much viewpoint change.
3. **Rolling shutter (drone/tower scenes): dead, measured.** Residual flow uncorrelated with
   drone velocity (cos≈0.09, essentially random) — no RS signature. **Untested for chair/
   bonsai** (handheld phone video, different capture physics) — see Part 2, item 5.
4. **Loss function: eliminated three independent ways.** Pure-L2, metric-exact log-MSE loss
   (`--metric_loss`, literally derived from this competition's own score formula), and reg-off
   all capped at or below the standard 0.8·L1+0.2·SSIM loss's ~27 dB train fit. The loss was
   never the ceiling.
5. **Regularization: eliminated.** Reg-off arm scored *worse*, not better.
6. **Raw capacity: closed, practically.** 5M→8M gaussians bought +0.085 dB (nothing). 16M
   capacity test OOM'd after 13h — even if it had worked, the scaling curve was already flat.
   Brute-force "more gaussians" is a dead end, not because it's wrong in theory but because it's
   too expensive for what it returns.
7. **Content/data ceiling — CONFIRMED for set-2's two video scenes, three independent kills:**
   eps2d, B3 detail-injection, and texture_weight all trace to the same root cause: chair/bonsai
   ground truth is **itself blur-limited** (phone DoF + motion blur). Sharpening the model past
   the GT's own blur level makes things *worse* — there's no recoverable detail past what the
   camera captured. This is a **data ceiling, not a method gap**, and it's specific to the 2
   video scenes (towers are sharp, VoL~4000; bonsai/chair p5 VoL 126-331, visibly soft).
8. **One loophole found, correctly NOT used:** `images.bin` contains real SIFT keypoint
   measurements *for the withheld test frames themselves* (their poses are baked into the
   sparse reconstruction). Quantified value if exploited: real, non-trivial (+points). **Never
   acted on** — this is measurement derived from test-GT pixels, which is exactly what Rule 10
   forbids ("no inferring/looking at test GT"). Stays off the table for the same reason here.
9. **Bottom line the set-1 audit reached, independently, twice:** top-1's 85.94 requires a
   genuinely different method class, not a better-tuned 3DGS. "Consolidate, don't moonshot"
   was the adopted verdict there. Set-2's current #1 (82.17) is *also* below 85 — nobody
   publicly ahead of us on THIS dataset has broken 85 either, with whatever method they're using.

**Honest verdict up front: 85 is very likely not reachable through per-scene 3DGS refinement in
5 days. The realistic, evidence-backed ceiling is ~80-83.** That said — every item above was
tested with the tools we had at the time; a few genuinely new mechanisms (not re-runs of dead
ideas under a new name) haven't been tried. Those are the actual "why not" shots, Part 2.

## PART 1 — THE FLOOR: bank what's already proven (main line, HIGH confidence, do this regardless)

This alone is projected to land ~79-82, i.e. competitive with or ahead of current #1 (82.17).
Zero new research risk — everything here already won at eval-split scale.

| step | action | scenes | status | ETA |
|---|---|---|---|---|
| F1 | chair EMA (decay .99) → production, swap into r16's 3-seed ensemble | chair | **running now**, GPU1 | ~04:15 today |
| F2 | tower EMA validate on a REAL set-2 tower — **sweep BOTH decay .99 AND .999** (proxy result flipped: tower prefers .999 +0.484 vs chair's .99 +0.435 — decay looks scene-dependent, don't assume .99) | 1 tower | queue next after v1 pool drained (done 02:30) | +2h × 2 arms, screen-tier |
| F3 | EMA-ify remaining chair seeds (aa7, aa13) if F1 zip confirms on LB | chair | conditional on F1 | +4h (2 seeds parallel) |
| F4 | EMA rollout to all 5 towers (production, full 60k/8M, all seeds) | 5 towers | conditional on F2 | ~2h/seed × N seeds, 2 GPUs |
| F5 | Mip-Splatting 3D smoothing filter — build once, apply everywhere (never regressed in prior recommendation, unbuilt) | all 7 | build 2-3h | build tonight/tomorrow, validate ~1 GPU-h/scene |
| F6 | Remaining phase-2 unbuilt items with a live mechanism: EFA-GS (chair thin legs), glossy-downweight + planarity reg (bonsai table), sky-dome (only neutral so far, re-test post-EMA) | chair, bonsai | ~2h impl each | day 2-3 |
| F7 | 5th decorrelated tower seed (proven 1/N ensemble gain, cheap, banked) | towers | 0 impl | ~2h/tower |
| F8 | Re-sweep family ensemble weights after F1-F7 land (free, CPU-only) | all | 0 impl | 10 min |

**Every completed step here = one zip, per the standing directive.** This is not "wait 5 days
then submit once" — bank each real gain as it lands.

## PART 2 — THE "WHY NOT" SHOTS: new mechanisms only, nothing re-tested under a new name

Each targets a SPECIFIC surviving gap from Part 0.5, not a vague hope. Bounded time-box; if the
diagnostic/screen-tier result doesn't show signal, it's killed same-day, not carried forward.

| # | idea | targets which finding | why it's actually new (not a repeat) | time-box | odds |
|---|---|---|---|---|---|
| 1 | **Pretrained monocular depth/normal prior as an init/geometry loss** (e.g. a frozen depth network's relative-depth ordering, patch-wise, weight ~0.01-0.05) | capacity-*efficiency*, not capacity-*quantity* (#6 above tested MORE gaussians; this tests BETTER-PLACED gaussians with the same budget) — sparse-SfM regions specifically: bonsai's glossy table has only 54k SfM pts | Rule 10 explicitly allows pretrained models as loss/metric; this is the same class as the already-permitted VGG/AlexNet LPIPS, just a depth net instead | build 3-4h, screen-tier test 1 scene (bonsai, weakest SfM) ~1h | medium — most likely candidate on this whole list, best mechanism-to-precedent match |
| 2 | **Temporal-blur-kernel modeling for chair/bonsai** (learn a small per-frame blur kernel applied to the RENDER before the photometric loss, so the underlying 3D stays sharp; at test render time, interpolate the kernel from the two nearest train-neighbor frames — legal, uses frame index only, same precedent as the already-validated appearance-interpolation finding) | finding #7 (video content is blur-limited) directly — this is the first idea in 3 tries that changes the MECHANISM instead of reweighting an already-blur-matched loss | build ~1 day (forward blur op + kernel param + neighbor interpolation at render) | medium-low — real mechanism, but risk that current soft renders already implicitly match GT blur (per the eps2d/texture_weight postmortem), in which case there's no daylight to gain |
| 3 | Rolling-shutter compensation, **chair only** (portrait handheld phone, visible motion blur — NOT the drone case that was already measured and killed) | finding #3, explicitly flagged untested for this capture type | native gsplat support confirmed; diagnostic first (check row-linear flow vs frame-to-frame camera velocity, same method as the drone test, ~1-2h), only implement if diagnostic shows a real (non-random) signal | diagnostic 1-2h, implement only if positive, +3-4h | low — same team already found this exact signature to be a null result once (different scene) |
| 4 | Second, independent Mip-Splatting-style filter validation on chair (thin structures: legs, tissue box edges) as its OWN family branch, not just a global swap | orthogonal to blur ceiling — this is a rendering-correctness fix (aliasing), not a content-recovery attempt, so it survives the blur ceiling finding | reuses F5's build | ~1 GPU-h | medium (this one already had 2 independent literature recommendations) |
| 5 | Ensemble scale-up beyond 3 seeds per scene where GPU-hours allow (variance reduction has a real, measured, non-zero return every time it's been tried — set-1 seed-stacking was the one lever with a positive track record throughout) | not a new mechanism — pure "we have GPU-hours left, spend them where the return is proven, not hopeful" | 0 impl, pure compute | ~2h/extra seed | high confidence, but small (diminishing per set-1's own note: r15→r16 tower step was only +0.14 combined) |

None of these are expected to single-handedly close a 5+ point gap. Combined, optimistic-case
(everything in Part 2 lands at the top of its range, nothing interacts negatively): maybe another
+1.5-3 on top of Part 1's ~80-83 → **83-85 ceiling in the best case we can currently construct.**
That is the honest maximum, not a promise.

## PART 3 — DAY-BY-DAY (2 GPUs, both saturated per standing directive)

- **Today 23/07 (already in flight):** F1 chair-EMA production (ETA ~04:15) → build+notify zip.
  v1 pool's last job (tower EMA proxy) drains → screen-tier pool takes GPU0. Once GPU1 frees
  after F1's zip, start F2 (real tower EMA validation) immediately. Begin building item #1
  (depth-prior) code in parallel — CPU/my-time work, doesn't block GPU.
- **24/07:** F2 result decides F4's go/no-go. Start F5 (Mip-Splatting) build if not done; validate
  on 1 scene per family. Launch item #1 (depth-prior) screen-tier test on bonsai the moment it
  compiles. Zip + notify on every scene-family confirmation.
- **25/07:** F3/F4 production rollout (EMA to remaining chair seeds + all towers) — this is the
  GPU-hour-heavy day, ~2h × (2 chair seeds + 5 towers × N seeds) across 2 GPUs. Item #1 verdict
  lands; if positive, queue production. Item #3 diagnostic (chair RS) runs as a filler job.
- **26/07:** F6 (EFA-GS, bonsai glossy/planarity), F7 (5th tower seed), F8 (re-weight). Item #2
  (blur-kernel) build+test if item #1 or #3 didn't already consume the day — this is the most
  speculative/expensive item, lowest priority if time is short.
- **27/07:** Final ensemble assembly across every confirmed family branch, full zip verification
  (`scripts/verify_zip.py`), submit the best composition. Freeze buffer.
- **28/07:** Freeze. No new training. Final argmax check only if a same-day result is still
  pending verification from the 27th.

## STANDING RULES THAT APPLY THROUGHOUT (unchanged)
- Zip + notify on every confirmed round. Never auto-submit — user submits, reports grades.
- Family-ensemble decision rule: keep as a branch if within ~0.15 of baseline AND
  method-decorrelated; keep as outright replacement only if it beats baseline structurally.
- Both GPUs saturated at all times (claim-based job pool, sentinel-file gating, no pgrep waits).
- Rule 10 is absolute: the test-keypoint loophole (finding #8) and anything like it stays closed,
  regardless of value, for the rest of this campaign.
