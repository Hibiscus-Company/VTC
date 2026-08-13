---
name: psnr-ceiling-retracted
description: "The \"PSNR ceiling proof\" (that 85.94 is structurally unreachable) EXPIRED at R7 — stop citing it; the 4-7 dB gap is real but is no longer a proof"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
---

**Do not cite the "PSNR ceiling proof" any more. It expired on 2026-07-14 at R7.**

The argument was: at our PSNR, even *perfect* LPIPS=0 and SSIM=1.0 cannot reach top-1's
85.94, so PSNR gains are mandatory. It was true and load-bearing for weeks:
- At R6's PSNR 25.915 the ceiling was **85.55** — a genuine 0.39-pt structural lockout.
- At R7's PSNR 26.470 the ceiling is **85.882** — only **0.096 dB** from being permissive.

One more +0.1 dB and the bound says nothing at all. **The hard-impossibility claim is dead.**

**But the retraction is NOT good news, and must not be reported as one.** What died is the
clean *proof*, not the gap. To actually reach 85.94:

| at these perceptual metrics | PSNR needed | we have |
|---|---|---|
| our LPIPS .098 / SSIM .882 | **39.0 dB** | 26.47 |
| excellent .05 / .93 | **33.4 dB** | 26.47 |
| near-perfect .03 / .96 | **30.6 dB** | 26.47 |

Top-1 still implies ~31-34 dB and we remain 4-7 dB short. The gap is empirical now, not
provable. See [[lens-field-correction]] for what closed the last 0.55 dB, and
[[3dgut-breakthrough]].

**Top-1 forensics is CLOSED**: the user confirmed other teams' submetrics are NOT visible on
the leaderboard, so we cannot read top-1's (PSNR, SSIM, LPIPS) triple and cannot settle how
they got there. Unresolved and on the record: top-1 jumped **76.75 -> 85.94 in one discrete
step** (+9.2, vs our largest-ever controlled gain of +1.82), and `images.bin` stores keypoints
at **5280x3956** against /4 cameras — the organizers COLMAPped at FULL resolution, so full-res
originals of these exact scenes exist. Don't re-litigate this without new information.
