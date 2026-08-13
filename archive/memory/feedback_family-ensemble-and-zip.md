---
name: feedback-family-ensemble-and-zip
description: Winning candidates become ensemble FAMILY BRANCHES (not recipe replacements); zip + notify on every potential submission round
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-07-23T03:47:07.849Z
---

Two standing directives from the user (22/07):

**1. Candidates = family branches, not replacements.** When an idea proves worthwhile, spin it
into its own ensemble family branch, develop it, then pixel-mean-average it into the final
ensemble — exactly like set1's FastGS-family + UT-family combination. This LOWERS the keep-bar:
an idea does NOT have to beat the baseline; if it lands within ~0.15 of baseline AND is
method-decorrelated (genuinely different mechanism, not a seed rerun) it can ADD as an ensemble
member even at parity (set1 precedent: B4warm was -0.33 as a single model but +0.05 as a member).
**Eval protocol:** after the single-model eval-split score, if a candidate is within ~0.3 of
baseline, ALSO compute the 2-member pixel-mean ensemble (candidate + best existing member) and
score that — that ensemble number is the real keep/drop decision, not the single-model score.
See [[ensemble-strategy]] (within-0.15-band + additive-only rules) and [[eval-split-method]].

**Why:** ensemble/composition changes transfer to the LB at ~0.7-1.4x (structural), vs ~0.3x for
recipe nudges — so a decorrelated member is high-transfer, and several "failed to beat baseline"
candidates may still be net-positive as members. Reframes the whole Phase-2 evaluation.

**2. Zip + notify on every potential-submission round.** Whenever a composed candidate is ready,
BUILD THE ZIP (build_submission_zip.py + verify_zip.py) and TELL the user — they submit manually
and report the grade. NEVER auto-submit. Every confirmed gain -> new zip in D:\avv\submissions\.
UPDATE 23/07 ("stop doing fragment improvement"): NO MORE single-member/single-slot swap zips —
r17 (+0.0024) and r18 (+0.0028) proved single-slot changes are unmeasurable on the LB and waste
submission rounds. BUNDLE everything confirmed into ONE combined zip per round (e.g. chair trio
+ all-5-tower swaps together). The old "one clean swap per zip for attribution" rule is
superseded: attribution comes from eval-split + in-sample checks, not from burning LB rounds.
UPDATE 22/07: user wants an ACTIVE PushNotification the moment a new submission zip drops (not
just a passive mention in the reply) — send PushNotification when a verified zip is built and
ready to submit. Also standing: keep BOTH GPUs at maximum utilization (use a claim-based job
pool, not rigid lanes, to avoid idle gaps — pool_runner.sh pattern).

**Pipeline note (confirmed against run_dataset.sh):** the lens-field is a POST-RENDER warp applied
to the ENSEMBLE MEAN, not per-member (ensemble_renders -> png_ens -> apply_field -> png), tower
scenes only. New tower family branches join the pre-field average; field logic unchanged. Video
scenes get no field — members average directly.

Full live plan: /mnt/d/avv/CANDIDATE_PLAN.md. [[round2-dataset]] [[audit-practice]]
