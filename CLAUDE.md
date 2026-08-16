# CLAUDE.md — project context for AI assistants

This repo is Hibiscus's platform for the **VAR 2026 NVS competition** (Round 3: city-scale
drone Large-Scene Novel View Synthesis). Read `docs/00_START_HERE.md` first, then
`docs/knowledge/00_distilled_memory.md` — it is the distilled record of ~170 measured
experiments from Rounds 1–2 and overrides intuition. Do not re-propose anything on its
kill list without new evidence.

## Layout

- `original/` — THE canonical pipeline (flat Python files, same-dir imports, run as
  `python original/<tool>.py`). Read-only reference; personal working copies live in
  `an/`, `bach/`, `tu/` (regenerate with `scripts/make_replicas.sh`, compare with
  `scripts/diff_replicas.sh`). Never edit `original/` casually — promote tested changes
  into it deliberately.
- `docs/` — knowledge base + runbooks (`docs/runbooks/` are the on-site playbooks).
- `configs/` — training recipes as `.args` files (CLI arg lines with provenance comments).
- `scripts/` — bash entry points (train, overnight queue, zip build, replicas, bundle).
- `notebooks/` — Jupyter wrappers for the on-site environment.
- `env/` — pinned requirements, vendored CUDA sources (fused-ssim carries a local
  shared-memory fix — never replace with upstream), build + weights-fetch scripts.
- `archive/` — the complete Rounds 1–2 working record (2,889 files). Historical; do not
  modify. The pre-refactor tree is at git tag `pre-reorg`.

## Standing rules (user directives, learned the hard way)

1. **NEVER auto-submit** anything to the competition. Build the zip, verify it, notify a
   human. Humans submit.
2. **Never delete a checkpoint behind a shipped/graded result.** Archive first. (A
   `rm -f ckpt.pt` habit once destroyed 50/62 submission weights the organiser later
   required.)
3. **Long GPU jobs**: launch with `setsid nohup ... & disown` and verify the SID, or the
   run dies with your process. Use `--ckpt_every` for anything unattended. Never
   pgrep/pkill a pattern that matches your own launcher.
4. **ETAs upfront**: state expected duration when launching anything long.
5. **Submission cap is 350 MiB = 367,001,600 bytes** (MiB, not decimal MB — empirically
   confirmed). JPEG q100/subsampling=2, single-generation encode from PNG archives,
   never re-encode an existing JPEG.
6. **Scoring**: use `original/eval_score.py` / `score_submission.py` (zero-pad conv SSIM
   — fused_ssim differs by up to ~0.003 SSIM and is for the training loss only).
7. **Measure before building**: pre-registered gates, isolated-every-k eval holdouts
   (never contiguous arcs), cross-validate on ≥3 scenes, and treat any constant
   calibrated in a previous round (gate transfer 0.84×, lens-field gain 1.30) as a
   prior to re-verify, not a fact.
8. **Trust organiser geometry** (poses measured correct to 0.018 px in Round 2). Never
   re-solve test poses; always render in the organiser's frame.

## Competition-environment constraints (Round 3 on-site)

3 days on-site, 07:00–19:00; the machine is sealed overnight **but keeps running** —
overnight jobs are the main compute. Hardware: single **H200** (sm_90). No GitHub, no
AI agents in the Jupyter environment, pip may not work (everything must be vendored or
uploaded). AI Q&A is allowed only WITHOUT internal data — so these docs must answer
data-specific questions by themselves.
