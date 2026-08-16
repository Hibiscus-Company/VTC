# START HERE

You have pulled this repo onto a new machine (or you are on-site). Reading order:

## 1. If you have 15 minutes

1. Root `README.md` — the layout.
2. `knowledge/00_distilled_memory.md` — **the single most important file.** Every
   measured result from Rounds 1–2: what won (with numbers), the kill list of ideas
   that measurably failed (do not retry them), the calibration/transfer rules, and
   the operational rules that were paid for in GPU-hours.
3. `round3/day1_protocol.md` — what to run in the first hours after the data drops.

## 2. If you are setting up a machine

- `../env/SETUP.md` — environment decision tree (local dev machine vs the on-site H200),
  pinned requirements, CUDA-extension builds, offline weights.
- Then `../scripts/smoke_test.sh` — verifies the pipeline end-to-end without a dataset.

## 3. If you are on-site

- `runbooks/onsite_playbook.md` — the 3-day battle rhythm: morning recovery checklist,
  daytime loop, end-of-day overnight-launch checklist (the machine is sealed overnight
  but keeps computing — overnight is most of our GPU budget).
- `runbooks/pipeline_runbook.md` — every pipeline stage as copy-paste commands.
- `runbooks/troubleshooting.md` — OOM ladder, crash patterns, known traps.
- `round3/` — strategy (`strategy.md`), the refereed idea slate (`idea_slate.md`),
  and the day-1 data-drop protocol (`day1_protocol.md`).

## 4. Deep history (rarely needed, never deleted)

- `REPRODUCE_r36.md` — how the Rounds 1–2 graded submission is rebuilt.
- `../archive/INDEX.md` — the complete Rounds 1–2 working record: experiment ledger
  (`archive/avv/EXPERIMENTS.md`, 397 KB, every arm ever run), idea ledger, production
  build scripts as-run, environment specs.
- `history/` — superseded planning docs from Rounds 1–2, kept with STALE banners.
- `reference/` — background reading (e.g., the surgical-3DGS formula tour, studied as
  a template for problem→mechanism thinking).
- Git tag `pre-reorg` — the entire pre-refactor tree, including the retired upstream
  FastGS stack.

## The two rules that protect you from the past repeating

1. **Check the kill list before proposing an idea** (`knowledge/00_distilled_memory.md`
   §5). Nearly every "obvious" improvement was already measured; most lost.
2. **Check the operational rules before launching a long job** (§9 there, and
   `runbooks/onsite_playbook.md`). Every rule exists because its violation once cost
   double-digit GPU-hours or destroyed irreplaceable artifacts.
