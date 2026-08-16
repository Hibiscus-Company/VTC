# On-Site Playbook — 3 days, 07:00–19:00, machine sealed overnight (but running)

The overnight window (19:00→07:00, ~12 h) is **half of the total compute budget and the
only time long runs fit**. The entire day is organized around launching it well.

## Day structure

```
07:00  MORNING RECOVERY (checklist A)          — harvest the overnight results
07:30  score + decide (eval harness, gates)    — what worked, what ships, what's next
09:00  DAYTIME LOOP: short probes (≤2 h each)  — recipe A/Bs, renders, field fits,
                                                  ensemble builds, zip candidates
17:00  OVERNIGHT PLANNING — pick tonight's runs; total wall-clock must fit ~11.5 h
18:00  OVERNIGHT LAUNCH (checklist B)          — everything queued and verified
18:45  final verification + notes for tomorrow-you
19:00  sealed
```

## Checklist A — morning recovery (07:00)

- [ ] `nvidia-smi` — is anything still running? If yes, decide: let finish or kill.
- [ ] `tail -50` every log in `runs/overnight_<date>/*.log` — completed? crashed? when?
- [ ] Crashed run: `ckpt_latest.pt` exists (thanks to `--ckpt_every`) → render it anyway;
      a 70%-trained model is a usable ensemble member and a data point.
- [ ] Render + score everything that finished (`scripts/render_pipeline.sh`,
      `original/eval_score.py` against the eval split).
- [ ] Write results into the day log (one file, `runs/DAYLOG.md`, append-only:
      what ran, numbers, verdicts). This file is the on-site experiment ledger —
      the Rounds 1–2 ledger is why we know anything; keep the habit.

## Checklist B — overnight launch (18:00, allow a full hour)

- [ ] Queue file written: `scripts/overnight_queue.sh` runs jobs SEQUENTIALLY (one GPU);
      sum of expected wall-clocks ≤ 11 h (leave margin — a hung job wastes the night).
- [ ] Every training arm has `--ckpt_every 2000` (or similar) — a crash at hour 8 must
      leave a renderable checkpoint.
- [ ] Launch detached: `setsid nohup bash scripts/overnight_queue.sh > runs/overnight_<date>/queue.log 2>&1 < /dev/null & disown`
- [ ] **Verify the detach**: `ps -eo pid,sid,cmd | grep overnight` — the SID must differ
      from your shell's. (A non-detached queue dies when Jupyter/your session ends;
      this exact mistake once cost 12.5 GPU-hours.)
- [ ] Verify the FIRST job is actually training (log advancing, `nvidia-smi` shows load)
      before you leave. A typo discovered at 07:00 wastes the whole night.
- [ ] Each queue item writes to its OWN output dir under `runs/overnight_<date>/`;
      never reuse a dir (`.DONE`-marker collisions silently skip jobs).
- [ ] The queue script must `touch <dir>/.DONE` after each job and continue on failure
      (`||` guards) — one bad arm must not kill the rest of the night.

## Priorities under time pressure (from the Rounds 1–2 post-mortem)

1. A submitted mediocre zip beats an unsubmitted good one — **always have a valid
   submission zip ready from day 1**, upgrade it as results land.
2. Structural moves beat recipe nudges (ensembling +1.82, 3DGUT +0.9 vs +0.03-class
   tweaks). If a structural idea and a tweak compete for the overnight, the structural
   idea wins.
3. The frontier died operationally last time, not scientifically — protect long/novel
   runs with `--ckpt_every`, sequential queues, and the detach checklist above.
4. Never let a script delete a checkpoint behind a shipped result.

## Submission discipline

- Cap: **350 MiB = 367,001,600 bytes** (MiB!). Build with
  `original/build_submission_zip.py` (budgets MiB), gate with `original/verify_zip.py`.
- JPEG q100 / subsampling=2, single-generation from PNG archives; NEVER re-encode a JPEG.
- A human submits. Always. Verify the zip → tell the team → someone clicks.
