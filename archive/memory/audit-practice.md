---
name: audit-practice
description: "How to run the periodic sub-agent audits — two mandates (code errors + game-changer tweaks), read-only"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
---

Run sub-agent audits periodically (user set ~6-hourly; also on request). As of 17/07 the user
wants TWO mandates each cycle: (1) CODE ERRORS — correctness audit of new/changed code; (2)
POTENTIAL GAME-CHANGER TWEAKS — strategy/ROI assessment of the highest-leverage legal moves.

**Why:** the project moves fast (new scripts, pipeline changes, per-scene recipes); a read-only
second opinion catches silent-ship bugs and surfaces levers I'm anchored away from.

**How to apply:**
- Code-error audit → resume the existing audit agent `a05c83f5cc3b23e12` via SendMessage (it has
  full codebase + history context; cheaper than a cold spawn). Round-numbered ("round N audit").
- Game-changer audit → a FRESH general-purpose agent gives an unanchored strategy read from
  /mnt/d/avv/EXPERIMENTS.md + RUNBOOK.md; ask for a ranked "do X, expect ~Y pts, because Z".
- Both READ-ONLY. Require CONFIRMED vs HYPOTHESIS labels and file:line for bugs. Most-severe first.
- Give serious weight; verify any file/flag it cites still exists before acting. See [[eval-split-method]].
