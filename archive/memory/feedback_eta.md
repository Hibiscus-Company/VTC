---
name: feedback-eta
description: "Always give a time estimate upfront for time-consuming tasks (training runs, sweeps, multi-step builds)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-07-21T10:15:32.841Z
---

For any time-consuming task (GPU training runs, eval-split sweeps, multi-stage builds, background
agent research), state an estimated completion time or duration upfront, not just after the user
asks. Give both a duration ("~2h") and, when useful, a wall-clock ETA ("~18:15-18:25").

**Why:** the user explicitly asked for this (21/07) after several rounds of launching background
jobs without an upfront estimate — they had to ask "estimate time left" separately. They want the
estimate proactively, as part of the normal report when a job is launched, not on request.

**How to apply:** When launching a training run, sweep, or other multi-step background job, include
a time estimate in the same message that reports the launch — derive it from measured durations of
similar past jobs (check process start times / log timestamps of comparable prior runs) rather than
guessing from iteration counts alone, since actual wall-clock varies with GPU contention and recipe.
Update the estimate if I later learn the actual duration differed. Applies to this project's FastGS/
NVS competition work generally, not just the specific jobs running when this was said.
