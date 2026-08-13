---
name: detached-gpu-jobs
description: Long GPU queues must be launched with setsid or they die when the Claude Code process exits — this cost 12.5 GPU-hours once
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-07-30T05:48:29.841Z
---

Launch every long-running training queue as
`setsid nohup bash queue.sh > /path/queue.log 2>&1 < /dev/null & disown`
and verify it took its own session (`ps -eo pid,sid,cmd` — the SID must differ from the launcher's).

**Why:** on 2026-07-30 the Claude Code process exited and took every child with it. A tower screen
died 2h07 in and a UBS-6D run died **10h30** in; neither had written a checkpoint, so ~12.5 GPU-hours
produced nothing. Plain `cmd &` from a tool call is a child of that process and shares its fate.
Related failure the same day: a background waiter was set on one queue's DONE marker but not the
other's, so a GPU sat idle 2h18 after its job finished.

**How to apply:** setsid for anything over ~20 minutes; write full logs to a file rather than
piping through `tail` (a `| tail -N` in an arm function leaves no readable progress signal, and
gsplat's trainer saves exactly one checkpoint at the final iteration, so there is nothing else to
watch); set a waiter on EVERY queue, not just the one being discussed.

Also, a bash trap that silently corrupted an output path the same day:
`local TAG=$1 SR=$2 M=$OUT/$TAG` expands `$TAG` to **empty** — bash does not see an earlier
assignment made inside the same `local` statement. Use separate statements. It would have written
every arm to the same directory and let the first arm's `.DONE` skip the rest.

See [[audit-practice]], [[production-harness]].
