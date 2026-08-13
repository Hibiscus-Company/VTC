---
name: never-delete-checkpoints
description: "Production scripts that `rm -f ckpt.pt` after rendering destroyed 50 of 62 submission checkpoints — the organiser later required them"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1c9cf7e9-44fa-44d0-9bb8-375127c7789b
  modified: 2026-08-01T02:19:28.443Z
---

**Never let a training script delete its checkpoint.** Archive it first:

```bash
mkdir -p /mnt/d/avv/WEIGHTS
cp -f $M/ckpt.pt /mnt/d/avv/WEIGHTS/<scene>_<tag>.pt   # then, optionally, rm the working copy
```

**Why:** on 2026-07-31 the organiser required "trọng số của kết quả tốt nhất" as part of a
post-verification package. By then **50 of the 62 checkpoints behind the graded submission no
longer existed** — every production script ended with `rm -f $M/ckpt.pt` after the model had
rendered its test views. The reasoning at the time was sound (a checkpoint is 1.1–1.8 GB; 62 of
them need ~90 GB the machine did not have) but it optimised for disk against a requirement nobody
had asked for yet. Disk is recoverable; a deleted checkpoint is not.

Two members were saved only by racing the deleter with a polling loop that copied `ckpt.pt` the
moment it appeared, in the window between render and `rm`.

**How to apply:** keep at minimum one checkpoint per scene per shipped recipe. If disk is tight,
delete old *experiment* checkpoints (they are re-derivable and nobody will ask for them), never
the ones behind a submission. Prune by "is this behind something we shipped?", never by age or
size alone.

**The mitigating fact, worth knowing in advance:** the submission stayed reproducible anyway,
because every model's *rendered PNG output* was retained and that — not the checkpoint — is what
the ensemble, restore, lens-field and encode stages actually read. Rebuilding bonsai from its 13
member render archives reproduced the shipped zip **28/28 byte-identical**. Render archives are
the load-bearing artifact; checkpoints are for the audit trail.

See [[detached-gpu-jobs]], [[production-harness]].
