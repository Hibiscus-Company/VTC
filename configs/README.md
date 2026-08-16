# configs/

## recipes/ — training recipes as `.args` files

One line of `train_gsplat.py` CLI args per file; consume with:

```bash
python original/train_gsplat.py --source $SCENE/train --images images \
  --out runs/$NAME $(cat configs/recipes/<recipe>.args) --seed 42 --ckpt_every 2000
```

Provenance (extracted verbatim from the as-run Rounds 1–2 production scripts in
`archive/jobs/` — these exact args produced the graded 77.7230 submission):

| Recipe | Used for | Notes |
|---|---|---|
| `tower_ut_production.args` | drone scenes with radial distortion (k1≠0) | 3DGUT; seeds 42/7/13/101/202 were the tower ensemble members |
| `tower_ut_ema_member.args` | same + parameter-EMA member | EMA valid only after refine_stop (topology frozen) |
| `pinhole_aa_production.args` | pinhole scenes (chair-class) | antialiased mode (no `--ut`); EMA 0.99 |
| `video_bonsai_churn.args` | view-inconsistent/glossy scenes that collapse | the churn fix: stop relocation+noise EARLY |
| `video_bonsai_long45k.args` | the best-measured bonsai recipe (r37 members) | length curve peaked at 45k |
| `round3_default.args` | Round-3 starting point, UNTUNED | add `--ut` per the day-1 camera check; on the H200, `--cap_max` well above 8M is unexplored territory (was never measurable on 16 GB cards) — sweep it |

Rules that travel with these numbers:
- `scale_reg` 0.1 was the BONSAI optimum at 30k iters; a 60k scene gets ~2× the total
  regularisation at the same coefficient — port the budget, not the number.
- Ensemble members: config-jitter (schedule/length/scale_reg/EMA), never bare seed
  reruns of the same config (seeds are pinned; they do not decorrelate).
- Every carried constant is a PRIOR on Round-3 data, not a fact.

## paths.example.sh

Machine-specific roots, sourced by the `scripts/` runners. Copy to `paths.sh`
(gitignored) and edit per machine.
