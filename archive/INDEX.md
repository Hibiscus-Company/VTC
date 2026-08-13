# archive/ — VAR 2026 "BTS Digital Twin" working record

Everything durable behind the graded submissions, gathered into the repo on **2026-08-13** so it
survives the machine it was produced on. 2,889 files, ~20 MB.

Three of these sources were **volatile** — they lived outside any repository and outside any
backup. In particular `archive/jobs/` was the *only* copy of the production build scripts.

| Archive path | Collected from | What it is |
|---|---|---|
| `avv/` | `/mnt/d/avv` | Experiment ledger, analysis probes, per-run logs and manifests |
| `jobs/` | `~/.claude/jobs/1c9cf7e9/tmp` | **Production pipeline scripts + probe code** (volatile scratch) |
| `memory/` | `~/.claude/projects/…/memory` | Distilled cross-session findings (21 notes + index) |
| `env/` | `~/VAR2026_env`, `/mnt/d/avv/DELIVERY/environment` | Reproducible environment specs |
| `spec/` | repo root | Competition specification (`.docx`) |

---

## 1. Start here

| File | Why |
|---|---|
| [`avv/EXPERIMENTS.md`](avv/EXPERIMENTS.md) (397 KB) | The lab notebook. Every arm, its measurement, and its verdict, in chronological order. The single most valuable file here. |
| [`memory/MEMORY.md`](memory/MEMORY.md) | One-line index into the 21 distilled findings — read this before re-deriving anything. |
| [`../REPRODUCE_r36.md`](../REPRODUCE_r36.md) | Step-by-step rebuild of the graded submission (in the repo root, not in `archive/`). |
| [`avv/IDEA_LEDGER.md`](avv/IDEA_LEDGER.md) | Ideas proposed, and for each: shipped / measured-and-killed / never-executed. |
| [`avv/BRIEFING_private_set2.md`](avv/BRIEFING_private_set2.md) | Round-2 dataset: 5 towers + 2 indoor scenes, geometry, metric definition. |
| [`env/WEIGHTS_MANIFEST.md`](env/WEIGHTS_MANIFEST.md) | The 62 trained models behind the ensemble, and which checkpoints still exist. |

## 2. `jobs/` — the production pipeline (the volatile rescue)

`build_r22.sh` … `build_r37.sh` (20 scripts) are the actual submission builders: they run the
ensemble average, energy restoration, lens-field warp, JPEG encode and zip. `build_r36.sh` produced
the graded 77.7230 entry; `build_r37.sh` produced the built-but-unsubmitted successor.

Alongside them: 487 Python probes and 198 shell drivers — training queues (`q*.sh`), scene
production runs, and the diagnostics that killed or confirmed each hypothesis (`gate_k2.py`, the
pre-registered ensemble gate; `score_split.py`, the two-population bonsai scorer).

Sub-directories are per-investigation scratch: `lens`/`fieldrefit` (lens-field correction),
`lever_ens`/`mixlever`/`reslever` (ensemble weighting), `rz_*` (per-scene resizing), `refute`
(adversarial checks), `eda`.

## 3. `avv/` — ledger, probes, run records

- **Docs:** `EXPERIMENTS.md`, `IDEA_LEDGER.md`, `PLAN_TO_85.md`, `CANDIDATE_PLAN.md`,
  `CONSULT_BRIEF{,2}.md`, `ENSEMBLE_MANIFEST.md`, `PROJECT_RETROSPECTIVE.html`,
  `TIER{1,3}_DELETED.txt` (what disk cleanup removed).
- **Probes:** `metric_probe/` (24 — how the score responds to encode, radial position, contrast),
  `data_supervision_probes/` (18 — coverage, blur, routing), `cvclassic/` (12 — classical CV
  baselines), `geoflat/` (12), `depthcurve/` (15), `evalsplit/` (10).
- **`r42_bonsai78/`** (1,026 files) — the campaign to lift bonsai to ≥78: oracle bound, blur
  predictor, diagnosis, operator, training arms, arithmetic.
- **Run records:** `r14/`…`r45_prod/`, `fields*/` (lens-field fits, text only), `submissions/`
  (`*.PROVENANCE.txt` — what went into each shipped zip).

## 4. `env/`

- `VAR2026_env_prelim/` — the preliminary-round package sent to the organisers: pinned
  `requirements.txt`, `verify_env.py` (exit 0 = ready), and **`gsplat_cuda128.patch`**, without
  which gsplat's CUDA sources do not compile under CUDA 12.8.
- `postverification/` — full conda exports for both environments (`fastgs2.yml`, `gsplat.yml`).

---

## Deliberately excluded

| Excluded | Size | Why / where it lives |
|---|---|---|
| Checkpoints, renders, submission zips | ~29 GB | Binary; delivered in `VAR2026_BTS_postverification.zip` (28.83 GiB, SHA256 `f6fd72a3…c2f91b`) on Drive. Git is the wrong store. |
| `.npy` / `.png` intermediates | GBs | Regenerable from the scripts kept here. |
| `an_plaza/` on the D drive | 50 GB | A different project (histopathology), still in use by others. |
| `gsplat_src/`, `Difix3D`, `ubs_repo`, `dbs_repo` | 354 MB | Third-party clones. `gsplat` is pinned by commit + patch in `env/` instead. |
| `data/SCARED-E2E` | — | Another project's dataset. |
| Session transcripts (`*.jsonl`) | 308 MB | Raw conversation logs — they would bloat git permanently and contain personal contact details. The durable content is already distilled into `memory/` and `EXPERIMENTS.md`. |
| `gates_queue.log` (10.7 MB), `gsplat_fix3.log` (4.1 MB) | 15 MB | Over the 2 MB per-file cap; queue/build noise, conclusions already recorded in the ledger. |

Collection rule: text and code only (`.py .sh .md .txt .json .log .csv .yml .patch .html`),
per-file cap 2 MB, `__pycache__` and `.git` pruned.
