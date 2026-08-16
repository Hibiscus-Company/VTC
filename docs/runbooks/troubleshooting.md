# Troubleshooting — known failures and their fixes

## Training

**CUDA OOM during training** — ladder, in order: reduce `--cap_max` (8M→5M→3M);
image downsampling at load; on the H200 (141 GB) OOM at sane caps usually means a
runaway densification — check the gaussian count in the log, tighten `--refine_stop`.

**Training crashed mid-run (any cause)** — if launched with `--ckpt_every`,
`runs/<name>/ckpt_latest.pt` is render-ready (same schema as ckpt.pt; carries `step`).
Render it, score it, decide whether to relaunch. A 70%-run is a valid ensemble member.

**Loss goes NaN / gaussian count collapses on a glossy or view-inconsistent scene** —
the Rounds 1–2 bonsai signature. Fix is the CHURN schedule, not capacity: stop
relocation/noise early (`--refine_stop`/`--noise_stop` well before end; see
`configs/recipes/video_bonsai_churn.args`).

**`--ut` render shows phantom smears at frame corners** — large negative k1 folding.
Render that scene with `--ut_render warp --distort auto --sparse <sparse/0>`.

**Same-seed reruns produce identical members** — expected: seeds pin all RNGs.
Ensemble diversity comes from CONFIG jitter (schedule/scale_reg/length), not reruns.

## Environment / build

**`ModuleNotFoundError: gsplat`** — wrong env. The pipeline runs in the `gsplat` env
(torch 2.7.1+cu128), not `fastgs2`. On-site: see `env/SETUP.md`.

**gsplat/fused-ssim won't compile** — set `TORCH_CUDA_ARCH_LIST` for the actual GPU
(H200 = `"9.0+PTX"`, RTX 5070 Ti = `"12.0+PTX"`) and `CUDA_HOME=$CONDA_PREFIX`;
`pip install --no-build-isolation <src dir>`. gcc13 + the old FastGS rasterizer needed
`#include <cstdint>` — irrelevant now (that stack is retired) but the same class of fix
may apply to other old CUDA forks.

**LPIPS tries to download VGG weights and fails (offline)** — torchvision needs
`vgg16-397923af.pth` in `~/.cache/torch/hub/checkpoints/`. `env/fetch_weights.sh`
stages it; copy it into place on the sealed machine.

## Operations

**Queue died when the session closed** — it wasn't detached. `setsid nohup ... & disown`
and VERIFY the SID (`ps -eo pid,sid,cmd`). See onsite_playbook checklist B.

**A queue silently skipped jobs** — stale `.DONE` markers from a reused output dir.
One fresh dir per job, always.

**`pkill`/`pgrep` killed the launcher itself** — the pattern matched your own command
line. Match on the child binary/script name only. (Happened twice in Rounds 1–2.)

**bash `local A=$1 B=$A` gives empty B** — bash does not see assignments made earlier
in the SAME `local` statement. Separate statements. (Silently corrupted an output path
once.)

**Zip rejected / oversize** — the cap is 350 MiB = 367,001,600 bytes. Both
`build_submission_zip.py` and `verify_zip.py` budget MiB now; if a zip is near the
limit, drop the quality ladder one step on the LARGEST scene first.

**Scores look ~1 pt off vs older notes** — scorer drift existed in Rounds 1–2
(eval_score changed 17/07; pre/post numbers incomparable). Only compare numbers
produced by the SAME scorer build; note the scorer version in the day log.

## Data

**COLMAP images.bin has more poses than images on disk** — normal (organiser drops
frames); the loader skips missing files. Test poses may appear in images.bin with
keypoints — DO NOT use test-frame keypoints for anything (gray zone; quantified
worthless anyway: +0.05 residual).

**Portrait / mixed-resolution scenes** — supported (Rounds 1–2 chair was 720×1280
portrait); W/H come from the CSV per pose.
