# VAR 2026 — Hibiscus NVS Platform

Team platform for the VAR 2026 Novel-View-Synthesis competition
(Round 3: city-scale drone Large-Scene NVS). Forked originally from
[FastGS](https://github.com/fastgs/FastGS); the upstream training stack was retired
after Rounds 1–2 (it survives at git tag `pre-reorg` and in `archive/`) — the live
method is a gsplat MCMC + 3DGUT pipeline in `original/`.

**Start here → [`docs/00_START_HERE.md`](docs/00_START_HERE.md)**

| Folder | What |
|---|---|
| `original/` | Canonical pipeline (train → render → field-correct → ensemble → zip → verify) |
| `an/` `bach/` `tu/` | Per-teammate working replicas of `original/` (see `scripts/make_replicas.sh`) |
| `docs/` | Knowledge base, Round-3 strategy, on-site runbooks, historical record pointers |
| `configs/` | Training recipes (`.args` files) + machine paths |
| `scripts/` | Bash entry points: training, overnight queue, zip build, replicas, bundling |
| `notebooks/` | Jupyter wrappers for the on-site environment |
| `env/` | Pinned requirements, vendored CUDA sources, H200 build + weights scripts |
| `archive/` | Complete Rounds 1–2 working record (read-only) |

Rounds 1–2 result: final graded 77.7230 on private_set2 (top-1: 82.17); full
post-mortem in [`docs/knowledge/00_distilled_memory.md`](docs/knowledge/00_distilled_memory.md).

Licenses: [`LICENSE`](LICENSE) (ours) + [`LICENSE_ORIGINAL.md`](LICENSE_ORIGINAL.md)
(Inria/MPII 3DGS terms — still applies to `original/metrics_utils.py`,
`original/colmap_loader.py`, and the vendored `env/vendor/fused-ssim`).
