# Round 3 Strategy — City-Scale Drone NVS

> Offline export (2026-08-16) of the "City-Scale Flight Plan" research briefing
> (6-lane web-grounded sweep, 2026-08-14, ~160 sources fetched) plus later corrections.
> **Hardware update:** the on-site machine is a single **H200 (sm_90, 141 GB VRAM)** —
> the 2×16 GB constraints below apply only to the local prep machines; on-site, memory
> pressure largely disappears and the never-measured capacity axis (>8M gaussians)
> becomes testable.

## The call, in four lines

1. **Stay on gsplat and extend our own trainer** (`original/train_gsplat.py`). gsplat is
   our only proven build (sm_120 locally; plain sm_90 on-site), and it now ships the two
   things we'd otherwise port: Grendel-style multi-GPU distributed training (merged
   upstream) and 3DGUT. Add VastGaussian-style independent cell partitioning + gated
   per-image appearance machinery.
2. **CityGaussianV2 is the benchmark-proven fallback** (maintained repo, limited-VRAM
   knobs, real Mill-19/MatrixCity lineage; needs torch ≥ 2.7 bump + 3 CUDA-fork
   recompiles). Probes: CityGS-X (best per-GPU memory numbers), Momentum-GS.
3. **The round is won at SfM + partitioning + appearance level, not in operators** —
   the Rounds 1–2 post-mortem conclusion restated: base reconstruction was the gap.
4. **Published wall-clocks assume 4–8 datacenter GPUs.** One H200 is roughly in that
   class for memory but not throughput: plan overnight-scale runs, not paper numbers.

## What the organiser said — and didn't

Said: expansion to Large-Scene NVS; drone imagery over a wide city area; emphasized
*large scenes, many viewpoints, maintaining spatial consistency*.
NOT said: dataset, image counts/resolution, metric definition, dates, rules on external
priors, test-pose distribution (interpolation vs extrapolation). All prep is restricted
to work robust to these unknowns; metric-tuning before the metric is known is waste.

## Method landscape (2026 state, key facts)

**Partition-then-train** (the family that fits any hardware):
- VastGaussian (CVPR24): airspace-aware cell assignment, decoupled appearance,
  overlap-expand→trim→merge. Cells trained in 10.4–11.9 GB (V100) — the only sub-16 GB
  datapoint in the field. No official code; reimplementations are poor.
- CityGaussianV2 (ICLR25): blocks + LOD, VRAM knobs (`max_cache_num`, downsampling),
  "no limit on GPU amount". MatrixCity-Aerial 27.23/0.857/0.169.
- Momentum-GS (ICCV25): blocks decoupled from GPU count, momentum-teacher consistency;
  24 GB floor default. CityGS-X (ICCV25): 1.4–2.6 GB/GPU on 4×4090, 5k imgs in 5 h —
  largest memory headroom reported; CUDA 11.6-era stack.
- Hierarchical-3DGS (SIGGRAPH24): chunk-bounded VRAM, zero inter-GPU comm, but leaf
  quality ≈ vanilla 3DGS and the heaviest build surface (6 submodules).

**Joint distributed:** gsplat's built-in distributed mode (from Grendel, ICLR25 oral) is
the runnable member. BlitzGS/Splaxel (2026) post the best numbers (Rubble 27.91 PSNR,
~38 min) but assume 4–8 datacenter cards / have no public code. Irrelevant on one GPU.

**Capacity valves:** CLM (ASPLOS26; 102M gaussians in 20.8 GB on a 4090, torch ≥ 2.6).
GS-Scale is Intel-CPU-only (our dev CPU is AMD). Likely unnecessary on the H200.

**Cross-cutting 2026 facts:**
- Every candidate repo except gsplat pins torch ≤ 2.3 (pre-Blackwell) — on our local
  sm_120 machines they need a torch bump + CUDA-fork recompiles before anything runs.
  (On sm_90 they'd build as published, but we prep locally.)
- **No published city-scale pipeline uses 3DGUT** — distorted-space rendering is still
  our differentiator if the drone cameras carry distortion.
- SoccerNet-2026 NVS winner (65 teams): EFA-GS densification variant + up-weighted loss
  on the underrepresented camera class + Depth-Anything-V2 depth supervision + a
  **3-model pixel-average ensemble forked from a shared 90k checkpoint** — the winning
  recipe is our ensemble playbook at our compute class.

## "Spatial consistency" decoded (the organiser's hint)

Three mechanisms likely scored implicitly (their failures are visible in single
held-out views):
1. **Per-image appearance decoupling with a deployable test-time policy** — drone
   passes drift in exposure/WB/sun. Canonical: VastGaussian per-image transform,
   dropped at render. Test policy: nearest-pose training embedding (test GT is never
   available). Failure = uniform PSNR offset on every test view.
2. **Floater suppression at altitude transitions / coverage holes** — airspace-aware
   partition supervision; *selective* handling only (our record: global depth priors
   lose — see the kill list). Post-hoc view-consistency floater culling first.
3. **Chunk-boundary seam consistency** — overlap-expand→trim→merge; scored whenever a
   test view straddles cells. (gsplat's distributed index-parity sharding avoids the
   problem by construction on multi-GPU; on one H200 with big VRAM, fewer/larger cells
   also reduce it.)
Cross-cutting ceiling: **SfM/pose quality** — misregistered poses cap PSNR everywhere
and mimic blur. GLOMAP is now a first-class COLMAP mapper; COLMAP has EXIF gravity
priors; rolling shutter matters above ~8 m/s flight speed.

## Benchmarks (for rehearsal; sizes vs our 629 GB ext4)

| Dataset | Size | Images | Bar (PSNR/SSIM/LPIPS @1600 px protocol) |
|---|---|---|---|
| Mill-19 Building | ~11 GB | 1,920 @ 4608×3456 | 23.5 / 0.82 / 0.13–0.21 |
| Mill-19 Rubble | ~9.8 GB | 1,678 | 26.9–27.9 / 0.86 / 0.13–0.19 |
| MatrixCity small-city aerial | 31.3 GB | 5,621 | 26.9–29.1 / 0.85–0.88 / 0.17–0.26 |
| UrbanScene3D Residence/Sci-Art | unknown (full set 1.43 TB) | 2,582 / 3,019 | 22.7–24.3 / 0.84–0.88 / 0.12–0.20 |

Trap (verified): the Mega-NeRF `*-pixsfm.tgz` files for UrbanScene3D scenes are
KB-sized pose-only archives — images ship separately (Dropbox/GDrive). Standard eval
downsamples the longer side to 1600 px; numbers are only comparable under that protocol.

## What transfers from Rounds 1–2

- The methodology wholesale (gates, isolated-every-k eval splits, provenance
  discipline, ops rules) — our real asset; see `knowledge/00_distilled_memory.md`.
- 3DGUT (if cameras are distorted — check day 1), the lens-field machinery
  (`fit_field.py`/`apply_field.py`, refit per-scene on the new rig), the fork-ensemble
  playbook, the churn/capacity tuning expertise.
- NOT transferable: the operator zoo constants (per-scene lambdas, gate ratio 0.84×,
  field gain 1.30) — all become priors to recalibrate on Round-3 probes.
