---
name: fastgs-cuda-crash-fixes
description: "Root causes and fixes for the random \"CUDA illegal memory access\" crashes in FastGS training on RTX 5070 Ti"
metadata: 
  node_type: memory
  type: project
  originSessionId: 60d62c42-eff2-4507-b2be-4de7dabfda94
---

FastGS training crashed randomly with async "illegal memory access" (= upstream issue fastgs/FastGS#7, unfixed there). All causes found and fixed locally (2026-07-04/05); sanitizer-clean afterwards.

**THE root cause (fix #4, the big one):** `processTiles` in diff-gaussian-rasterization_fastgs/cuda_rasterizer/auxiliary.h summed `max_tile_v - min_tile_v` raw; for degenerate slices the intersection interval comes back inverted (from the intentionally-inverted init values) making the difference NEGATIVE, so the counting pass could return e.g. -3, stored as uint32 tiles_touched = 4294967293. That corrupts the InclusiveSum offsets (locally decreasing), and duplicateWithKeys then writes past the binning buffer. Fix: `tiles_count += max(0, max_tile_v - min_tile_v);`. Verified: a captured crashing input (snapshot_fw.dump replay) went from 100% crash to 10/10 clean, compute-sanitizer 0 errors.

Supporting fixes, all still in place:
1. **fused-ssim shared-memory overflow**: submodules/fused-ssim/ssim.cu conv scratch was `CCX = BX + 0` (32 cols) but written at column threadIdx.x+5 → needs BX+10. Fixed defines; matches reference SSIM to 2e-8.
2. **SnugBox write budget + padding**: duplicateToTilesTouched/processTiles write pass is capped at the counted slot budget and pads shortfall with the last valid key (identifyTileRanges also tolerates 0xFFFFFFFF sentinel tiles). Protects against count/write FP divergence between the two passes.
3. **Backward prefetch overread guards** (`in_bucket`/`pix_ok` in PerGaussianRenderCUDA): loads past sampled_ar/pixel_colors were value-unused but could fault on Blackwell.
4. Also fixed: the python wrapper's debug branch unpacked 7 of 9 return values (stale upstream code) — needed for `--debug` runs.

Debug recipe that cracked it: run train.py with `--debug` (makes the rasterizer's CHECK_CUDA sync+name the failing stage and dump snapshot_fw.dump with the exact failing inputs via cpu_deep_copy_tuple) → replay snapshot against `_C.rasterize_gaussians` for a deterministic repro → compute-sanitizer on the single call → device printf of the violated invariant. Async CUDA errors surface at unrelated call sites (rasterizer has CHECK_CUDA disabled without debug), so never trust the reporting traceback line.

`--data_device cpu` (default in run_scenes.sh) keeps ~3.7GB of GT images in RAM; not the crash cause but good VRAM hygiene. run_scenes.sh retries training up to 3x as general insurance.
[[fastgs-env-and-build]]
