# Mip-Splatting integration into FastGS rasterizer — implementation spec

Branch: `mip-splatting`. Goal: add Mip-Splatting's two anti-aliasing filters to
the FastGS CUDA rasterizer. Justified by measured camera scale spread
far/near = 2.3–4.1× (see PROMPT.md §4). Target metric: LPIPS/SSIM.

Two independent components. **(A) is self-contained and lower-risk → do & test
first. (B) is the bigger scale-aliasing win → do second.** Each needs matching
forward + backward or training silently diverges.

## (A) 2D screen-space filter with opacity compensation ("antialiasing")

Replaces the fixed `cov[0][0]+=0.3; cov[1][1]+=0.3` at
`cuda_rasterizer/forward.cu:115` (no opacity comp today) with the
determinant-ratio-compensated low-pass.

**Forward (preprocessCUDA, forward.cu ~L224–260):**
```
float3 cov = computeCov2D(...);              // BEFORE adding filter (edit computeCov2D to NOT add 0.3)
float det_orig = cov.x*cov.z - cov.y*cov.y;
float3 cov_f = {cov.x + 0.3f, cov.y, cov.z + 0.3f};
float det_f = cov_f.x*cov_f.z - cov_f.y*cov_f.y;
float comp = sqrtf(max(0.0f, det_orig / det_f));   // <=1, opacity attenuation
// use cov_f for conic; multiply stored opacity by comp
conic_opacity[idx].w = opacity_activated * comp;
```
Store `comp` (or det_orig,det_f) per-gaussian in a new geom buffer for backward.

**Backward (backward.cu, computeCov2DCUDA + preprocess):**
`d(opacity*comp)/d(params)` has two paths: through `opacity` (existing) and
through `comp` (new, depends on cov2D → cov3D → scale/rot/mean). Add
`dL_dcomp = dL_dopacity_scaled * opacity` and propagate `d comp / d cov`:
```
comp = sqrt(det_orig/det_f)
d comp/d cov.x = 0.5/comp * (cov.z*det_f - det_orig*cov_f.z)/det_f^2   (and symmetric for cov.z, cov.y)
```
Chain into existing `dL_dcov2D` accumulation. **This gradient is the risky part
— unit-test against finite differences on a single gaussian before full train.**

Wire the existing (currently no-op) `antialiasing` flag in PipelineParams →
GaussianRasterizationSettings → kernel, so (A) is toggleable.

## (B) 3D smoothing filter (per-gaussian sampling-rate low-pass)

Each gaussian gets scalar `filter_3D` = (max over training cams of
dist_to_cam / focal) * 0.2, added to the 3D covariance diagonal before
projection, with 3D-determinant opacity compensation. This is what actually
fixes the 2.3–4.1× scale aliasing.

1. **New per-gaussian buffer** `filter_3D` (float, P). Add to GeometryState or a
   persistent model tensor (survives densification → store on GaussianModel like
   scaling, resized in densify/prune).
2. **New CUDA kernel `computeFilter3D`**: for each gaussian, loop over all
   training camera centers (upload as a (Ncam,3) tensor + their focal), compute
   min sampling interval, set filter_3D. Run once after init and after each
   densification (cheap: P × Ncam).
3. **Apply in computeCov3D** (forward.cu): `cov3D += filter_3D*I` (3×3 diag),
   opacity *= sqrt(det3D_orig/det3D_filtered). Backward: analogous determinant
   chain rule on the 3D covariance + the filter contribution to opacity grad.
4. Expose `filter_3D` at inference (render_test_poses) — it's baked into the
   saved model if we store activated scales, else apply at load.

## Validation protocol (public GT, our eval harness)

- Baseline = current density-adaptive g2 model per public scene.
- Test order: (A) alone → (A)+(B). Keep only if mean public Score improves AND
  no scene regresses. Watch LPIPS specifically (weight 0.4).
- Finite-difference gradient check on both new opacity-comp terms BEFORE any
  30k run (catches the high-risk backward bug in seconds, not hours).

## Build

`CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="12.0+PTX" pip install
--no-build-isolation ./submodules/diff-gaussian-rasterization_fastgs`
(env `fastgs2`). Remember the `#include <cstdint>` and the `max(0,...)` tile-count
fixes already on main are NOT on this branch until merged — rebuild from branch.

---

## ROUND-2 AUDIT CORRECTIONS (2026-07-12) — READ BEFORE IMPLEMENTING

Verdict: **implement (A) 2D opacity-compensated filter ONLY. Skip (B) 3D filter** — the spec above has two math errors ((i) must use MIN dist/focal over seeing cameras, not max; (ii) filter is a std: filter_3D=(d/f)·sqrt(0.2), added as filter², not linear 0.2·(d/f) on the diagonal), and the official impl is pure-Python autograd (get_scaling_with_3D_filter) — a CUDA kernel is the wrong layer. On merit (B) is likely net-negative here anyway: test poses are interpolated in-trajectory (no sampling-rate shift to fix) and added low-pass is the closed failure family.

Corrections to (A):
1. **cov.y backward is NOT "symmetric"**: ∂D0/∂b = ∂D1/∂b = −2b → dcomp/db = (b/comp)·(D0−D1)/D1². Pattern-matching the cov.x formula gives an off-by-2×.
2. **comp guard**: use `comp = sqrt(max(0.000025f, det_orig/det_f))` (comp≥0.005). `max(0,·)` allows comp=0 → 1/comp = inf gradients on near-degenerate thin splats.
3. **t≤0 tile guard (fork-specific crash)**: duplicateToTilesTouched/processTiles compute t=2·log(con_o.w·255) with no guard (auxiliary.h:382). Compensated opacity < 1/255 → negative t → sqrt(negative) → NaN bbox → the old illegal-memory-access crash family. Add `if (con_o.w*255.0f <= 1.0f) { pad/skip; return 0; }` in BOTH passes.
4. **Existing opacity backward must be scaled**: render backward computes dL/do′ w.r.t. EFFECTIVE opacity o′=o·comp → chain `dL_do = comp·dL_do′` (and dL_dcomp = o·dL_do′). Forgetting the comp scale silently over-steps opacities.
5. h: keep 0.3 as default but expose as constant; official 2D-filter value is 0.1 (sharper) — worth an A/B.

Minimal diff estimate for (A): forward.cu ~15-20 LoC (comp computed in preprocess, stored per-gaussian buffer, multiplied into con_o.w), rasterizer_impl ~6 LoC (buffer), backward.cu ~30-40 LoC (3 comp terms into dL_da/db/dc in computeCov2DCUDA — dilated a,b,c already reconstructed at lines 199-203; + opacity chain), auxiliary.h ~4 LoC (guard), plumbing antialiasing flag end-to-end ~30-40 LoC. Total ≈ 90-140 LoC. REQUIRES RETRAIN (comp dims every splat). Keep the FD gradient-check protocol before any 30k run.
