# Day-1 Data-Drop Protocol

Run this in the FIRST HOURS after the Round-3 dataset lands. Every check is CPU-cheap
and decides a mechanism gate. `notebooks/01_day1_forensics.ipynb` is the runnable
version; this file is the reference with decision thresholds.

Convention below: `$DATA` = a scene folder (expects `train/images*`, `train/sparse/0`,
`test/test_poses.csv` — adapt if Round 3 ships a different layout, and record the
actual layout in this file immediately).

## 0. Inventory (10 min)

- [ ] Scene count, per-scene image counts, resolutions, disk size.
- [ ] Layout: does each scene have `sparse/0` (cameras.bin, images.bin, points3D.bin)?
      A test-pose CSV? GT for any public scene?
- [ ] Record everything in a `DATA_NOTES.md` next to the data.

## 1. Camera forensics → 3DGUT decision (15 min)

```bash
python original/preflight.py --data_root <dataset root>   # validates every scene, prints camera model + k1
```
- [ ] Camera model and k1 per scene.
- **Decision:** any radial distortion (|k1| > 0) → train with `--ut` (3DGUT), render
  `--ut_render native`. Pure pinhole (k1 == 0) → drop `--ut`, use antialiased mode
  (Rounds 1–2: +0.49/scene on pinhole video). Large NEGATIVE k1 (≲ −0.05) → render via
  `--ut_render warp` (forward distortion folds; warp path is safe).
- Rationale: the +0.9 3DGUT win was earned at k1 ≈ +0.009 — small distortion still pays.

## 2. Pose sanity → SfM decision (30 min)

- [ ] Reproject `points3D.bin` through 20 random train cameras; median reprojection
      error. (Cell in the forensics notebook.)
- **Decision:** median < 1.5 px → TRUST organiser geometry (Rounds 1–2: 0.018 px).
  Worse, or poses absent → GLOMAP/COLMAP contingency — but NEVER re-solve test poses;
  always render in the organiser's frame.
- [ ] Confirm test_poses.csv convention: rows should match images.bin entries
      (world-to-camera qw,qx,qy,qz,tx,ty,tz). Spot-check any overlap.

## 3. Test-pose structure → holdout design (15 min)

- [ ] Plot test poses vs train poses (forensics notebook cell; camera centers, 3 views).
- **Decision:** test interleaved within the capture (Rounds 1–2 style) → eval split =
  isolated every-k train frames (`original/make_eval_split.py`). Test in separate
  areas/passes → mimic THAT structure instead. Never contiguous-arc holdouts.

## 4. Photometric drift → appearance gate (20 min)

- [ ] Per-image exposure statistics (mean luma, gray-world WB) vs capture order/pass
      (forensics notebook cell).
- **Decision:** inter-pass drift clearly super-LSB (> ~2/255 systematic) → enable the
  appearance contingency (per-image embedding, nearest-pose test policy). Sub-LSB /
  noise → SKIP appearance machinery entirely (it lost 4 ways on stable captures:
  bilagrid −7.6, affine −0.9, PPISP ~0, oracle cap +0.09 dB).

## 5. Sky fraction → sky-policy gate (10 min)

- [ ] Estimate sky fraction inside TEST-pose frusta (luma/gradient heuristic on the
      nearest train images is fine).
- **Decision:** > ~10–15% of pixels → activate the sky-policy contingency. Else skip
  (sky dome measured neutral on the towers).

## 6. Scale → partitioning decision (15 min)

- [ ] Scene spatial extent (camera-center bounding box), image count vs H200 memory.
- **Decision:** if a scene trains monolithically on the H200 (likely for anything up to
  a few thousand images at 141 GB), prefer monolithic + big `--cap_max` (the unmeasured
  capacity axis!) and skip cell seams entirely. Partition only when memory or wall-clock
  forces it; then: airspace-aware cells, overlap-expand→trim→merge, seam metric on.

## 7. First training probe (launch before anything else finishes)

- [ ] Smallest scene, default recipe (`configs/recipes/round3_default.args`), 2–5k iters
      — verifies the whole loop (data → train → render → eval) on-site within the hour.
- [ ] Then launch the first real overnight run before 19:00 sealing
      (see `runbooks/onsite_playbook.md`).
