# dataset/ — disk convention (gitignored)

- raw/        original downloads (competition data, benchmarks) — never modified
- processed/  derived layouts the pipeline consumes (undistorted copies, eval splits)

Keep this folder on a large drive (symlink it if needed: `ln -s /path/to/big dataset`).
Rehearsal benchmarks that fit alongside everything else: Mill-19 Building+Rubble
(~21 GB, storage.cmusatyalab.org via the Mega-NeRF repo) and MatrixCity small_city
aerial (31.3 GB, HuggingFace BoDai/MatrixCity). UrbanScene3D pixsfm tgz are POSE-ONLY —
images ship separately.
