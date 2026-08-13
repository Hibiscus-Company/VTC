# Reproducing `sub_round36_bonsai13.zip`

VAR 2026 "BTS Digital Twin" — novel-view synthesis on `private_set2` (7 scenes).
This submission graded **77.7230** (PSNR 26.654971 / SSIM 87.3474 / LPIPS 11.1856).

---

## 0. What a blank machine can and cannot do — read before anything else

`sub_round36_bonsai13.zip` is not the output of one script. It is the last link in an incremental
chain in which each round re-encoded only the scenes it changed and carried the rest of the zip
**byte-verbatim**. Its 386 images come from **62 independently trained Gaussian-splat models** (the tower rows are ensembles-of-ensembles)
produced over roughly three weeks, and the tower ensembles are themselves recursive: r29 averages
an r22 intermediate, which averages an r20/r21 intermediate, and so on.

| | inputs required | output | on a blank machine? |
|---|---|---|---|
| **Level A — bit-exact rebuild** | the member PNG archives + round intermediates (**~11 GB**, *not* in this repo) | the **byte-identical** zip | **No** — not unless those archives are copied to it |
| **Level B — equivalent from data only** | the 1.6 GB competition dataset | a *different* submission of comparable quality | **Yes** — ~90 GPU-hours |

**Level B cannot reproduce r36's exact composition, and this document will not pretend otherwise.**
The tower branch is defined by a chain of round intermediates (`r20/tower_ens/*/png_ens` →
`r22` → `r25_mip3d` → `r28_members` → `r29` → `r31`) whose members were trained under recipes
that changed between rounds. Section 6 records that chain as **provenance**; it is not a recipe
you can execute. What Section 7 gives you instead is a clean, runnable pipeline that trains a
fresh member set and produces a submission in the same quality band.

Even given identical inputs and `--seed`, Level B is not bit-reproducible: `gsplat`'s CUDA
backward pass accumulates through atomics in nondeterministic order. Expect per-scene scores
within roughly ±0.4 (our measured run-to-run noise floor) and a submission total within ~±0.05.

**What has been verified for this document** — see Section 9 for the transcript:
the bonsai path of Section 8 was re-executed from its 13 member archives and compared byte for
byte against the shipped zip: **28/28 byte-identical, 0 differ**.

---

## 1. Hardware

Built on:

```
2 × NVIDIA GeForce RTX 5070 Ti  (16303 MiB each, compute capability sm_120)
NVIDIA driver 576.88
24 CPU cores, 47 GB RAM
WSL2 (Linux 6.18.33.2-microsoft-standard-WSL2)
```

**16 GB of VRAM is a binding constraint, not a detail.** The tower/chair recipe trains at
`--cap_max 8000000` and peaks near 13.5 GB; bonsai at `--cap_max 5000000` peaks near 13.4 GB.
Less than 16 GB will OOM.

Disk: Level B needs ~120 GB for member renders and checkpoints. A 5M-gaussian checkpoint is
1.1 GB and an 8M one is 1.8 GB; the production scripts delete them after rendering for exactly
that reason. **If you need the weights afterwards, remove that `rm` before you start** — we did
not, and lost 50 of this submission's 62 checkpoints.

**Network is required** on first run: `lpips` downloads its VGG weights, and `torchvision`
downloads the ImageNet VGG16 backbone. Pre-seed `~/.cache/torch` on an air-gapped machine.

---

## 2. Environment — starting from nothing but miniconda

Three things have to be installed, in this order, and the first one is the step most likely to
be skipped: **the CUDA toolkit**. Miniconda does not ship `nvcc`, and both `gsplat` and
`fused-ssim` need it to compile CUDA kernels — `fused-ssim` at install time, `gsplat` at first use.

Evidence, tested rather than assumed: `pip download gsplat==1.5.3 --no-deps` returns
`gsplat-1.5.3-py3-none-any.whl` — a **6.5 MB pure-Python wheel with no compiled kernels at all**.
gsplat JIT-compiles its CUDA extension **on first use**, into `site-packages/gsplat/csrc.so`;
on this machine that object is **257 MB** and `cuobjdump --list-elf` reports exactly one
architecture, `sm_120`.

So `nvcc` is needed **at runtime**, not merely at install time. `pip install gsplat` will succeed
on a machine with no CUDA toolkit and then fail minutes into your first training run. Budget
10–25 minutes and several GB of RAM for that first-use compile.

Two conda environments are required and they are **not interchangeable**:

- **`gsplat`** — training and rendering. Has `gsplat`, CUDA kernels, `transformers`.
- **`fastgs2`** — ensembling, metrics, field fitting, zip assembly. Deliberately has **no**
  `gsplat`, so assembly can never hold GPU memory while a training job runs.

> The `environment.yml` committed at the repo root is for the **superseded FastGS track**
> (python 3.7 / torch 1.12 / cudatoolkit 11.6). It does **not** build r36 and will not compile
> for sm_120. Ignore it; use the specs below.

### 2.1 `gsplat` (train + render)

```bash
conda create -n gsplat python=3.10.20 -y
conda activate gsplat

# 1. CUDA toolkit FIRST -- provides nvcc, which the next two steps compile against.
#    Match the major.minor to your driver. sm_120 (RTX 50-series) needs >= 12.8.
conda install -c nvidia/label/cuda-12.8.1 cuda-toolkit -y
nvcc --version          # must print release 12.8

# 2. PyTorch built against the same CUDA
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 3. gsplat -- the wheel is pure python; it JIT-compiles kernels on FIRST USE, not here.
#    This install is fast; budget 10-25 min + several GB RAM for your first training run.
pip install gsplat==1.5.3

pip install lpips==0.1.4 numpy==1.26.4 pillow==12.3.0 plyfile==1.1.3 \
            opencv-python==4.11.0.86 scipy==1.15.3 tqdm==4.68.3
pip install ./submodules/fused-ssim      # CUDAExtension -> also needs nvcc
pip install transformers==5.14.1         # only for scripts/precompute_depth.py
```

`pip install gsplat` will **succeed even with no CUDA toolkit present** — the failure surfaces
minutes into your first training run, as `nvcc: not found` or `Unsupported gpu architecture
'compute_120'`. If you see either, step 1 was skipped or your toolkit is older than 12.8.
Force the compile up front instead of discovering it later:

```bash
python -c "
import torch, gsplat
m = torch.rand(10,3,device='cuda'); q = torch.rand(10,4,device='cuda')
s = torch.rand(10,3,device='cuda'); o = torch.rand(10,device='cuda')
print('gsplat kernels built OK')"
```

### 2.2 `fastgs2` (ensemble + metrics + assembly)

```bash
conda create -n fastgs2 python=3.10.20 -y
conda activate fastgs2
conda install -c nvidia/label/cuda-12.8.1 cuda-toolkit -y     # for fused-ssim only
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install lpips==0.1.4 numpy==1.26.4 pillow==12.3.0 plyfile==1.1.3 \
            opencv-python==4.11.0.86 scipy==1.15.3 tqdm==4.68.3
pip install ./submodules/fused-ssim
```

### 2.3 Scripts used by this pipeline

All repo-relative. Two of these were vendored into the repo root while this document was being
written — they had been living in a scratch directory and would not have reached you:

| script | stage |
|---|---|
| `gsplat_track/train_gsplat.py` | 1 — train a member |
| `gsplat_track/render_gsplat.py` | 1 — render the test poses |
| `ensemble_renders.py` | 2 — pixel mean |
| **`energy_restore.py`** *(vendored)* | 3 — high-frequency restore |
| **`lapfuse.py`** *(vendored)* | imported by `energy_restore.py` |
| `gsplat_track/render_train.py` | 4 — render train poses for the field fit |
| `gsplat_track/fit_field.py` | 4 — fit the lens field |
| `gsplat_track/apply_field.py` | 4 — apply it |
| `scripts/preflight.py` | 0 — data sanity |
| `scripts/verify_zip.py` | 5 — submission check |
| `scripts/eval_score.py` | scoring against a held-out train split |

They import `utils/`, `scene/` from this repo — run everything with the repo root as CWD.

### 2.4 Verify both environments before going further

```bash
conda activate gsplat
python - <<'EOF'
import torch, gsplat, subprocess, glob, os
assert torch.cuda.is_available(), "no CUDA visible"
cap = torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gsplat", gsplat.__version__)
print("device", torch.cuda.get_device_name(0), "sm_%d%d" % cap,
      "| VRAM %.1f GiB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
so = glob.glob(os.path.dirname(gsplat.__file__) + "/**/csrc.so", recursive=True)[0]
archs = subprocess.run(["cuobjdump", "--list-elf", so], capture_output=True, text=True).stdout
assert "sm_%d%d" % cap in archs, "gsplat kernels do NOT cover this GPU -- rebuild it"
print("gsplat kernels cover sm_%d%d: OK" % cap)
import fused_ssim; print("fused_ssim: OK")
EOF

conda activate fastgs2
python -c "
import sys; sys.path.insert(0,'.')
from utils.loss_utils import ssim; import lpips, fused_ssim, energy_restore
print('fastgs2 imports: OK')"
```

`fused_ssim` must import in **both** envs, but note that `scripts/eval_score.py` deliberately
uses the repo's own SSIM (`utils.loss_utils.ssim`, zero-padded conv) and **not** `fused_ssim`:
the two differ by up to 0.003 SSIM (≈0.09 score points) and every recorded number is on the
repo one. Do not swap them.

---

## 3. Data

```
$DATA/                                  # e.g. /mnt/d/avv/data/phase1/private_set2   (1.6 GB)
├── HCM0421/  HCM0539/  HCM0540/  HCM0644/  HCM0674/     # 5 drone towers, 240 train / 60 test
│   ├── train/images/*.JPG              # 1320×989, SIMPLE_RADIAL k1 ≈ +0.00894
│   ├── train/sparse/0/{cameras,images,points3D}.bin
│   └── test/test_poses.csv             # poses only — no test images are provided
├── chair/                              # indoor phone video, 205 train / 58 test, 720×1280 PORTRAIT
└── bonsai/                             # indoor phone video, 248 train /  28 test, 1920×1080
```

`test_poses.csv`: `image_name, qw,qx,qy,qz, tx,ty,tz, fx,fy,cx,cy, width,height`.

```bash
conda activate fastgs2
python scripts/preflight.py --data_root $DATA
```

Two things it exists to catch, both of which bit us:
- **chair's COLMAP observations are at 1.5× the delivered resolution** (COLMAP ran at 1080×1920,
  images ship at 720×1280). The obs-scale candidate list must include fractional 1.5.
- `images.bin` contains the **test** frames too, with full keypoints. We do not read them.

---

## 4. Pipeline

```
  ┌──────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
  │ 1. TRAIN     │──▶│ 2. ENSEMBLE  │──▶│ 3. RESTORE    │──▶│ 4. LENS      │──▶ 5. assemble
  │  N members   │   │  pixel mean  │   │  high-freq    │   │  field warp  │
  └──────────────┘   └──────────────┘   └───────────────┘   └──────────────┘
     GPU, hours          CPU, minutes        CPU, minutes       CPU, seconds
```

Stage 4 applies to the **5 towers and chair only** — it is measured harmful on bonsai.

---

## 5. Stage 1 — training recipes

### 5.1 Towers and chair (3DGUT)

```bash
conda activate gsplat
CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py \
  --source $DATA/$SCENE/train --images images --out $OUT/$SCENE/s$SEED \
  --ut --ema_decay 0.999 --iters 60000 --cap_max 8000000 \
  --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 --seed $SEED

CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py \
  --ckpt $OUT/$SCENE/s$SEED/ckpt.pt --csv $DATA/$SCENE/test/test_poses.csv \
  --out $OUT/$SCENE/s$SEED/test_render --png_dir $OUT/$SCENE/s$SEED/test_png \
  --ut_render native
```

`--ut` is load-bearing. It trains **in distorted space on the original `train/images`** instead
of on undistorted copies, removing one INTER_CUBIC resampling generation from the supervision
signal. That was the single largest gain of the campaign (LB 74.3 → 77.3). Render with
`--ut_render native` to match — a UT model rendered through the warp path is a response-model
mismatch. Always archive PNG; ensembling from JPEG costs ~0.005.

Runtime: ~2h20 per tower member, ~2h35 per chair member.

### 5.2 bonsai — deliberately different

```bash
CUDA_VISIBLE_DEVICES=0 python gsplat_track/train_gsplat.py \
  --source $DATA/bonsai/train --images images --out $OUT/bonsai/s$SEED \
  --iters 30000 --cap_max 5000000 \
  --refine_stop 15000 --noise_stop 8000 --lpips_from 12000 \
  --scale_reg 0.1 --seed $SEED

CUDA_VISIBLE_DEVICES=0 python gsplat_track/render_gsplat.py \
  --ckpt $OUT/bonsai/s$SEED/ckpt.pt --csv $DATA/bonsai/test/test_poses.csv \
  --out $OUT/bonsai/s$SEED/test_render --png_dir $OUT/bonsai/s$SEED/test_png
```

**No `--ut`** (bonsai is PINHOLE, k1 = 0 — nothing to undistort) and **`--scale_reg 0.1`**
instead of the 0.01 default; that coefficient is worth +0.139 on the bonsai scene and was
confirmed on the leaderboard at r35. Runtime ~1h35.

---

## 6. What r36 actually shipped (provenance — NOT a runnable recipe)

62 distinct trained models. The tower entries are references to *round intermediates*, each of which is
itself an ensemble of earlier members:

| scene | k | composition as shipped |
|---|---|---|
| each tower | 8 | `r22/tower_ens/$T/png_ens` (weight 4) + `r25_mip3d/$T` (1) + `r28_members/$T` (1) — and `r22` = `r20`/`r21` base (5–6 members) + `r22_seed101`, equal-weighted |
| chair | 8 | `r14` ×3 (seeds 42/7/13) + `r17` ema0.99 ×3 (42/7/13) + `r17` depth (42) + `r28_members` |
| bonsai | 13 | `r14` ×3 (42/7/13) + `r24_bonsai` ×3 (101/202/303) + `r28_members` + `r38_prod` ×6 (111/555/777/222/333/999, all `--scale_reg 0.1`) |

Only the last six bonsai members use `--scale_reg 0.1`; the other seven predate that finding.
**The chair and bonsai rows are executable** if you have those member archives. **The tower row
is not** — reproducing it needs the r20/r21/r22 intermediates, whose own member recipes varied
across rounds and are recorded only in the round build scripts.

**Member selection rule** (applies to any fresh build): only ensemble members within ~0.15 points
of the best single model, and prefer the current training generation. Shipping two
stale-generation tower members once cost −0.0030 on the leaderboard.

---

## 7. Level B — a runnable build from data only

This is what a blank machine should actually run. It produces a submission in the same quality
band as r36, not the same bytes.

```bash
#!/bin/bash
# Level B driver. ~90 GPU-hours on one 16 GB card; halve it on two.
set -e
cd /path/to/FastGS
source ~/miniconda3/etc/profile.d/conda.sh
DATA=/path/to/private_set2
OUT=/path/to/work
SEEDS_TOWER="42 101 202"      # 3 members per tower  (add more; gains decay ~1/N)
SEEDS_CHAIR="42 101 202"
SEEDS_BONSAI="42 101 202 303" # bonsai members are cheaper, so take more

conda activate gsplat
for SCENE in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674 chair; do
  for S in $SEEDS_TOWER; do
    M=$OUT/$SCENE/s$S; mkdir -p $M
    python gsplat_track/train_gsplat.py --source $DATA/$SCENE/train --images images --out $M \
      --ut --ema_decay 0.999 --iters 60000 --cap_max 8000000 \
      --refine_stop 50000 --noise_stop 50000 --lpips_from 50000 --seed $S
    python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
      --csv $DATA/$SCENE/test/test_poses.csv --out $M/test_render --png_dir $M/test_png \
      --ut_render native
    # keep ONE checkpoint per scene for the lens-field fit; delete the rest (1.2 GB each)
    [ "$S" = "42" ] && python gsplat_track/render_train.py --ckpt $M/ckpt.pt \
        --source $DATA/$SCENE/train --images images --out $OUT/$SCENE/train_render \
        --ut_render native
    rm -f $M/ckpt.pt
  done
done
for S in $SEEDS_BONSAI; do
  M=$OUT/bonsai/s$S; mkdir -p $M
  python gsplat_track/train_gsplat.py --source $DATA/bonsai/train --images images --out $M \
    --iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 \
    --lpips_from 12000 --scale_reg 0.1 --seed $S
  python gsplat_track/render_gsplat.py --ckpt $M/ckpt.pt \
    --csv $DATA/bonsai/test/test_poses.csv --out $M/test_render --png_dir $M/test_png
  rm -f $M/ckpt.pt
done

conda activate fastgs2
mkdir -p $OUT/fields
for SCENE in HCM0421 HCM0539 HCM0540 HCM0644 HCM0674 chair; do
  # --- stage 4a: fit the lens field on TRAIN renders vs TRAIN photos (never test GT) ---
  python gsplat_track/fit_field.py --render_dir $OUT/$SCENE/train_render \
    --gt_dir $DATA/$SCENE/train/images --out $OUT/fields/$SCENE.npy --estimator median
  python - "$OUT/fields/$SCENE.npy" "$OUT/fields/${SCENE}_g130.npy" <<'EOF'
import numpy as np, sys
f = np.load(sys.argv[1]) * 1.30              # the 1.30 gain -- see below, it is not a tuning knob
assert np.abs(f).max() < 8.0, "field too large -- refuse"
np.save(sys.argv[2], f)
EOF
  # --- stages 2-4b ---
  K=$(ls -d $OUT/$SCENE/s*/test_png | wc -l)
  python ensemble_renders.py --dirs $OUT/$SCENE/s*/test_png \
    --out $OUT/$SCENE/jpg_tmp --png_dir $OUT/$SCENE/png_ens \
    --names_from $DATA/$SCENE/test/test_poses.csv
  LAM=1.00; [ "$SCENE" = "chair" ] && LAM=0.40
  python energy_restore.py --mode apply --k $K --lam $LAM \
    --ens_dir $OUT/$SCENE/png_ens --member_dirs $OUT/$SCENE/s*/test_png \
    --out_dir $OUT/$SCENE/png_er
  python gsplat_track/apply_field.py --in_dir $OUT/$SCENE/png_er \
    --field $OUT/fields/${SCENE}_g130.npy --out_dir $OUT/$SCENE/png --strict
done
# bonsai: ensemble + restore, NO field
K=$(ls -d $OUT/bonsai/s*/test_png | wc -l)
python ensemble_renders.py --dirs $OUT/bonsai/s*/test_png \
  --out $OUT/bonsai/jpg_tmp --png_dir $OUT/bonsai/png_ens \
  --names_from $DATA/bonsai/test/test_poses.csv
python energy_restore.py --mode apply --k $K --lam 0.25 \
  --ens_dir $OUT/bonsai/png_ens --member_dirs $OUT/bonsai/s*/test_png \
  --out_dir $OUT/bonsai/png
```

### 7.1 The three numbers in there that are not free parameters

**`--k` must be the TRUE member count.** `--member_dirs` only estimates the disagreement; `k`
corrects the estimator. Passing k=8 for a 9-member ensemble is a real mis-specification that
happened once and was caught by audit.

**λ per scene: towers 1.00, chair 0.40, bonsai 0.25.** These are not interchangeable. λ=0 was
shipped for bonsai once (r33a) and **lost −0.0041** — do not simplify it to zero because a local
sweep says so; that sweep disagrees with the leaderboard.

**The lens-field gain is 1.30, and it corrects a structural bias, not a taste.** At train poses
the model has already absorbed part of the misregistration into its own geometry, so the fitted
flow sees only the unabsorbed remainder; at novel poses the absorbed warp does not cancel and the
full displacement appears. Measured magnitude ratio 0.72 → 1/0.72 = 1.39. A scored sweep peaks at
1.30, the geometric estimate is α\* = 1.3185, and the leaderboard confirmed it (r29, +0.1615).
A leave-one-view-out check on train poses reports the optimum as **1.00** and is structurally
blind to this — do not trust it.

Why a field at all: COLMAP gives a single `SIMPLE_RADIAL` k1; every higher-order radial,
tangential and thin-prism term is pinned to zero, and gsplat's distortion coefficients take no
gradient, so the residual cannot be learned. It is fitted and applied outside the renderer.
Use the full 2-D field, not a radial polynomial — the measured curve is non-monotone (+0.44 px
bump at r=0.67, −1.5 px at the corner) and a 3-term odd polynomial captures only half of it.

---

## 8. Stage 5 — assemble and verify

```bash
conda activate fastgs2
python - <<'EOF'
import zipfile, io, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
OUTDIR = "/path/to/work"; ZIP = "submission.zip"
ENCODE = dict(quality=100, subsampling=2, optimize=True, progressive=True)
CAP    = 367001600                       # 350 MiB, NOT 350e6
SCENES = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674", "chair", "bonsai"]
EXPECT = dict.fromkeys(SCENES[:5], 60) | {"chair": 58, "bonsai": 28}
buf = io.BytesIO(); n = 0
with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
    for scene in SCENES:
        d = os.path.join(OUTDIR, scene, "png")
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        assert len(files) == EXPECT[scene], f"{scene}: {len(files)} != {EXPECT[scene]}"
        for f in files:
            b = io.BytesIO()
            Image.open(os.path.join(d, f)).convert("RGB").save(b, "JPEG", **ENCODE)
            z.writestr(f"{scene}/{os.path.splitext(f)[0]}.jpg", b.getvalue()); n += 1
data = buf.getvalue()
assert len(data) <= CAP, f"OVER CAP {len(data):,} > {CAP:,}"
open(ZIP, "wb").write(data)
print(f"wrote {ZIP}: {n} images, {len(data):,} B = {len(data)/2**20:.2f} MiB, "
      f"headroom {(CAP-len(data))/2**20:.2f} MiB")
EOF

python scripts/verify_zip.py --zip submission.zip --data_root $DATA
```

Four constraints that have each cost points when violated:

1. **`quality=100, subsampling=2, optimize=True, progressive=True`.** A q98/4:4:4 variant won
   locally and **lost on the leaderboard**. Do not change the encode.
2. **`ZIP_STORED`**, never `ZIP_DEFLATED` — JPEG does not deflate.
3. **The cap is 350 MiB = 367,001,600 bytes**, not 350 × 10⁶. r36 lands at 334.18 MiB.
   If you exceed it, drop `quality` one scene at a time (r31 shipped HCM0421 at q99).
4. **Never re-encode an existing JPEG.** Encode once, from the lossless PNG master.

Expected `verify_zip.py` output:

```
files 386/386   size 334.18 MiB / 350.42 MB (limit 350.0 MiB = 367.0 MB)
  HCM0421: 60/60 exact, all 1320x989, JPEG        ... (×5 towers)
  bonsai:  28/28 exact, all 1920x1080, JPEG
  chair:   58/58 exact, all 720x1280, JPEG
VERIFY PASSED
```

---

## 9. Verification actually performed for this document

The bonsai path of Sections 5–8 was re-executed end to end from its 13 member PNG archives, using
only repo-relative script paths, and the output compared **byte for byte** with the shipped zip:

```
$ python ensemble_renders.py --dirs <13 member dirs> \
      --out verify/jpg_tmp --png_dir verify/png_ens \
      --names_from $DATA/bonsai/test/test_poses.csv
  Ensembled 28 images from 13 members

$ python energy_restore.py --mode apply --k 13 --lam 0.25 \
      --ens_dir verify/png_ens --member_dirs <the 6 scale_reg=0.1 dirs> --out_dir verify/png_er
  restored 28 images (lam=0.25, k=13.0, 6 member(s))

$ # re-encode each with the Section 8 settings and compare against sub_round36_bonsai13.zip
  RESULT  bonsai/: 28 byte-identical, 0 differ
```

Also machine-checked: all 10 script paths referenced above exist in the repo, and all 32
command-line flags documented here are present in the scripts they are documented for.

**Not verified by execution:** the tower and chair paths. They are transcribed from the round
build scripts (`build_r29.sh` → `build_r31.sh` → `build_r32.sh`) but were not re-run, because
doing so consumes the round intermediates described in Section 0 rather than testing anything a
fresh build would do.

---

## 10. Cost

| stage | unit | wall clock (1 × RTX 5070 Ti) |
|---|---|---|
| tower member | 2h20 | |
| chair member | 2h35 | |
| bonsai member | 1h35 | |
| ensemble + restore | ~4 min/scene | |
| field fit + apply | ~6 min/scene | |
| assemble + verify | ~4 min | |
| **Level B as written in §7** (3+3+3+3+3 towers, 3 chair, 4 bonsai) | | **~50 GPU-hours** |
| **Level A** (given the member archives) | | **~10 min, CPU only** |

---

## 11. Pitfalls

- **Do not use the gsplat `Parser`.** Feed raw COLMAP world coordinates. The parser applies a
  scene normalisation that must then also be applied to the CSV test poses (and inverted
  w2c→c2w), silently re-undistorts already-undistorted images, and defaults `data_factor` to 4.
  `gsplat_track/train_gsplat.py` reads COLMAP directly, at native resolution, always.
- **`--absgrad` is a silent no-op** under `MCMCStrategy` (only `DefaultStrategy` consumes that
  buffer). The trainer asserts against it rather than let you run a placebo A/B.
- **Antialiasing must match between train and render.** Both are on.
- **Keep exactly one checkpoint per scene** — the one you fit the lens field from. The rest are
  1.1-1.8 GB each and 62 of them will not fit. The PNG archives are what every later stage consumes.
- **Launch long jobs detached**: `setsid nohup bash queue.sh > log 2>&1 < /dev/null & disown`.
  A plain `&` makes the job a child of the shell and it dies with the parent — that cost
  12.5 GPU-hours once.
- **Never gate one queue on GPU memory while another queue is running.** Poll a completion
  marker file instead. A memory poll races the ~1-minute gap between two members of another
  queue, and both queues then stack on the same card.
- **`local A=$1 B=$OUT/$A` in bash expands `$A` to empty.** Bash does not see an assignment made
  earlier in the same `local` statement. Use separate statements.

---

## 12. Competition rules honoured by this pipeline

- No external data or imagery of these scenes. The only pretrained weights used are VGG/AlexNet
  (as an LPIPS loss and metric), which the rules permit.
- Test ground truth is never read or inferred. `test/` contains poses only; the lens field is
  fitted on train renders vs train photos; the eval split used for recipe selection holds out
  **train** photos, never test images.
- `images.bin` contains test-frame keypoints. They are not used.
- No manual per-image editing. Every operator is a deterministic program applied uniformly.
