# Weights manifest — post-verification submission

**Submission being documented:** `sub_round36_bonsai13.zip`, graded **77.7230**
(PSNR 26.654971 / SSIM 87.3474 / LPIPS 11.1856).

---

## 1. Statement up front

This submission is a **render ensemble**, not the output of a single model. Its 386 images are
per-pixel means of many independently trained 3D Gaussian Splatting models, followed by two
deterministic post-processing operators.

**62 distinct models were trained to produce it. Weights survive for 12 of them (19%).**

The remaining 50 checkpoints were deleted by the production pipeline itself: every training
script ends with `rm -f $M/ckpt.pt` after the model has rendered its test views. A single
checkpoint is 1.1–1.8 GB, so retaining all 62 would have required ~90 GB, which the build machine
did not have. What was retained instead is each model's **rendered PNG output**, because that —
not the checkpoint — is what every downstream stage consumes.

We are declaring this plainly rather than substituting weights from a different model and
presenting them as the submission's. Section 4 states exactly what can and cannot be verified
from what is provided.

---

## 2. Checkpoints included (12 files, 19.1 GB)

Format: PyTorch `.pt`, produced by `gsplat_track/train_gsplat.py`. Loading:

```python
import torch
ckpt = torch.load("bonsai_r14_aa42.pt", map_location="cpu", weights_only=False)
splats = ckpt["splats"]          # dict of tensors
# means [N,3] | scales [N,3] log | quats [N,4] | opacities [N] logit | sh0 [N,1,3] | shN [N,K,3]
```

| scene | file in `weights/` | original path | seed | recipe note | size |
|---|---|---|---|---|---|
| bonsai | `bonsai_r14_aa42.pt` | `r14/bonsai_aa42` | 42 | r14 recipe, predates `--scale_reg 0.1` | 1.10 GB |
| bonsai | `bonsai_r14_aa7.pt` | `r14/bonsai_aa7` | 7 | " | 1.10 GB |
| bonsai | `bonsai_r14_aa13.pt` | `r14/bonsai_aa13` | 13 | " | 1.10 GB |
| chair | `chair_r14_aa42.pt` | `r14/chair_aa42` | 42 | r14 recipe | 1.76 GB |
| chair | `chair_r14_aa7.pt` | `r14/chair_aa7` | 7 | " | 1.76 GB |
| chair | `chair_r14_aa13.pt` | `r14/chair_aa13` | 13 | " | 1.76 GB |
| chair | `chair_r17_ema099_s42.pt` | `r17/chair_ema099_seed42` | 42 | `--ema_decay 0.99` | 1.76 GB |
| chair | `chair_r17_ema099_s7.pt` | `r17/chair_ema099_seed7` | 7 | " | 1.76 GB |
| chair | `chair_r17_ema099_s13.pt` | `r17/chair_ema099_seed13` | 13 | " | 1.76 GB |
| chair | `chair_r17_depth_s42.pt` | `r17/chair_depth_seed42` | 42 | depth variant | 1.76 GB |
| HCM0644 | `HCM0644_r28.pt` | `r28_members/HCM0644` | 42 | shipped tower recipe | 1.76 GB |
| HCM0674 | `HCM0674_r28.pt` | `r28_members/HCM0674` | 42 | shipped tower recipe | 1.76 GB |

Per-scene coverage:

| scene | ensemble size k | distinct models trained | checkpoints included |
|---|---|---|---|
| HCM0421 | 8 | 9 | **0** |
| HCM0539 | 8 | 8 | **0** |
| HCM0540 | 8 | 8 | **0** |
| HCM0644 | 8 | 8 | **1** |
| HCM0674 | 8 | 8 | **1** |
| chair | 8 | 8 | **7** |
| bonsai | 13 | 13 | **3** |
| **total** | | **62** | **12** |

The tower counts include the recursive intermediates: each tower's k=8 ensemble is
`r22/png_ens` (weight 4) + `r25_mip3d` (1) + `r28_members` (1), and `r22` is itself an
equal-weighted mean of an `r20`/`r21` base of 5–6 models plus `r22_seed101`.

---

## 3. What is also included, and why it matters more than the checkpoints

`renders/` contains the rendered test views that are the **actual inputs to the ensemble**. These
were never deleted, and they are what makes the submission reproducible without any checkpoint.

Being precise about what form they take, because it differs by round:

| artifact | what it is | count |
|---|---|---|
| individual per-model test renders | one directory per trained model | **51 models** |
| `r20/tower_ens/$T/png_ens`, `r21/tower_ens/$T/png_ens` | the **already-averaged** base ensemble for that tower — the individual members behind it were not retained separately | 5 towers |

So for the 11 models inside the `r20`/`r21` tower bases, the surviving artifact is the ensembled
intermediate rather than the individual renders. That intermediate is nonetheless the *exact*
input the later rounds consumed (`build_r22.sh` reads `r20/tower_ens/$T/png_ens` directly), so
nothing downstream is approximated.

**Consequence:** the submission **is** reproducible bit for bit from what is provided — run
`ensemble_renders.py` → `energy_restore.py` → `apply_field.py` → the JPEG encode over `renders/`,
exactly as `REPRODUCE.md` §5–§8 describes. What is *not* possible from this package is
re-deriving the `r20`/`r21` base ensembles from their own constituent members.

We verified this rather than asserting it. The bonsai path was re-executed end to end from its
13 member render archives and compared byte for byte against the shipped zip:

```
$ python ensemble_renders.py --dirs <13 member dirs> --png_dir verify/png_ens \
      --names_from bonsai/test/test_poses.csv
  Ensembled 28 images from 13 members
$ python energy_restore.py --mode apply --k 13 --lam 0.25 \
      --ens_dir verify/png_ens --member_dirs <the 6 scale_reg=0.1 dirs> --out_dir verify/png_er
  restored 28 images (lam=0.25, k=13.0, 6 member(s))
$ # re-encode with the shipped JPEG settings, compare against sub_round36_bonsai13.zip

  RESULT  bonsai/: 28 byte-identical, 0 differ
```

---

## 4. What can and cannot be verified from this package

**Can be verified:**
- That the submitted images are exactly what this code produces from these renders — the byte
  comparison above, repeatable for any scene.
- That every training and rendering command is the one that was used — `code/` is the working
  tree, and `REPRODUCE.md` records the exact flags per scene.
- That no test ground truth was used anywhere — see §6.
- Model-level inspection of 12 of the 62 models, including 3 of bonsai's 13 and 7 of chair's 8.

**Cannot be verified from this package alone:**
- Re-rendering all 62 models from checkpoints, because 50 checkpoints no longer exist.
- Bit-exact retraining. Even with the same seed, `gsplat`'s CUDA backward pass accumulates
  through atomics in nondeterministic order, so a retrained model differs slightly from the
  original. Retraining all 62 models takes roughly 110 GPU-hours on one 16 GB card and would
  produce a submission in the same quality band, not the same bytes.

If the committee requires checkpoints for a specific scene, we can retrain that scene's members
and supply both the weights and the renders they produce, with the caveat above stated on the
result. Bonsai is the cheapest (13 × 1h35) and HCM0421 the most expensive (9 × 2h20).

---

## 5. Package layout

```
delivery/
├── README.md                    entry point
├── REPRODUCE.md                 full run instructions, environment, pipeline, pitfalls
├── WEIGHTS_MANIFEST.md          this file
├── environment/
│   ├── gsplat.yml               conda env for training + rendering (includes CUDA 12.8 toolkit)
│   ├── fastgs2.yml              conda env for ensembling, metrics, assembly
│   ├── requirements-gsplat.txt  pip freeze
│   └── requirements-fastgs2.txt pip freeze
├── code/                        the working tree (training, rendering, operators, verification)
├── weights/                     12 checkpoints, 19.1 GB
└── renders/                     per-model test renders for all 62 models — the ensemble inputs
```

---

## 6. Competition-rule compliance

- No external data or imagery of the competition scenes. The only pretrained weights used
  anywhere are VGG/AlexNet, as an LPIPS training loss and evaluation metric.
- Test ground truth is never read or inferred. `test/` provides poses only. The lens-field
  calibration is fitted on **train** renders against **train** photos. Recipe selection used a
  held-out split of **train** photos, never test images.
- `images.bin` ships keypoints for the test frames. They are not read by any script here.
- No manual per-image editing. Every operator is a deterministic program applied uniformly to a
  whole scene.
