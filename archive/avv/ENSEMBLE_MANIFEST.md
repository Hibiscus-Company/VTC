# Shipped ensemble composition — verified by reconstruction, 27/07

Written because a wrong-source-dir bug nearly shipped a silent bonsai regression once already
(`/mnt/d/avv/r14/bonsai_ens/png` matched **0/28** shipped bytes — it was an older ensemble).
Every path below was checked against the bytes actually inside `sub_round27_lanczos.zip`.

## Source dirs that reproduce the shipped JPEGs BYTE-FOR-BYTE
Re-encoding these PNGs at the shipped profile (`quality=100, subsampling=2, optimize, progressive`)
reproduces the shipped file exactly:

| scene | source PNG dir | check |
|---|---|---|
| chair | `/mnt/d/avv/r21/video_ens/chair7/png` | **58/58 byte-identical** |
| bonsai | `/mnt/d/avv/r24/bonsai/png` | **28/28 byte-identical** |
| 5 towers | `/mnt/d/avv/r27/tower_ens/<T>/png` | built this round; HCM0421 encodes at Q=99, rest Q=100 |

Towers are `r25/tower_ens/<T>/png_ens` (the pre-field ensemble) → `apply_field.py --strict`
with `/mnt/d/avv/fields_median/<T>.npy` → JPEG. `apply_field` now defaults to `INTER_LANCZOS4`.

## Member lists (each verified by rebuilding the pixel mean and comparing)

**chair — 7 members, uniform weight, NO field.** Reconstructed mean is `exact=True` against
`r21/video_ens/chair7/png`:
1. `/mnt/d/avv/r14/chair_aa42/test_png`
2. `/mnt/d/avv/r14/chair_aa7/test_png`
3. `/mnt/d/avv/r14/chair_aa13/test_png`
4. `/mnt/d/avv/r17/chair_ema099_seed42/test_png`
5. `/mnt/d/avv/r17/chair_ema099_seed7/test_png`
6. `/mnt/d/avv/r17/chair_ema099_seed13/test_png`
7. `/mnt/d/avv/r17/chair_depth_seed42/test_png`  ← the depth-prior member added in r21

**bonsai — 6 members, uniform weight, NO field.** Reconstructed mean differs from
`r24/bonsai/png` by 0.083/255 mean, i.e. rounding tie-breaks only, not a different member set —
so rebuild with `ensemble_renders.py` (which produced it) rather than a hand-rolled mean:
1. `/mnt/d/avv/r14/bonsai_aa42/test_png`
2. `/mnt/d/avv/r14/bonsai_aa7/test_png`
3. `/mnt/d/avv/r14/bonsai_aa13/test_png`
4. `/mnt/d/avv/r24_bonsai/aa101/test_png`
5. `/mnt/d/avv/r24_bonsai/aa202/test_png`
6. `/mnt/d/avv/r24_bonsai/aa303/test_png`

**towers — 7 members**, composed as `0.8 * (r22 6-member mean) + 0.2 * mip3d`, i.e. mip3d at
w=0.200 and each of the 6 originals at 0.1333. mip3d member: `/mnt/d/avv/r25_mip3d/<T>/test_png`.

## Training recipes (for adding members)
- towers: `--ut --ema_decay 0.999 --iters 60000 --cap_max 8000000 --refine_stop 50000
  --noise_stop 50000 --lpips_from 50000`; render with `--ut_render native`
- chair: `--ema_decay 0.99 --iters 60000 --cap_max 8000000 --refine_stop 50000 --noise_stop 50000
  --lpips_from 30000` (no `--ut`, so gsplat's antialiased rasteriser)
- bonsai: `--iters 30000 --cap_max 5000000 --refine_stop 15000 --noise_stop 8000 --lpips_from 12000`
  (no `--ut`; the low refine/noise stops are the churn-stop that fixed the bonsai collapse)
- add `--mip3d 0.2 --mip3d_every 100` for a mip3d member; it is independent of `--ut` and gets
  baked into the ckpt at save, so the stock renderer needs no change.

Seeds already used: towers 7/13/42/77/101 + mip3d 555; chair 7/13/42; bonsai 7/13/42/101/202/303.
