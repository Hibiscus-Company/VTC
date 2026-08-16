#!/usr/bin/env python
"""PREFLIGHT: validate a dataset BEFORE spending a single GPU-second.

Audit round 10: "a 30-second preflight is the difference between 'the overnight run
failed at 06:00 with a clear message' and 'the overnight run burned 18 GPU-hours and
died at compose'."

Every check here corresponds to a way the pipeline hard-fails or, worse, SILENTLY
degrades on data we have not seen:

  sparse/0 present            - obvious, but fails late otherwise
  exactly ONE camera          - load_scene asserts len(cams)==1; a multi-camera scene
                                dies 40 min into a train
  camera model supported      - OPENCV/FULL_OPENCV/FISHEYE assert out. RADIAL is
                                ACCEPTED but only k1 is used -> a SILENT partial-model
                                error, which is worse than the assert
  |k1| < 0.2                  - fov_mask's Newton inversion has NO ROOT beyond this
                                (rd_max > g(r_fold)); it returns finite garbage with no
                                NaN, producing a ~100%-unsupervised mask that silently
                                zeroes a whole member family
  image dir non-empty         - a renamed image dir kills every scene
  test CSV single camera      - render_gsplat asserts one camera across rows
  W/H agree (cameras.bin/CSV) - a mismatch scores 0 while passing every name check

Exit non-zero on any failure. Warnings do not block.
"""
import argparse, csv, os, sys
import numpy as np

from colmap_loader import read_intrinsics_binary, read_extrinsics_binary

SUPPORTED = ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_PINHOLE", "PINHOLE")
K1_SAFE = 0.20      # beyond this fov_mask's Newton has no root -- see fov_mask guards


def check_scene(root, scene, images_dirname="images"):
    errs, warns, info = [], [], {}
    S = os.path.join(root, scene, "train")
    sparse = os.path.join(S, "sparse", "0")

    if not os.path.isdir(sparse):
        return [f"no sparse/0 at {sparse}"], [], info
    for f in ("cameras.bin", "images.bin", "points3D.bin"):
        if not os.path.exists(os.path.join(sparse, f)):
            errs.append(f"missing {f}")
    if errs:
        return errs, warns, info

    cams = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    if len(cams) != 1:
        errs.append(f"{len(cams)} cameras (load_scene asserts exactly 1)")
        return errs, warns, info
    cam = list(cams.values())[0]
    p = list(cam.params)
    info["model"] = cam.model
    info["wh"] = (cam.width, cam.height)

    if cam.model not in SUPPORTED:
        errs.append(f"camera model {cam.model} unsupported (need one of {SUPPORTED})")
        return errs, warns, info
    if cam.model == "RADIAL" and len(p) > 4 and abs(float(p[4])) > 1e-9:
        warns.append(f"RADIAL k2={float(p[4]):+.5f} is IGNORED by the pipeline "
                     "(only k1 is used) -- silent partial-model error")

    k1 = float(p[3]) if cam.model in ("SIMPLE_RADIAL", "RADIAL") else 0.0
    info["k1"] = k1
    info["path"] = "warp" if k1 < 0 else "native"
    f, cx, cy = float(p[0]), float(p[1]), float(p[2])
    W, H = cam.width, cam.height

    # fov_mask Newton: for k1<0 the cubic rd = ru(1+k1 ru^2) folds at r_fold, and beyond
    # g(r_fold) NO undistorted radius maps to the frame corner at all.
    if k1 < 0:
        corners = [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]
        rd_max = max(np.hypot((x - cx) / f, (y - cy) / f) for x, y in corners)
        r_fold = 1.0 / np.sqrt(-3.0 * k1)
        g_fold = r_fold * (1 + k1 * r_fold ** 2)
        info["rd_max"] = rd_max
        info["fold"] = r_fold
        if rd_max >= g_fold:
            errs.append(f"fov_mask has NO ROOT: rd_max {rd_max:.3f} >= g(r_fold) "
                        f"{g_fold:.3f} at k1={k1:+.4f}. Newton would return finite "
                        "GARBAGE and silently zero a member family.")
        elif abs(k1) > K1_SAFE:
            warns.append(f"|k1|={abs(k1):.3f} > {K1_SAFE}: near the fov_mask limit")

    img_dir = os.path.join(S, images_dirname)
    if not os.path.isdir(img_dir):
        errs.append(f"no image dir {img_dir}")
        return errs, warns, info
    imgs = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    on_disk = [v for v in imgs.values()
               if os.path.exists(os.path.join(img_dir, v.name))]
    info["n_train"] = len(on_disk)
    # load_scene() filters missing files SILENTLY. A mostly-empty image dir (12 of 240)
    # would sail through and train to convergence on 12 views (audit r11). Gate on the
    # ABSOLUTE count: the sparse legitimately holds train + test + dropped poses, so
    # "only half of images.bin exists on disk" is EXPECTED here, not a defect.
    if len(on_disk) < 50:
        errs.append(f"only {len(on_disk)} usable train images (need >=50); "
                    f"{len(imgs)} entries in images.bin")
    elif len(on_disk) < 150:
        warns.append(f"{len(on_disk)} train images -- field fit needs stride 1")

    # points3D: MCMC seeds from these; an empty cloud starts from nothing
    try:
        from colmap_loader import read_points3D_binary
        xyz, _, _ = read_points3D_binary(os.path.join(sparse, "points3D.bin"))
        info["n_pts"] = len(xyz)
        if len(xyz) < 1000:
            errs.append(f"only {len(xyz)} points3D -- degenerate sparse cloud")
    except Exception as e:
        errs.append(f"points3D.bin unreadable: {e}")

    # THE convention check (audit r11): reproject this scene's own COLMAP 2D observations.
    # If a new release changes the pose convention, EVERY render is silently garbage and
    # nothing else here would catch it. Seconds, CPU-only.
    #
    # TWO subtleties, both of which I got wrong on the first attempt and which the existing
    # geom_check() in train_gsplat.py already encodes:
    #   - point3D_ids are COLMAP IDs, NOT row indices into the xyz array.
    #   - images.bin stores xys at the ORIGINAL capture resolution (5280x3956) while
    #     cameras.bin is the delivered /4 size (1320x989). Compare obs/s, fitting the
    #     integer downscale s per view, or you measure ~2500px of pure unit mismatch.
    try:
        import numpy as _np
        from colmap_loader import qvec2rotmat
        xyz3, _rgb, _e = read_points3D_binary(os.path.join(sparse, "points3D.bin"))
        # read_points3D_binary DISCARDS the COLMAP point IDs and stores xyz in FILE order,
        # so rebuild the id->row map by re-reading in the same order (NOT sorted -- sorting
        # would scramble the mapping). Same approach as train_gsplat.py's geom_check caller.
        import struct
        pt_ids = []
        with open(os.path.join(sparse, "points3D.bin"), "rb") as fh:
            n_pts = struct.unpack("<Q", fh.read(8))[0]
            for _ in range(n_pts):
                d = struct.unpack("<QdddBBBd", fh.read(43))
                pt_ids.append(d[0])
                tl = struct.unpack("<Q", fh.read(8))[0]
                fh.read(8 * tl)
        pt_index = {pid: row for row, pid in enumerate(pt_ids)}
        errs_px, scales = [], []
        for v in on_disk[:: max(1, len(on_disk) // 3)][:3]:
            valid = v.point3D_ids >= 0
            ids, xys = v.point3D_ids[valid], v.xys[valid]
            rows = _np.array([pt_index.get(i, -1) for i in ids])
            keep = rows >= 0
            if keep.sum() < 20:
                continue
            w2c = _np.eye(4)
            w2c[:3, :3] = qvec2rotmat(v.qvec)
            w2c[:3, 3] = v.tvec
            pc = (w2c[:3, :3] @ xyz3[rows[keep]].T + w2c[:3, 3:4]).T
            front = pc[:, 2] > 1e-3
            # APPLY the SIMPLE_RADIAL forward model. A pure pinhole projection makes a
            # high-|k1| scene look like a convention failure: at k1=-0.115 the radial
            # displacement is ~13x that of a +0.009 scene, giving ~12px of residual that
            # is real distortion, not a bug.
            xn = pc[front, :2] / pc[front, 2:3]
            r2 = (xn ** 2).sum(axis=1, keepdims=True)
            proj = f * xn * (1.0 + k1 * r2) + _np.array([cx, cy])
            obs = xys[keep][front]
            # obs may be at the ORIGINAL capture resolution while cameras.bin is the
            # delivered one; the ratio is not always integer (set2 chair: 1080p->720p = 1.5)
            s, e = min(((c, float(_np.median(_np.linalg.norm(obs / c - proj, axis=1))))
                        for c in (1, 1.5, 2, 3, 4, 6, 8)), key=lambda t: t[1])
            errs_px.append(e); scales.append(s)
        if errs_px:
            med = float(_np.median(errs_px))
            info["reproj"] = med
            info["obs_scale"] = scales[0]
            if med > 5.0:
                errs.append(f"reprojection median {med:.2f}px at obs-scale 1/{scales[0]} "
                            "-- POSE CONVENTION LOOKS WRONG (expect <2px). "
                            "Every render would be garbage.")
            elif med > 2.0:
                warns.append(f"reprojection median {med:.2f}px (expect <2px)")
    except Exception as e:
        warns.append(f"reprojection check skipped: {e}")

    # resolution vs the validated envelope: 8M gaussians at much higher res will OOM
    px = cam.width * cam.height
    if px > 2 * 1320 * 989:
        warns.append(f"{cam.width}x{cam.height} is {px/(1320*989):.1f}x the validated "
                     "resolution -- 8M gaussians may OOM; consider lowering --cap_max")

    csvp = os.path.join(root, scene, "test", "test_poses.csv")
    if not os.path.exists(csvp):
        errs.append(f"no test_poses.csv at {csvp}")
        return errs, warns, info
    with open(csvp, newline="") as fh:
        rows = list(csv.DictReader(fh))
    info["n_test"] = len(rows)
    if not rows:
        errs.append("test_poses.csv is empty")
        return errs, warns, info
    r0 = rows[0]
    for r in rows:
        if (r["fx"], r["fy"], r["width"], r["height"]) != \
           (r0["fx"], r0["fy"], r0["width"], r0["height"]):
            errs.append("test_poses.csv has MULTIPLE cameras (render asserts one)")
            break
    if (int(r0["width"]), int(r0["height"])) != (cam.width, cam.height):
        errs.append(f"W/H mismatch: cameras.bin {(cam.width, cam.height)} vs "
                    f"csv {(int(r0['width']), int(r0['height']))} -- would score 0")
    names = [r["image_name"] for r in rows]
    if len(names) != len(set(names)):
        errs.append("duplicate image_name in test_poses.csv")

    return errs, warns, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--images", default="images")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    root = os.path.expanduser(args.data_root)
    # dirs only: a stray .DS_Store must not inflate the count, or a genuinely MISSING
    # scene later would hide inside a number that never matched the listing anyway
    scenes = args.scenes or sorted(d for d in os.listdir(root)
                                   if os.path.isdir(os.path.join(root, d)))
    n_err = 0
    print(f"PREFLIGHT {root}  ({len(scenes)} scenes)\n")
    print(f"{'scene':12s} {'model':14s} {'WxH':11s} {'k1':>9s} {'path':7s} "
          f"{'train':>6s} {'test':>5s}  status")
    for s in scenes:
        if not os.path.isdir(os.path.join(root, s)):
            continue
        errs, warns, i = check_scene(root, s, args.images)
        status = "OK" if not errs else "FAIL"
        print(f"{s:12s} {i.get('model','-'):14s} "
              f"{str(i.get('wh','-')):11s} {i.get('k1',0):+9.5f} "
              f"{i.get('path','-'):7s} {i.get('n_train','-'):>6} "
              f"{i.get('n_test','-'):>5}  {status}")
        for w in warns:
            print(f"             WARN: {w}")
        for e in errs:
            print(f"             ERR : {e}")
            n_err += 1

    print()
    if n_err:
        print(f"PREFLIGHT FAILED: {n_err} error(s). Fix before spending GPU time.")
        sys.exit(1)
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
