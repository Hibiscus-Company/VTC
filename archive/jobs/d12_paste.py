#!/usr/bin/env python
"""D12: per-image PSNR of pasting the nearest TRAIN photo into a test view.

D3 reported a MEAN oracle paste of 9.4 dB and closed photo-reuse. But the score averages
PSNR PER IMAGE, so the mean is the wrong statistic. D11 found that 23% of test views sit
within 0.5 deg of a train view (median 1.19 deg). If a TAIL of test views are genuine
near-duplicates, the train photo could beat our ~26.5 dB render ON THOSE IMAGES, and a
pose-selected hybrid would gain real points.

Selection must be POSE-ONLY to be legal. Public test GT is used here to MEASURE what the
paste scores -- diagnosis only, exactly as D1/D5 did.

Reports paste-PSNR per image vs pose proximity, and asks the only question that matters:
is there ANY test view where the raw train photo beats our render?
"""
import argparse, csv, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, "/mnt/c/Users/BKAI/an_plaza2/FastGS")
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat

Image.MAX_IMAGE_PIXELS = None


def psnr(a, b):
    m = float(((a - b) ** 2).mean())
    return 10.0 * np.log10(1.0 / max(m, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--render_root", default="/mnt/d/avv/output")
    ap.add_argument("--member", default="gsplatB9ut")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    scenes = args.scenes or sorted(d for d in os.listdir(root)
                                   if os.path.isdir(os.path.join(root, d)))
    rows_all = []
    for s in scenes:
        sp = os.path.join(root, s, "train", "sparse", "0")
        imd = os.path.join(root, s, "train", "images")
        csvp = os.path.join(root, s, "test", "test_poses.csv")
        gtd = os.path.join(root, s, "test", "images")
        rnd = os.path.join(args.render_root, f"{s}_{args.member}", "test_poses_renders_png")
        if not all(os.path.exists(p) for p in (sp, imd, csvp, gtd)):
            continue

        train = []
        for v in read_extrinsics_binary(os.path.join(sp, "images.bin")).values():
            p = os.path.join(imd, v.name)
            if not os.path.exists(p):
                continue
            R = qvec2rotmat(v.qvec)
            train.append((R, -R.T @ v.tvec, p))
        trC = np.stack([c for _, c, _ in train])
        scale = float(np.linalg.norm(trC - trC.mean(0), axis=1).mean())

        gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gtd)}
        with open(csvp, newline="") as fh:
            for r in csv.DictReader(fh):
                stem = os.path.splitext(r["image_name"])[0]
                if stem not in gt_by:
                    continue
                q = np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])])
                t = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])
                R = qvec2rotmat(q); C = -R.T @ t
                zt = R.T @ np.array([0.0, 0.0, 1.0])

                # nearest train view by camera-centre distance (pose-only, legal)
                d = np.linalg.norm(trC - C, axis=1)
                j = int(d.argmin())
                Rr, Cr, path = train[j]
                zr = Rr.T @ np.array([0.0, 0.0, 1.0])
                ang = float(np.degrees(np.arccos(np.clip(np.dot(zt, zr), -1, 1))))
                base = float(d[j] / max(scale, 1e-9))

                gt = np.asarray(Image.open(os.path.join(gtd, gt_by[stem])).convert("RGB"),
                                dtype=np.float32) / 255.0
                tr_img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
                if tr_img.shape != gt.shape:
                    continue
                p_paste = psnr(tr_img, gt)

                p_render = np.nan
                rp = os.path.join(rnd, stem + ".png")
                if os.path.exists(rp):
                    rr = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
                    if rr.shape == gt.shape:
                        p_render = psnr(rr, gt)
                rows_all.append((s, stem, ang, base, p_paste, p_render))

    if not rows_all:
        sys.exit("no data")
    ang = np.array([r[2] for r in rows_all])
    base = np.array([r[3] for r in rows_all])
    pp = np.array([r[4] for r in rows_all])
    pr = np.array([r[5] for r in rows_all])

    print(f"\n===== D12: paste-nearest-train vs our render ({len(rows_all)} test views) =====")
    print(f"paste  PSNR: mean {np.nanmean(pp):6.2f}  median {np.nanmedian(pp):6.2f}  "
          f"max {np.nanmax(pp):6.2f}")
    print(f"render PSNR: mean {np.nanmean(pr):6.2f}  median {np.nanmedian(pr):6.2f}  "
          f"max {np.nanmax(pr):6.2f}")

    win = np.nansum(pp > pr)
    print(f"\n*** test views where the RAW TRAIN PHOTO beats our render: {int(win)} / "
          f"{len(rows_all)} ({100*win/len(rows_all):.1f}%) ***")

    print("\npaste PSNR bucketed by baseline (camera-centre dist / scene scale):")
    print(f"  {'baseline':>16s} {'n':>4s} {'paste':>7s} {'render':>7s} {'ang':>6s}")
    edges = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (base >= lo) & (base < hi)
        if m.sum() == 0:
            continue
        print(f"  {lo:6.3f}-{hi:6.3f} {int(m.sum()):4d} {np.nanmean(pp[m]):7.2f} "
              f"{np.nanmean(pr[m]):7.2f} {np.nanmean(ang[m]):6.2f}")

    k = min(10, len(rows_all))
    order = np.argsort(-pp)[:k]
    print(f"\ntop {k} paste views (best-case near-duplicates):")
    print(f"  {'scene':9s} {'ang':>6s} {'base':>7s} {'paste':>7s} {'render':>7s}")
    for i in order:
        s, st, a, b, x, y = rows_all[i]
        print(f"  {s:9s} {a:6.2f} {b:7.3f} {x:7.2f} {y:7.2f}")


if __name__ == "__main__":
    main()
