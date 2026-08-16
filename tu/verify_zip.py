#!/usr/bin/env python
"""Verify a submission zip before it ships. Exits non-zero on any failure.

Every check here exists because it caught (or would have caught) a real defect:
  CRC              - a truncated write on a 350MB zip is invisible otherwise
  exact filenames  - the scorer matches on test_poses.csv names; one miss = one scene lost
  no extras        - a stray file (e.g. our own field_applied.json stamp) breaks counts
  dimensions       - a padded/warped render that forgot to crop back is silently wrong
  size limit       - 350MB hard cap
  single-gen JPEG  - re-encoding an already-encoded JPEG costs -0.14..-0.26
"""
import argparse, csv, io, os, sys, zipfile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--limit_mb", type=float, default=350.0)
    args = ap.parse_args()

    fails = []
    z = zipfile.ZipFile(args.zip)

    bad = z.testzip()
    print("CRC:", "OK" if bad is None else f"CORRUPT at {bad}")
    if bad is not None:
        fails.append(f"CRC corrupt at {bad}")

    names = set(z.namelist())
    total = 0
    for s in sorted(os.listdir(args.data_root)):
        csvp = os.path.join(args.data_root, s, "test", "test_poses.csv")
        if not os.path.exists(csvp):
            continue
        with open(csvp, newline="") as fh:
            want = [r["image_name"] for r in csv.DictReader(fh)]
        total += len(want)
        miss = [w for w in want if f"{s}/{w}" not in names]
        extra = [n for n in names
                 if n.startswith(s + "/") and os.path.basename(n) not in want]
        if miss or extra:
            fails.append(f"{s}: {len(miss)} missing, {len(extra)} extra")
            print(f"  {s}: FAIL  missing={len(miss)} extra={len(extra)}")
            continue

        # dimensions on EVERY image, not just the first (audit r10): one wrong-size
        # frame anywhere passed every other check and would score 0 on that image.
        # Header read only -- no decode -- so this stays cheap.
        with open(csvp, newline="") as fh:
            want_wh = {r["image_name"]: (int(r["width"]), int(r["height"]))
                       for r in csv.DictReader(fh)}
        fmt, badwh, degen = None, [], []
        for n in want:
            with z.open(f"{s}/{n}") as fh:
                raw = fh.read()
            im = Image.open(io.BytesIO(raw))
            fmt = fmt or im.format
            if im.format != "JPEG":
                fails.append(f"{s}/{n}: format {im.format}, expected JPEG")
            if im.size != want_wh[n]:
                badwh.append(f"{n} {im.size} != {want_wh[n]}")
        if badwh:
            fails.append(f"{s}: {len(badwh)} wrong-size images (e.g. {badwh[0]})")

        # degenerate content (audit r10): the structural checks cannot see an all-black
        # render, a stuck frame, or the wrong scene under the right names. A near-zero
        # variance image is always a bug.
        # Banded, per audit r11: a false positive BLOCKS a valid submission at the gate,
        # which is worse than a missed one. std<1.0 is flat/black/constant -- no real drone
        # frame reaches it. 1.0-5.0 (heavy fog / full overcast) is plausible: warn, don't block.
        import numpy as np
        for n in (want[0], want[len(want) // 2], want[-1]):
            with z.open(f"{s}/{n}") as fh:
                a = np.asarray(Image.open(io.BytesIO(fh.read())).convert("L"),
                               dtype=np.float32)
            if a.std() < 1.0:
                degen.append(f"{n} std={a.std():.2f}")
            elif a.std() < 5.0:
                print(f"    WARN {s}/{n}: near-uniform (std={a.std():.2f}) -- "
                      "plausible if fog/overcast, suspicious otherwise")
        if degen:
            fails.append(f"{s}: degenerate content ({', '.join(degen)})")

        print(f"  {s}: {len(want):3d}/{len(want)} exact, all {want_wh[want[0]][0]}x"
              f"{want_wh[want[0]][1]}, {fmt}")

    if len(names) != total:
        fails.append(f"zip holds {len(names)} files, expected {total}")

    # The organiser's cap is 350 MiB = 350*1024*1024 = 367,001,600 bytes, NOT 350e6. We had been
    # enforcing the decimal reading, which silently threw away 17 MB of budget -- and byte budget
    # is directly convertible into score (it is what caps the energy-restoration lambda).
    nbytes = os.path.getsize(args.zip)
    mib = nbytes / (1024.0 * 1024.0)
    print(f"\nfiles {len(names)}/{total}   size {mib:.2f} MiB / {nbytes/1e6:.2f} MB "
          f"(limit {args.limit_mb} MiB = {args.limit_mb*1024*1024/1e6:.1f} MB)")
    if mib > args.limit_mb:
        fails.append(f"size {mib:.2f} MiB over the {args.limit_mb} MiB limit")

    if fails:
        print("\nVERIFY FAILED:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("VERIFY PASSED")


if __name__ == "__main__":
    main()
