import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *

SCENES = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674", "chair", "bonsai"]
for s in SCENES:
    root = os.path.join(SET2, s)
    tr, sp = train_poses(root)
    te = load_poses_csv(os.path.join(root, "test/test_poses.csv"))
    on_disk = set(os.listdir(os.path.join(root, "train/images")))
    sparse_names = set(p["name"] for p in tr)
    test_names = set(t["name"] for t in te)
    print(f"=== {s}")
    print(f"  images on disk        : {len(on_disk)}")
    print(f"  images in sparse model: {len(sparse_names)}")
    print(f"  test poses            : {len(test_names)}")
    print(f"  disk ∩ sparse         : {len(on_disk & sparse_names)}")
    print(f"  test ∩ sparse         : {len(test_names & sparse_names)}")
    print(f"  sparse - disk - test  : {len(sparse_names - on_disk - test_names)}")
    extra = sorted(sparse_names - on_disk - test_names)
    print(f"  examples extra: {extra[:4]}")
    # verify pose agreement for test images present in sparse
    byname = {p["name"]: p for p in tr}
    errs = []
    for t in te:
        if t["name"] in byname:
            p = byname[t["name"]]
            c1 = cam_center(p["q"], p["t"]); c2 = cam_center(t["q"], t["t"])
            errs.append(np.linalg.norm(c1 - c2))
    if errs:
        print(f"  center mismatch for test-in-sparse: max={max(errs):.3e} median={np.median(errs):.3e}")
