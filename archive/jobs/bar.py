"""D4 HARNESS. One number, one scorer, one surface: 28 bonsai eval holes vs /mnt/d/avv/evalsplit/bonsai/eval_gt.
Usage:  python bar.py <render_dir> [<render_dir> ...]      # any new checkpoint's eval_png
        python bar.py --baselines                          # re-score 6-member mean + best single TODAY
Everything goes through scripts/eval_score.py so it is directly comparable to BAR=71.799  # replicate mean of the two matched-seed lam=0.01 runs (71.9030, 71.6951); noise floor 0.407 for 1v1.
Resolves the open question in the critique: the +0.110 (72.013 vs 71.903) and BAR 70.7965 differ
by 1.2165 while the eval_score.py change is worth 0.95 -- they may straddle it, so both are
re-measured here with TODAY's scorer.
"""
import os,sys,subprocess,tempfile,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
ES="/mnt/d/avv/evalsplit/bonsai"; B="/mnt/d/avv/bonsai_eval"; BAR=71.799  # replicate mean of the two matched-seed lam=0.01 runs (71.9030, 71.6951); noise floor 0.407 for 1v1
def score(d,tag):
    r=subprocess.run(["python","scripts/eval_score.py","--render_dir",d,"--gt_dir",f"{ES}/eval_gt","--tag",tag],
                     cwd="/mnt/c/Users/BKAI/an_plaza2/FastGS",capture_output=True,text=True)
    out=(r.stdout+r.stderr).strip().splitlines()
    line=[l for l in out if "SCORE" in l.upper() or "score" in l]
    return line[-1] if line else out[-1] if out else "(no output)"
if "--baselines" in sys.argv:
    ARMS=[a for a in sorted(os.listdir(B)) if os.path.isdir(f"{B}/{a}/eval_png") and a!="K2_clip_full"]
    print(f"arms: {ARMS}")
    st=sorted(f[:-4] for f in os.listdir(f"{B}/{ARMS[0]}/eval_png") if f.endswith(".png"))
    tmp=tempfile.mkdtemp(prefix="mean6_",dir="/mnt/d/avv")
    for s in st:
        a=np.mean([np.asarray(Image.open(f"{B}/{x}/eval_png/{s}.png").convert("RGB"),dtype=np.float64) for x in ARMS],0)
        Image.fromarray(np.clip(a+0.5,0,255).astype(np.uint8)).save(f"{tmp}/{s}.png")
    print(f"\n{'arm':>22} : {'result'}")
    print(f"{'6-member MEAN':>22} : {score(tmp,'mean6')}")
    for a in ARMS: print(f"{a:>22} : {score(f'{B}/{a}/eval_png',a)}")
    print(f"\nBAR (campaign, today's scorer) = {BAR}")
    import shutil; shutil.rmtree(tmp,ignore_errors=True)
else:
    for d in sys.argv[1:]:
        n=len([f for f in os.listdir(d) if f.endswith('.png')]) if os.path.isdir(d) else 0
        print(f"{os.path.basename(d.rstrip('/')):>28} (n={n:3d}) : {score(d,'cand')}   [BAR {BAR}]")
