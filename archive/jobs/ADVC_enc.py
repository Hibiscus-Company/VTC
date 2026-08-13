import os
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
SRC={"mean6_enc":"/mnt/d/avv/ADVC/mean6","K1_enc":"/mnt/d/avv/bonsai_eval/K1_noUT_aa/eval_png"}
for tag,src in SRC.items():
    d=f"/mnt/d/avv/ADVC/{tag}"; os.makedirs(d,exist_ok=True); tot=0
    for f in sorted(os.listdir(src)):
        if not f.endswith(".png"): continue
        im=Image.open(os.path.join(src,f)).convert("RGB")
        p=os.path.join(d,f[:-4]+".jpg")
        im.save(p,"JPEG",quality=100,subsampling=2,progressive=True)
        tot+=os.path.getsize(p)
    print(f"{tag}: {tot/1e6:.2f} MB / 28")
