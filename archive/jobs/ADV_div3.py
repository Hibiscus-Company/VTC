import os,itertools,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
B="/mnt/d/avv/bonsai_eval"
D={a:f"{B}/{a}/eval_png" for a in ['K1_noUT_aa','K4_pC_seed1k','K4_pC_seed7','eps10','eps20','ppisp_pc']}
D['sr001seed42']="/mnt/d/avv/r36_shape/sr001seed42/eval_png"
D['sr0.03']="/mnt/d/avv/r35_scalereg/sr0.03/eval_png"
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"
stems=sorted(os.path.splitext(f)[0] for f in os.listdir(GT))[:10]
A={k:np.stack([np.asarray(Image.open(f"{v}/{s}.png").convert("RGB"),dtype=np.float32) for s in stems]) for k,v in D.items()}
print("pairwise disagreement RMS in LSB (10 of 28 frames)")
rows=[]
for a,b in itertools.combinations(sorted(A),2):
    rows.append((float(np.sqrt(((A[a]-A[b])**2).mean())),a,b))
for r,a,b in sorted(rows): print(f"  {r:6.3f}  {a} | {b}")
SEED3=['sr001seed42','K4_pC_seed1k','K4_pC_seed7']
ALL=['K1_noUT_aa','K4_pC_seed1k','K4_pC_seed7','eps10','eps20','ppisp_pc']
for tag,G in (("SEED3(same recipe)",SEED3),("MEAN6(6 recipes)",ALL)):
    v=[r for r,a,b in rows if a in G and b in G]
    st=np.stack([A[g] for g in G]); sd=float(st.std(0).mean())
    print(f"{tag:22s} mean pair RMS {np.mean(v):6.3f}  member-sd/px {sd:6.3f}")
