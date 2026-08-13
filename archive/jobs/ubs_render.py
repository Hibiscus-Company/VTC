"""Render our 28 bonsai eval poses from a trained UBS model. Their train.py has no
render-from-CSV path, so build their Camera objects from test_poses.csv directly.
R,T follow 3DGS convention: R = qvec2R(q).T (world->cam rotation TRANSPOSED), T = tvec."""
import os,sys,csv,math,numpy as np,torch
sys.path.insert(0,"/home/bkai/.claude/jobs/1c9cf7e9/tmp/ubs_repo")
from PIL import Image
from scene.beta_model import BetaModel
from scene.cameras import Camera
def qvec2R(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]],dtype=np.float64)
ply,csvp,out=sys.argv[1],sys.argv[2],sys.argv[3]
sh=int(sys.argv[4]) if len(sys.argv)>4 else 0
idim=int(sys.argv[5]) if len(sys.argv)>5 else 6
os.makedirs(out,exist_ok=True)
m=BetaModel(sh_degree=sh); m.input_dim=idim
m.load_ply(ply)
print("loaded",ply,"N=",m._xyz.shape[0],"input_dim",getattr(m,"input_dim","?"),flush=True)
rows=list(csv.DictReader(open(csvp)))
for i,r in enumerate(rows):
    q=np.array([float(r["qw"]),float(r["qx"]),float(r["qy"]),float(r["qz"])])
    T=np.array([float(r["tx"]),float(r["ty"]),float(r["tz"])])
    R=qvec2R(q).T
    W,H=int(r["width"]),int(r["height"]); fx,fy=float(r["fx"]),float(r["fy"])
    cam=Camera(colmap_id=i,R=R,T=T,FoVx=2*math.atan(W/(2*fx)),FoVy=2*math.atan(H/(2*fy)),
               image=torch.zeros(3,H,W),gt_alpha_mask=None,
               image_name=os.path.splitext(r["image_name"])[0],uid=i,resolution=(W,H))
    cam=cam.cuda()
    with torch.no_grad(): img=m.render(cam)["render"].clamp(0,1)
    a=(img.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    Image.fromarray(a).save(os.path.join(out,os.path.splitext(r["image_name"])[0]+".png"))
print("rendered",len(rows),"->",out,flush=True)
