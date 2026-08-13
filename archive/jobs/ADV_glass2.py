import os, numpy as np
from PIL import Image, ImageDraw
GT="/mnt/d/avv/evalsplit/bonsai/eval_gt"; ARM="/mnt/d/avv/bonsai_eval/K4_pC_seed7/eval_render"
GY,GX=8,12
fns=sorted(os.listdir(GT))
# --- annotated visual of one eval GT + its render, band lines drawn
fn=fns[len(fns)//2]; st=os.path.splitext(fn)[0]
g=Image.open(os.path.join(GT,fn)).convert("RGB"); r=Image.open(os.path.join(ARM,st+".jpg")).convert("RGB")
W,H=g.size
canv=Image.new("RGB",(W,2*H),(0,0,0)); canv.paste(g,(0,0)); canv.paste(r,(0,H))
d=ImageDraw.Draw(canv)
for k in (2,3,5):
    y=int(H*k/8); d.line([(0,y),(W,y)],fill=(255,0,0),width=6); d.line([(0,y+H),(W,y+H)],fill=(255,0,0),width=6)
for i in range(1,GX):
    x=int(W*i/GX); d.line([(x,0),(x,2*H)],fill=(0,255,255),width=2)
canv.resize((960,1080)).save("/home/bkai/.claude/jobs/1c9cf7e9/tmp/ADV_bands.png")
print("wrote ADV_bands.png  image",st)
# --- detail-deficit regression
def cells(a,ys,xs): return np.array([[a[ys[i]:ys[i+1],xs[j]:xs[j+1]].mean() for j in range(GX)] for i in range(GY)])
E=[];GTV=[];RTV=[]
for fn in fns:
    st=os.path.splitext(fn)[0]; rp=os.path.join(ARM,st+".jpg")
    if not os.path.exists(rp): continue
    g=np.asarray(Image.open(os.path.join(GT,fn)).convert("RGB"),np.float32)
    r=np.asarray(Image.open(rp).convert("RGB"),np.float32)
    H,W,_=g.shape; ys=np.linspace(0,H,GY+1).astype(int); xs=np.linspace(0,W,GX+1).astype(int)
    gg=g.mean(2); rr=r.mean(2)
    tv=lambda a: np.abs(np.diff(a,axis=0,prepend=a[:1]))+np.abs(np.diff(a,axis=1,prepend=a[:,:1]))
    E.append(cells(np.abs(g-r).mean(2),ys,xs)); GTV.append(cells(tv(gg),ys,xs)); RTV.append(cells(tv(rr),ys,xs))
E=np.array(E).mean(0);GTV=np.array(GTV).mean(0);RTV=np.array(RTV).mean(0)
y=E.ravel(); x1=GTV.ravel(); x2=(GTV-RTV).ravel()
def fit(X,y):
    X=np.column_stack([np.ones(len(y))]+X); b=np.linalg.lstsq(X,y,None)[0]
    p=X@b; return b, 1-((y-p)**2).sum()/((y-y.mean())**2).sum(), y-p
b1,r2a,res1=fit([x1],y); b2,r2b,res2=fit([x2],y); b3,r2c,res3=fit([x1,x2],y)
print("model err~GTtv        : R2=%.3f  b=%.3f  intercept %.2f"%(r2a,b1[1],b1[0]))
print("model err~DEFICIT     : R2=%.3f  b=%.3f  intercept %.2f"%(r2b,b2[1],b2[0]))
print("model err~GTtv+DEFICIT: R2=%.3f  bGT=%.3f bDEF=%.3f"%(r2c,b3[1],b3[2]))
bands={"top01":[0,1],"row2":[2],"mid34":[3,4],"bot567":[5,6,7]}
for k,rows in bands.items():
    m=np.zeros((GY,GX),bool); m[rows]=True; m=m.ravel()
    print("  %-7s band residual: GTtv-only %+6.3f   DEFICIT-only %+6.3f   both %+6.3f"%(k,res1[m].mean(),res2[m].mean(),res3[m].mean()))
# row-wise
print("\nrow: err  GTtv  ratio  RTV/GTV  residual(GTtv-only)")
for i in range(GY):
    print(" %d  %5.2f %5.2f  %5.3f  %5.3f   %+6.3f"%(i,E[i].mean(),GTV[i].mean(),E[i].mean()/GTV[i].mean(),RTV[i].mean()/GTV[i].mean(),res1[i*GX:(i+1)*GX].mean()))
print("\ncorr(band ratio, band 1-RTV/GTV) over 8 rows: %.3f"%np.corrcoef(E.mean(1)/GTV.mean(1), 1-RTV.mean(1)/GTV.mean(1))[0,1])
