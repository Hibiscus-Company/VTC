import os, sys, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import *
SC=["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]
print(f"{'scene':10s} {'lum_first':>9s} {'lum_last':>9s} {'drift':>7s} {'|slope|/img':>11s} {'lum_iqr':>8s} {'VoL_med':>8s} {'VoL_p5':>7s} {'frac<0.5med':>11s}")
for s in SC:
    src=os.path.join(SET2,s); d=os.path.join(src,"train/images")
    ns=sorted(os.listdir(d)); step=max(1,len(ns)//70); samp=ns[::step]
    L=[];V=[]
    for n in samp:
        im=cv2.imread(os.path.join(d,n),cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        L.append(im.mean())
        im2=cv2.resize(im,None,fx=1400/max(im.shape),fy=1400/max(im.shape),interpolation=cv2.INTER_AREA) if max(im.shape)>1400 else im
        V.append(cv2.Laplacian(im2,cv2.CV_64F).var())
    L=np.array(L);V=np.array(V)
    x=np.arange(len(L)); sl=np.polyfit(x,L,1)[0]
    print(f"{s:10s} {L[:5].mean():9.1f} {L[-5:].mean():9.1f} {L[-5:].mean()-L[:5].mean():+7.1f} {abs(sl)*len(L)/len(ns):11.4f} {np.percentile(L,75)-np.percentile(L,25):8.1f} {np.median(V):8.0f} {np.percentile(V,5):7.0f} {(V<0.5*np.median(V)).mean()*100:10.1f}%")
