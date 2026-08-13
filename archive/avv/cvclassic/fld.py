import os,sys,numpy as np,cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
mu=np.load('/tmp/cvdiag/res/field_mean.npy')
src,dst=sys.argv[1],sys.argv[2]
os.makedirs(dst,exist_ok=True)
for f in os.listdir(src):
    if os.path.splitext(f)[1].lower() not in ('.png','.jpg','.jpeg'): continue
    im=np.asarray(Image.open(os.path.join(src,f)).convert('RGB'),dtype=np.float32)
    H,W,_=im.shape
    fu=cv2.resize(mu,(W,H),interpolation=cv2.INTER_CUBIC)
    yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
    o=cv2.remap(im,(xx+fu[...,0]).astype(np.float32),(yy+fu[...,1]).astype(np.float32),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
    Image.fromarray(np.clip(o+0.5,0,255).astype(np.uint8)).save(os.path.join(dst,os.path.splitext(f)[0]+'.png'))
