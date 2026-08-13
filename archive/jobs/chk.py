import struct,os,csv,numpy as np
D='/mnt/d/avv/data/phase1/private_set2'
def read_images_bin(p, want_pts=False):
    out={}
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            iid,qw,qx,qy,qz,tx,ty,tz,cid=struct.unpack('<idddddddi',f.read(64))
            name=b''
            while True:
                c=f.read(1)
                if c==b'\x00': break
                name+=c
            npts=struct.unpack('<Q',f.read(8))[0]
            raw=f.read(24*npts)
            if want_pts:
                a=np.frombuffer(raw,dtype=np.dtype([('x','<f8'),('y','<f8'),('id','<i8')]))
                out[name.decode()]=(iid,npts,a)
            else:
                out[name.decode()]=(iid,npts,None)
    return out
for scene in ['HCM0421','chair','bonsai']:
    ib=f'{D}/{scene}/train/sparse/0/images.bin'
    im=read_images_bin(ib, want_pts=True)
    trainfiles=set(os.listdir(f'{D}/{scene}/train/images'))
    testnames=set(r['image_name'] for r in csv.DictReader(open(f'{D}/{scene}/test/test_poses.csv')))
    keys=set(im.keys())
    print(f"\n=== {scene}")
    print(f"  images.bin entries : {len(im)}")
    print(f"  train/images files : {len(trainfiles)}")
    print(f"  test csv names     : {len(testnames)}")
    print(f"  bin ∩ train        : {len(keys&trainfiles)}")
    print(f"  bin ∩ test         : {len(keys&testnames)}   <-- test frames present in the released model?")
    tn=sorted(keys&testnames); trn=sorted(keys&trainfiles)
    if tn:
        npt_te=np.array([im[k][1] for k in tn]); npt_tr=np.array([im[k][1] for k in trn])
        print(f"  2D keypoint tracks per TEST image : median {np.median(npt_te):.0f}  (train median {np.median(npt_tr):.0f})")
        valid=[np.sum(im[k][2]['id']>=0) for k in tn]
        print(f"  of which have a 3D point id       : median {np.median(valid):.0f}")
