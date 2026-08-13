import struct,os,csv
D='/mnt/d/avv/data/phase1/private_set2'; P='/mnt/d/avv/data/phase1/public_set'
def names(p):
    out=[]
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            f.read(64); nm=b''
            while True:
                c=f.read(1)
                if c==b'\x00': break
                nm+=c
            k=struct.unpack('<Q',f.read(8))[0]; f.read(24*k); out.append(nm.decode())
    return out
print("=== private_set2 ===")
for s in ['HCM0421','HCM0539','HCM0540','HCM0644','HCM0674','chair','bonsai']:
    ib=f'{D}/{s}/train/sparse/0/images.bin'
    nm=set(names(ib)); tr=set(os.listdir(f'{D}/{s}/train/images'))
    te=set(r['image_name'] for r in csv.DictReader(open(f'{D}/{s}/test/test_poses.csv')))
    extra=nm-tr-te
    print(f"  {s:8s} bin={len(nm):4d} train={len(tr):4d} test={len(te):3d} EXTRA={len(extra):3d}  ex.sample={sorted(extra)[:2]}")
print("=== public_set (sanctioned GT surface) ===")
for s in sorted(os.listdir(P)):
    ib=f'{P}/{s}/train/sparse/0/images.bin'
    if not os.path.exists(ib): print(f"  {s}: no bin at {ib}"); continue
    nm=set(names(ib)); tr=set(os.listdir(f'{P}/{s}/train/images'))
    tedir=f'{P}/{s}/test/images'
    te=set(os.listdir(tedir)) if os.path.isdir(tedir) else set()
    print(f"  {s:8s} bin={len(nm):4d} train={len(tr):4d} testGTimgs={len(te):3d} bin∩test={len(nm&te):3d} EXTRA={len(nm-tr-te):3d}")
