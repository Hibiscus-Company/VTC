import zipfile, struct, hashlib, collections
R31="/mnt/d/avv/submissions/sub_round31_fields.zip"
R32="/mnt/d/avv/submissions/sub_round32_videolam.zip"
def parse(b):
    dqt=[]; samp=None; i=2; n=len(b)
    while i<n-1:
        if b[i]!=0xFF: break
        m=b[i+1]; i+=2
        while m==0xFF: m=b[i]; i+=1
        if m in (0xD8,0x01) or 0xD0<=m<=0xD7: continue
        L=struct.unpack('>H',b[i:i+2])[0]
        seg=b[i+2:i+L]
        if m==0xDB:
            dqt.append(hashlib.md5(seg).hexdigest()[:8])
        if m in (0xC0,0xC2):
            nc=seg[5]; comps=[]
            for c in range(nc):
                cid=seg[6+3*c]; hv=seg[7+3*c]
                comps.append((cid,hv>>4,hv&15))
            samp=tuple(comps)
        if m==0xDA: break
        i+=L
    return tuple(sorted(dqt)), samp
for tag,p in [("r31",R31),("r32",R32)]:
    z=zipfile.ZipFile(p)
    agg=collections.defaultdict(collections.Counter)
    for i in z.infolist():
        sc=i.filename.split('/')[0]
        agg[sc][parse(z.read(i.filename))]+=1
    print("==",tag)
    for sc in ["HCM0421","HCM0674","chair","bonsai"]:
        for k,v in agg[sc].items():
            print(f"   {sc:8s} n={v:3d} dqt={k[0]} sampling={k[1]}")
