import zipfile, csv, os, io, struct, hashlib, collections
R31="/mnt/d/avv/submissions/sub_round31_fields.zip"
R32="/mnt/d/avv/submissions/sub_round32_videolam.zip"
ROOT="/mnt/d/avv/data/phase1/private_set2"
SCENES=["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]

def jpeg_dims(b):
    if b[:2]!=b'\xff\xd8': return None,"not-jpeg-soi"
    i=2; n=len(b)
    while i<n-1:
        if b[i]!=0xFF: return None,"bad-marker@%d"%i
        m=b[i+1]; i+=2
        while m==0xFF: m=b[i]; i+=1
        if m in (0xD8,0x01) or 0xD0<=m<=0xD7: continue
        L=struct.unpack('>H',b[i:i+2])[0]
        if m in (0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF):
            prec=b[i+2]; h=struct.unpack('>H',b[i+3:i+5])[0]; w=struct.unpack('>H',b[i+5:i+7])[0]
            ncomp=b[i+7]
            return (w,h,ncomp,m),None
        i+=L
    return None,"no-sof"

csvmap={}
for s in SCENES:
    with open(os.path.join(ROOT,s,"test","test_poses.csv")) as f:
        rd=csv.DictReader(f)
        csvmap[s]=[(r["image_name"],int(r["width"]),int(r["height"])) for r in rd]

for tag,p in [("r31",R31),("r32",R32)]:
    z=zipfile.ZipFile(p)
    byscene=collections.defaultdict(dict)
    for i in z.infolist():
        sc,_,nm=i.filename.partition('/')
        byscene[sc][nm]=i
    print("=====",tag)
    prob=[]
    for s in SCENES:
        want={n:(w,h) for n,w,h in csvmap[s]}
        got=byscene.get(s,{})
        missing=sorted(set(want)-set(got)); extra=sorted(set(got)-set(want))
        dimbad=[]; fmtbad=[]; eoibad=[]; prog=0; ncomps=collections.Counter()
        for n,inf in got.items():
            b=z.read(inf.filename)
            d,err=jpeg_dims(b)
            if err: fmtbad.append((n,err)); continue
            w,h,nc,m=d
            ncomps[nc]+=1
            if m==0xC2: prog+=1
            if n in want and (w,h)!=want[n]: dimbad.append((n,(w,h),want[n]))
            if b[-2:]!=b'\xff\xd9': eoibad.append(n)
        print(f"  {s}: csv={len(want)} zip={len(got)} missing={len(missing)} extra={len(extra)} dimbad={len(dimbad)} fmtbad={len(fmtbad)} noEOI={len(eoibad)} progressive={prog} ncomp={dict(ncomps)}")
        if missing[:3]: print("     missing ex:",missing[:3])
        if extra[:3]: print("     extra ex:",extra[:3])
        if dimbad[:3]: print("     dimbad ex:",dimbad[:3])
        if fmtbad[:3]: print("     fmtbad ex:",fmtbad[:3])
        if eoibad[:3]: print("     noEOI ex:",eoibad[:3])
