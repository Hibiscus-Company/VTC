import zipfile, csv, os, io, collections, struct, hashlib
R31="/mnt/d/avv/submissions/sub_round31_fields.zip"
R32="/mnt/d/avv/submissions/sub_round32_videolam.zip"
CSVROOT="/mnt/d/avv/data/phase1/private_set2"

def info(z):
    with zipfile.ZipFile(z) as f:
        return f.infolist()

for tag,p in [("r31",R31),("r32",R32)]:
    il=info(p)
    names=[i.filename for i in il]
    dirs=[n for n in names if n.endswith('/')]
    files=[n for n in names if not n.endswith('/')]
    print(tag,"filesize",os.path.getsize(p),"<=367001600:",os.path.getsize(p)<=367001600)
    print(tag,"entries total",len(names),"files",len(files),"dir-entries",len(dirs))
    dup=[k for k,v in collections.Counter(names).items() if v>1]
    print(tag,"dup arcnames:",dup)
    # top-level structure
    tops=collections.Counter(n.split('/')[0] for n in files)
    print(tag,"top-level:",dict(tops))
    depth=collections.Counter(n.count('/') for n in files)
    print(tag,"path depth hist:",dict(depth))
    ext=collections.Counter(os.path.splitext(n)[1].lower() for n in files)
    print(tag,"ext:",dict(ext))
    print(tag,"uncompressed sum",sum(i.file_size for i in il),"compressed sum",sum(i.compress_size for i in il))
    print()
