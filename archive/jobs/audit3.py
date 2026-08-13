import zipfile, hashlib, collections
R31="/mnt/d/avv/submissions/sub_round31_fields.zip"
R32="/mnt/d/avv/submissions/sub_round32_videolam.zip"
TOW={"HCM0421","HCM0539","HCM0540","HCM0644","HCM0674"}
a=zipfile.ZipFile(R31); b=zipfile.ZipFile(R32)
A={i.filename:i for i in a.infolist()}; B={i.filename:i for i in b.infolist()}
assert set(A)==set(B), (len(set(A)-set(B)),len(set(B)-set(A)))
print("arcname sets identical:",set(A)==set(B))
tow_diff=[]; tow_n=0; vid_diff=[]; szdelta=collections.Counter()
bytes31=collections.Counter(); bytes32=collections.Counter()
for n in sorted(A):
    sc=n.split('/')[0]
    ia,ib=A[n],B[n]
    bytes31[sc]+=ia.file_size; bytes32[sc]+=ib.file_size
    same_crc = ia.CRC==ib.CRC and ia.file_size==ib.file_size
    if sc in TOW:
        tow_n+=1
        if not same_crc: tow_diff.append(n)
        else:
            # deep verify actual bytes for a sample-free full check via sha of content
            pass
    else:
        if not same_crc: vid_diff.append(n)
print("tower entries:",tow_n,"CRC/size differing:",len(tow_diff), tow_diff[:5])
print("video entries:",58+28,"CRC/size differing:",len(vid_diff))
# full byte verify of all tower entries (sha256)
bad=0
for n in sorted(A):
    if n.split('/')[0] not in TOW: continue
    if hashlib.sha256(a.read(n)).digest()!=hashlib.sha256(b.read(n)).digest(): bad+=1
print("tower entries byte-identical (sha256): ", tow_n-bad, "/", tow_n, " mismatches:", bad)
print()
for sc in ["HCM0421","HCM0539","HCM0540","HCM0644","HCM0674","chair","bonsai"]:
    print(f"{sc:9s} r31={bytes31[sc]:>10d}  r32={bytes32[sc]:>10d}  delta={bytes32[sc]-bytes31[sc]:+d}")
print("TOTAL   r31=%d r32=%d delta=%+d"%(sum(bytes31.values()),sum(bytes32.values()),sum(bytes32.values())-sum(bytes31.values())))
# headroom
import os
print("zip file bytes r32=%d  cap=367001600  headroom=%d (%.2f MiB)"%(os.path.getsize(R32),367001600-os.path.getsize(R32),(367001600-os.path.getsize(R32))/1048576))
