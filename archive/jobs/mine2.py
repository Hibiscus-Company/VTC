import json,glob,os
d="/home/bkai/.claude/projects/-mnt-c-Users-BKAI-an-plaza2-FastGS/1c9cf7e9-44fa-44d0-9bb8-375127c7789b/subagents/workflows/wf_36cd031e-c6b/"
for f in sorted(glob.glob(d+"agent-*.jsonl"), key=lambda p:-os.path.getsize(p))[:3]:
    print("="*100); print("FILE",os.path.basename(f))
    for line in open(f, encoding='utf-8', errors='replace'):
        try: o=json.loads(line)
        except: continue
        m=o.get("message") or {}
        c=m.get("content")
        if not isinstance(c,list): continue
        for b in c:
            if b.get("type")=="tool_use":
                inp=b.get("input",{})
                s=inp.get("command") or inp.get("file_path") or inp.get("pattern") or ""
                print("\n>>> TOOL", b.get("name"), ":", str(s)[:600])
            if b.get("type")=="tool_result":
                cc=b.get("content")
                t = cc if isinstance(cc,str) else " ".join(x.get("text","") for x in cc if isinstance(x,dict))
                t=t.strip()
                if t: print("    RES:", t[:900].replace("\n","\n    "))
