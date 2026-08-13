import json,sys,glob,os
d="/home/bkai/.claude/projects/-mnt-c-Users-BKAI-an-plaza2-FastGS/1c9cf7e9-44fa-44d0-9bb8-375127c7789b/subagents/workflows/wf_36cd031e-c6b/"
for f in sorted(glob.glob(d+"agent-*.jsonl"), key=lambda p:-os.path.getsize(p))[:3]:
    print("="*100); print("FILE",os.path.basename(f), os.path.getsize(f))
    for line in open(f, encoding='utf-8', errors='replace'):
        try: o=json.loads(line)
        except: continue
        m=o.get("message") or {}
        if o.get("type")=="assistant" or m.get("role")=="assistant":
            c=m.get("content")
            if isinstance(c,list):
                for b in c:
                    if b.get("type")=="text" and b.get("text","").strip():
                        print("\n--- ASSISTANT TEXT ---"); print(b["text"][:4000])
