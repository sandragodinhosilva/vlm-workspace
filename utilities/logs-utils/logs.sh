#!/usr/bin/env bash
# logs-utils/logs.sh — query the centralized run index at /mnt/data/sgsilva/logs/index.jsonl.
#   logs.sh                      last 20 runs, all categories
#   logs.sh <category>           runs in one category (grpo|sft|sam3d|dataset|export|eval|oracle|serve|claude|misc)
#   logs.sh --since 7d|24h       runs started within window
#   logs.sh --running            in-flight (status=running) runs
#   logs.sh --failed             status in {failed,killed,nfs_lost}
#   logs.sh --grep PATTERN       match run_name/cmd/logfile (and grep bodies); add --paths for raw logfile paths
#   logs.sh --reconcile          fold .index_pending.jsonl files back into index.jsonl (also auto-run each call)
#   logs.sh --import-archive     index legacy logs in _archive/pre_20260617/ (one-shot migration Phase 2)
#   logs.sh --rebuild-index      re-walk all *.meta.json and rebuild index.jsonl from scratch
#   logs.sh --gc                 gzip terminal logs >14d, delete zero-byte logs, prune misc >180d
#   logs.sh --reap               mark stale 'running' entries as nfs_lost (node dead, no trap fired)
set -uo pipefail
ROOT=/mnt/data/sgsilva/logs
IDX="$ROOT/index.jsonl"
LOCK="$ROOT/index.lock"
PY=/home/sgsilva/vlm-post-training-home-venv/bin/python

reconcile() {
    { exec 9>"$LOCK"; } 2>/dev/null
    if flock -w 10 9 2>/dev/null; then
        find "$ROOT" -name '.index_pending.jsonl' -type f 2>/dev/null | while read -r p; do
            cat "$p" >>"$IDX" && : >"$p"
        done
        flock -u 9
    fi
}

# --- special modes handled in bash before passing to Python ---

if [[ "${1:-}" == "--reconcile" ]]; then
    reconcile
    echo "Reconciled pending index entries."
    exit 0
fi

if [[ "${1:-}" == "--import-archive" ]]; then
    reconcile
    archive="$ROOT/_archive/pre_20260617"
    if [[ ! -d "$archive" ]]; then echo "Archive dir not found: $archive" >&2; exit 1; fi
    count=0
    while IFS= read -r -d '' f; do
        # infer category from path
        case "$f" in
            */grpo_logs/*) cat=grpo ;;
            */sft_logs/*)  cat=sft  ;;
            */export_logs/*) cat=export ;;
            *) cat=misc ;;
        esac
        fname=$(basename "$f")
        mtime=$(stat -c%Y "$f" 2>/dev/null || echo 0)
        ts=$(date -u -d "@$mtime" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -r "$mtime" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "unknown")
        line="{\"ts\":\"$ts\",\"epoch\":$mtime,\"category\":\"$cat\",\"run_name\":\"${fname%.log}\",\"origin\":\"archive\",\"node\":\"archive\",\"status\":\"done\",\"logfile\":\"$f\",\"exit\":0,\"dur_s\":null,\"archived\":true}"
        { exec 9>"$LOCK"; flock -w 10 9 2>/dev/null && printf '%s\n' "$line" >>"$IDX" && flock -u 9; } 2>/dev/null
        count=$((count+1))
    done < <(find "$archive" -type f -name '*.log' -print0 2>/dev/null)
    echo "Indexed $count archive log files."
    exit 0
fi

if [[ "${1:-}" == "--rebuild-index" ]]; then
    echo "Rebuilding index.jsonl from all meta.json files..."
    tmp="$IDX.rebuild.$$"
    find "$ROOT" -name 'meta.json' -o -name '*.meta.json' 2>/dev/null | sort | while read -r m; do
        # emit the meta as an index line
        /usr/bin/python3 -c "
import json, sys
try:
    d=json.load(open('$m'))
    lf=d.get('logfile','')
    line={
        'ts': d.get('started',''),
        'epoch': d.get('start_epoch',0),
        'category': d.get('category','misc'),
        'run_name': d.get('run_name',''),
        'origin': d.get('origin',''),
        'node': d.get('node',''),
        'status': d.get('status','done'),
        'logfile': lf,
        'exit': None,
        'dur_s': None,
    }
    print(json.dumps(line))
except Exception as e:
    print(f'# skip $m: {e}', file=sys.stderr)
" 2>/dev/null
    done >"$tmp"
    mv "$tmp" "$IDX"
    wc -l <"$IDX"
    echo "Done rebuilding index."
    exit 0
fi

if [[ "${1:-}" == "--gc" ]]; then
    echo "GC: gzipping terminal logs >14d in _archive and rotated segments >14d..."
    cutoff=$(date -d '14 days ago' +%s 2>/dev/null || date -v-14d +%s 2>/dev/null)
    find "$ROOT/_archive" -type f -name '*.log' ! -name '*.gz' 2>/dev/null | while read -r f; do
        mt=$(stat -c%Y "$f" 2>/dev/null); [[ -z "$mt" ]] && continue
        (( mt < cutoff )) && gzip -q "$f" && echo "  gz: $f"
    done
    find "$ROOT" -type f -name 'run.log.[0-9]*' ! -name '*.gz' 2>/dev/null | while read -r f; do
        mt=$(stat -c%Y "$f" 2>/dev/null); [[ -z "$mt" ]] && continue
        (( mt < cutoff )) && gzip -q "$f" && echo "  gz: $f"
    done
    echo "GC: removing zero-byte .log files..."
    find "$ROOT" -type f -name '*.log' -empty 2>/dev/null | grep -v '_archive' | while read -r f; do
        echo "  rm (zero-byte): $f"; rm -f "$f"
    done
    echo "GC done."
    exit 0
fi

if [[ "${1:-}" == "--reap" ]]; then
    echo "Reap: marking stale 'running' entries whose node is dead..."
    "$PY" - <<'PYEOF'
import json, subprocess, time
IDX="/mnt/data/sgsilva/logs/index.jsonl"
LOCK="/mnt/data/sgsilva/logs/index.lock"
# stale = running and epoch older than 24h
cutoff = time.time() - 86400
runs = {}
with open(IDX, errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        lf = r.get("logfile")
        if not lf: continue
        cur = runs.get(lf)
        if cur is None or (r.get("epoch") or 0) >= (cur.get("epoch") or 0):
            runs[lf] = r
stale = [r for r in runs.values() if r.get("status") == "running" and (r.get("epoch") or 0) < cutoff]
if not stale:
    print("No stale running entries found.")
else:
    import fcntl
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(IDX, "a") as out, open(LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for r in stale:
            lf = r.get("logfile", "")
            entry = {"ts": ts, "epoch": int(time.time()), "category": r.get("category",""),
                     "run_name": "reap", "origin": "", "node": "",
                     "status": "nfs_lost", "logfile": lf, "exit": None, "dur_s": None}
            out.write(json.dumps(entry) + "\n")
            print(f"  nfs_lost: {r.get('run_name','')} ({lf})")
        fcntl.flock(lock, fcntl.LOCK_UN)
    print(f"Reaped {len(stale)} stale entries.")
PYEOF
    exit 0
fi

# --- auto-reconcile, then hand off to Python for query ---
reconcile

"$PY" - "$@" <<'PYEOF'
import sys, json, time, os, subprocess
IDX="/mnt/data/sgsilva/logs/index.jsonl"
args=sys.argv[1:]
cat=None; since=None; only=None; pat=None; paths=False
i=0
while i < len(args):
    a=args[i]
    if a=="--since":    since=args[i+1]; i+=2; continue
    if a=="--running":  only="running";  i+=1; continue
    if a=="--failed":   only="failed";   i+=1; continue
    if a=="--grep":     pat=args[i+1];   i+=2; continue
    if a=="--paths":    paths=True;      i+=1; continue
    if a=="--reconcile": sys.exit(0)
    if not a.startswith("--"): cat=a; i+=1; continue
    i+=1

def parse_since(s):
    if not s: return None
    n=int(s[:-1]); u=s[-1]
    return time.time() - n*{"d":86400,"h":3600,"m":60}.get(u, 1)
cutoff=parse_since(since)

# collapse to last line per logfile
runs={}
try:
    with open(IDX, errors="replace") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: r=json.loads(line)
            except Exception: continue
            lf=r.get("logfile")
            if not lf: continue
            cur=runs.get(lf)
            if cur is None:
                runs[lf]=r
            else:
                merged=dict(cur)
                for k,v in r.items():
                    if v not in (None,"","finalize"): merged[k]=v
                if r.get("status") in ("done","failed","killed","nfs_lost"):
                    merged["status"]=r["status"]; merged["exit"]=r.get("exit"); merged["dur_s"]=r.get("dur_s")
                    merged["ts"]=r.get("ts"); merged["epoch"]=r.get("epoch")
                runs[lf]=merged
except FileNotFoundError:
    print("No index.jsonl yet. Run a job first or check /mnt/data/sgsilva/logs/", file=sys.stderr)
    sys.exit(0)

rows=list(runs.values())
if cat:          rows=[r for r in rows if r.get("category")==cat]
if only=="running": rows=[r for r in rows if r.get("status")=="running"]
if only=="failed":  rows=[r for r in rows if r.get("status") in ("failed","killed","nfs_lost")]
if cutoff:       rows=[r for r in rows if (r.get("epoch") or 0) >= cutoff]
if pat:
    pl=pat.lower()
    def hit(r):
        if pl in (r.get("run_name","")+" "+r.get("logfile","")).lower(): return True
        lf=r.get("logfile","")
        if lf and os.path.isfile(lf):
            try: return subprocess.run(["grep","-qi",pat,lf],timeout=5).returncode==0
            except Exception: pass
        return False
    rows=[r for r in rows if hit(r)]

rows.sort(key=lambda r: r.get("epoch") or 0, reverse=True)
if not (cat or only or since or pat): rows=rows[:20]

if paths:
    for r in rows: print(r.get("logfile",""))
    sys.exit(0)

ICON={"running":"…","done":"OK","failed":"XX","killed":"KILL","nfs_lost":"NFS?"}
print(f"{'STATUS':6} {'CAT':8} {'STARTED':17} {'DUR':>7} {'RUN':38} ORIGIN")
print("-"*100)
for r in rows:
    st=ICON.get(r.get("status",""),r.get("status","")[:6])
    d=r.get("dur_s"); dur=f"{d}s" if isinstance(d,int) else ("--" if r.get("status")!="running" else "live")
    rn=r.get("run_name",""); rn=rn if rn!="finalize" else "(finalize)"
    print(f"{st:6} {r.get('category',''):8} {(r.get('ts') or '')[:16]:17} {dur:>7} {rn[:38]:38} {r.get('origin','')}")
print(f"\n{len(rows)} run(s).  Full paths: logs.sh ... --paths | Tail: tail -F <path>")
PYEOF
