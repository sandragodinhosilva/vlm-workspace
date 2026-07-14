#!/usr/bin/env bash
# logs-utils/daily_digest.sh — emit the DETERMINISTIC facts for a single day, as raw
# material for the /daily skill to summarize into prose. No LLM here; just gather.
#
#   daily_digest.sh              today (UTC)
#   daily_digest.sh 2026-07-13   a specific day (YYYY-MM-DD)
#
# Sources folded (matching the /daily "Index + memory + git" decision):
#   1. Run index   /mnt/data/sgsilva/logs/index.jsonl  — what actually RAN (grouped, deduped)
#   2. Vault git   ~/.claude commits that day           — memory/reports/plans touched
#   3. Reports     ~/.claude/reports/<date>_*.md         — findings docs authored that day
#
# Output is plain text on stdout: three clearly-delimited sections the model reads verbatim.
set -uo pipefail

DAY="${1:-$(date -u +%Y-%m-%d)}"
if ! [[ "$DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "usage: daily_digest.sh [YYYY-MM-DD]" >&2; exit 2
fi

IDX=/mnt/data/sgsilva/logs/index.jsonl
VAULT=/home/sgsilva/.claude
PY=/home/sgsilva/vlm-post-training-home-venv/bin/python

echo "########## DAILY DIGEST FACTS — $DAY (UTC) ##########"
echo

# ---- 1. Runs from the index -------------------------------------------------
echo "===== RUNS (from index.jsonl) ====="
if [[ -f "$IDX" ]]; then
  "$PY" - "$IDX" "$DAY" <<'PYEOF'
import json, sys
from collections import defaultdict
idx, day = sys.argv[1], sys.argv[2]
rows = []
with open(idx) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if str(r.get("ts", "")).startswith(day):
            rows.append(r)

if not rows:
    print("(no runs in index for this day)")
    sys.exit(0)

# Dedup index noise: 'finalize'/empty-origin folds are status updates, not new work.
# Keep the last status per (category, run_name); drop bare 'finalize' rows.
latest = {}
for r in rows:
    name = r.get("run_name", "")
    if name == "finalize":
        continue
    key = (r.get("category", ""), name)
    latest[key] = r  # index.jsonl is append-order, so last wins

by_cat = defaultdict(list)
for (cat, name), r in latest.items():
    by_cat[cat].append(r)

status_glyph = {"done": "✓", "running": "…", "failed": "✗", "killed": "✗",
                "nfs_lost": "?"}
total = sum(len(v) for v in by_cat.values())
print(f"{total} distinct runs across {len(by_cat)} categories\n")
for cat in sorted(by_cat):
    runs = sorted(by_cat[cat], key=lambda r: r.get("ts", ""))
    print(f"[{cat}] ({len(runs)})")
    for r in runs:
        g = status_glyph.get(r.get("status", ""), "·")
        dur = r.get("dur_s")
        durs = f" {int(dur)}s" if isinstance(dur, (int, float)) and dur else ""
        hhmm = r.get("ts", "")[11:16]
        print(f"  {g} {hhmm} {r.get('run_name','')}{durs}")
    print()
PYEOF
else
  echo "(index.jsonl not found at $IDX)"
fi
echo

# ---- 2. Vault git activity --------------------------------------------------
echo "===== VAULT GIT COMMITS (~/.claude) ====="
if git -C "$VAULT" rev-parse --git-dir >/dev/null 2>&1; then
  next_day=$(date -u -d "$DAY +1 day" +%Y-%m-%d 2>/dev/null || echo "$DAY")
  log=$(git -C "$VAULT" log --since="$DAY 00:00" --until="$next_day 00:00" \
        --name-only --pretty=format:'· %s' 2>/dev/null)
  if [[ -n "$log" ]]; then echo "$log"; else echo "(no vault commits this day)"; fi
else
  echo "(vault is not a git repo)"
fi
echo

# ---- 3. Reports authored that day ------------------------------------------
echo "===== REPORTS AUTHORED (~/.claude/reports/${DAY}_*) ====="
found=$(find "$VAULT/reports" -name "${DAY}_*.md" 2>/dev/null | sort)
if [[ -n "$found" ]]; then
  while IFS= read -r f; do
    echo "· ${f#"$VAULT/"}"
  done <<< "$found"
else
  echo "(no reports authored this day)"
fi
echo
echo "########## END FACTS ##########"
