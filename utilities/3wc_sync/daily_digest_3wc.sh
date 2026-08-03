#!/usr/bin/env bash
# 3wc_sync/daily_digest_3wc.sh — emit the DETERMINISTIC facts for one day of 3WC work,
# as raw material for the /3wc-daily-journal skill to summarize into prose. No LLM here.
#
#   daily_digest_3wc.sh              today (UTC)
#   daily_digest_3wc.sh 2026-07-28   a specific day (YYYY-MM-DD)
#
# The 3WC twin of logs-utils/daily_digest.sh, but the FACT SOURCES DIFFER — 3WC has no run
# index of its own, so the skeleton is git in dawn-research plus the vault's 3WC layer:
#   1. dawn-research  — MY commits that day (the real "what I built"), split from the team's
#   2. Team commits   — what upstream did that day (context, not my work)
#   3. Vault (3WC)    — reports/3wc + 3WC_* docs + 3WC memory touched, auto-snapshots filtered
#   4. Reports        — reports/3wc/<date>_*.md authored that day
#   5. Runs           — 3WC-ish rows in the shared run index (incidental; usually 'misc')
#   6. Artifacts      — /mnt/data/sgsilva/3wc/ files modified that day
#
# Output is plain text on stdout: clearly-delimited sections the model reads verbatim.
set -uo pipefail

DAY="${1:-$(date -u +%Y-%m-%d)}"
if ! [[ "$DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "usage: daily_digest_3wc.sh [YYYY-MM-DD]" >&2; exit 2
fi
NEXT=$(date -u -d "$DAY +1 day" +%Y-%m-%d 2>/dev/null || echo "$DAY")

REPO=/home/sgsilva/dawn-research
VAULT=/home/sgsilva/.claude
DATA=/mnt/data/sgsilva/3wc
IDX=/mnt/data/sgsilva/logs/index.jsonl
ME=sandragodinhosilva

echo "########## 3WC DAILY DIGEST FACTS — $DAY (UTC) ##########"
echo

# ---- 1+2. dawn-research commits, split mine vs team --------------------------
echo "===== MY COMMITS (dawn-research, author=$ME) ====="
echo "  [subtree-split duplicates collapsed by message; --numstat totals per commit]"
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  # NOT --all: the outbound subtree split (SUBTREES.md) replays every 3wc/ commit under a
  # stripped path, so --all reports each one twice. HEAD alone is the monorepo truth.
  mine=$(git -C "$REPO" log --author="$ME" \
         --since="$DAY 00:00" --until="$NEXT 00:00" \
         --pretty=format:'@@%h %s' --numstat 2>/dev/null \
    | awk '
        /^@@/ { if (h != "") printf "· %s  (%d file%s, +%d/-%d)\n", h, nf, (nf==1?"":"s"), add, del
                h = substr($0, 3); nf = 0; add = 0; del = 0; next }
        NF == 3 { nf++; if ($1 != "-") add += $1; if ($2 != "-") del += $2 }
        END { if (h != "") printf "· %s  (%d file%s, +%d/-%d)\n", h, nf, (nf==1?"":"s"), add, del }')
  if [[ -n "$mine" ]]; then echo "$mine"; else echo "(no commits by me this day)"; fi
else
  echo "(dawn-research is not a git repo at $REPO)"
fi
echo

echo "===== AREAS I TOUCHED (dawn-research, by directory — WHERE the work landed) ====="
echo "  [__pycache__/.pyc noise dropped; count = files changed under that dir]"
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  areas=$(git -C "$REPO" log --author="$ME" \
          --since="$DAY 00:00" --until="$NEXT 00:00" \
          --name-only --pretty=format: 2>/dev/null \
          | grep . \
          | grep -v '__pycache__' | grep -v '\.pyc$' \
          | sed -E 's#^3wc/##' \
          | sort -u \
          | awk -F/ '{ if (NF<=1) d="(repo root)"; else if (NF==2) d=$1; else d=$1"/"$2; c[d]++ }
                     END { for (k in c) printf "%6d  %s\n", c[k], k }' \
          | sort -rn | head -20)
  if [[ -n "$areas" ]]; then echo "$areas" | sed 's/^/· /'; else echo "(none)"; fi
fi
echo

echo "===== TEAM COMMITS (dawn-research, others — context only, NOT my work) ====="
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  # Subjects upstream are sometimes terse ('m', a bare filename) — keep the sha so a
  # meaningless subject is still traceable rather than silently uninformative.
  theirs=$(git -C "$REPO" log --all --perl-regexp --author="^(?!.*$ME).*$" \
           --since="$DAY 00:00" --until="$NEXT 00:00" \
           --pretty=format:'%an — %s' 2>/dev/null | sort | uniq -c \
           | awk '{ n=$1; $1=""; sub(/^ /,""); printf "· %s%s\n", $0, (n>1 ? "  (×" n " incl. subtree replay)" : "") }')
  if [[ -n "$theirs" ]]; then echo "$theirs"; else echo "(no team commits this day)"; fi
fi
echo

# ---- 3. Vault activity, 3WC paths only, auto-snapshots filtered -------------
echo "===== VAULT COMMITS touching the 3WC layer (~/.claude) ====="
echo "  [auto-snapshot commits are excluded; file lists are the signal]"
if git -C "$VAULT" rev-parse --git-dir >/dev/null 2>&1; then
  vlog=$(git -C "$VAULT" log --since="$DAY 00:00" --until="$NEXT 00:00" \
         --name-only --pretty=format:'· %s' \
         -- reports/3wc '3WC_*.md' skills/3wc-'*' \
            projects/-home-sgsilva-dawn-research 2>/dev/null \
         | grep -v '^· auto: session-end snapshot' )
  if [[ -n "$vlog" ]]; then echo "$vlog"; else echo "(no 3WC-layer vault commits this day)"; fi
else
  echo "(vault is not a git repo)"
fi
echo

# ---- 4. Reports authored that day -------------------------------------------
echo "===== 3WC REPORTS AUTHORED (~/.claude/reports/3wc/${DAY}_*) ====="
found=$(find "$VAULT/reports/3wc" -name "${DAY}_*.md" 2>/dev/null | sort)
if [[ -n "$found" ]]; then
  while IFS= read -r f; do echo "· ${f#"$VAULT/"}"; done <<< "$found"
else
  echo "(no 3WC reports authored this day)"
fi
echo

# ---- 5. Run-index rows that look 3WC-ish ------------------------------------
echo "===== RUNS mentioning 3wc (shared index — incidental, usually 'misc') ====="
echo "  [final status per run: ✓ done · … running · ✗ failed/killed · ? nfs_lost]"
if [[ -f "$IDX" ]]; then
  # A run appears as several append-only rows; a 'finalize' row carries the TRUE end status
  # but not the name. Resolve status by LOGFILE (shared across a run's rows), so a finished
  # run is never reported as still 'running'. [[feedback_no_silent_fail]]
  python3 - "$IDX" "$DAY" <<'PYEOF' || echo "(no 3wc-tagged runs in the index this day)"
import json, sys
idx, day = sys.argv[1], sys.argv[2]
by_log, order = {}, []
with open(idx) as f:
    for line in f:
        line = line.strip()
        if not line or '3wc' not in line.lower():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not str(r.get("ts", "")).startswith(day):
            continue
        lg = r.get("logfile", "")
        if lg not in by_log:
            by_log[lg] = {"name": None, "cat": r.get("category", ""), "ts": r.get("ts", "")}
            order.append(lg)
        if r.get("run_name") and r["run_name"] != "finalize":
            by_log[lg]["name"] = r["run_name"]
        by_log[lg]["status"] = r.get("status", "")   # append-order: last row wins
glyph = {"done": "✓", "running": "…", "failed": "✗", "killed": "✗", "nfs_lost": "?"}
rows = [by_log[l] for l in order if by_log[l]["name"]]
if not rows:
    sys.exit(1)
for r in rows:
    print(f"· {r['ts'][11:16]} {glyph.get(r['status'],'·')} [{r['cat']}] {r['name']}")
PYEOF
else
  echo "(index.jsonl not found at $IDX)"
fi
echo

# ---- 6. Data artifacts touched ----------------------------------------------
echo "===== MY 3WC DATA ARTIFACTS modified ($DATA) ====="
echo "  [rolled up per dir — n files, total bytes, time span; scratch dirs stay collapsed]"
if [[ -d "$DATA" ]]; then
  # Collapse to the TOP-LEVEL artifact dir: a labelling run writes hundreds of per-example
  # scratch dirs, and one row per dataset is the useful granularity for a day summary.
  arts=$(find "$DATA" -type f \
         -newermt "$DAY 00:00" ! -newermt "$NEXT 00:00" \
         -printf '%TH:%TM\t%s\t%P\n' 2>/dev/null \
    | awk -F'\t' '{ split($3, p, "/"); d = (length(p) > 1 ? p[1] : "(top level)")
                    n[d]++; b[d]+=$2; if (lo[d]==""||$1<lo[d]) lo[d]=$1; if ($1>hi[d]) hi[d]=$1 }
                  END { for (k in n) printf "%s\t%d\t%d\t%s-%s\n", k, n[k], b[k], lo[k], hi[k] }' \
    | sort -k4 \
    | awk -F'\t' '{ mb = $3/1048576
                    printf "· %s  %s  (%d file%s, %.1f MB)\n", $4, $1, $2, ($2==1?"":"s"), mb }' \
    | head -25)
  if [[ -n "$arts" ]]; then echo "$arts"; else echo "(no artifacts modified this day)"; fi
else
  echo "(no $DATA dir)"
fi
echo
echo "########## END FACTS ##########"
