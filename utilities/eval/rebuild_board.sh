#!/usr/bin/env bash
# rebuild_board.sh — regenerate the aux eval_matrix (combined + per-base) and recompile the master
# board, with a BEFORE backup + an AFTER diff so the blast radius is always visible.
#
# WHY THIS EXISTS (2026-06-22): the matrix→board rebuild is a multi-step sequence with two silent
# traps that bit us:
#   1. export_eval_matrix.py with MULTIPLE --base-model writes ONLY the combined eval_matrix.csv;
#      the per-base eval_matrix_<base>.csv files need a SEPARATE single-base invocation each, else
#      they go STALE — and the compiler reads the per-base file FIRST (PRIMARY), so a stale
#      per-base value SHADOWS the fresh combined one on the board.
#   2. Without --full-scan the exporter's incremental cache REUSES existing rows, so an exporter
#      code change silently does NOT propagate.
# This script does all of it in the right order, every time. See [[feedback_backup_before_mutating]].
#
# Usage:
#   rebuild_board.sh              # full rebuild: backup -> full-scan regen (combined+4b+27b) -> compile -> diff
#   rebuild_board.sh --no-backup  # skip the backup (e.g. when called from eval_all.sh which is incremental)
#   rebuild_board.sh --incremental# drop --full-scan (fast; only new aux runs land — for eval_all.sh's normal path)
set -uo pipefail

VPT_PY=/home/sgsilva/vlm-post-training-home-venv/bin/python
# CANONICAL (git-tracked) exporter in the vlm-post-training fork. The historical path
# aux_tasks/evals/export_eval_matrix.py is a SYMLINK into the (untracked) results tree — running the
# tracked copy here keeps the executed code under version control. The file uses absolute paths
# (EVAL_ROOT/MASTER_OUT_DIR), so it runs identically regardless of its own location.
EXPORTER=/home/sgsilva/vlm-post-training/aux_tasks/scripts/export_eval_matrix.py
COMPILER="$(dirname "$0")/compile_eval_results.py"
AUX_DIR=/mnt/data/sgsilva/results/aux
MASTER=/mnt/data/sgsilva/results/master/eval_master.csv
BACKUP_ROOT=/mnt/data/sgsilva/results/_backups
# The board's per-base split families (compiler ERA_FAMILIES) -> their exporter base-model keys.
BASES=(qwen3.5-4b qwen3.5-27b)

DO_BACKUP=1
SCAN_FLAG="--full-scan"
for a in "$@"; do
  case "$a" in
    --no-backup)   DO_BACKUP=0 ;;
    --incremental) SCAN_FLAG="" ;;
    *) echo "[rebuild_board] unknown arg: $a" >&2; exit 2 ;;
  esac
done

ts="$(date +%Y-%m-%d_%H%M%S)"
rc=0

# 1. BACKUP (before any mutation) — key artifacts to a dated dir, then prune to the last N so they
#    don't accumulate (every eval_all run takes one).
BACKUP_KEEP=20
if [[ "$DO_BACKUP" == 1 ]]; then
  bdir="$BACKUP_ROOT/$ts"
  mkdir -p "$bdir"
  cp -p "$AUX_DIR"/eval_matrix*.csv "$bdir"/ 2>/dev/null
  cp -p /mnt/data/sgsilva/results/master/eval_master*.csv "$bdir"/ 2>/dev/null
  echo "[rebuild_board] backup -> $bdir"
  # prune oldest, keep newest BACKUP_KEEP (only our own auto-snapshot dirs: YYYY-MM-DD_HHMMSS)
  mapfile -t _old < <(ls -1d "$BACKUP_ROOT"/20*_*/ 2>/dev/null | sort | head -n -"$BACKUP_KEEP")
  for d in "${_old[@]}"; do rm -rf "$d" && echo "[rebuild_board] pruned old backup $d"; done
fi

# Stash the pre-rebuild combined matrix for the AFTER diff (in scratch, not durable).
prev=/mnt/data/sgsilva/tmp/eval_matrix_prev_$ts.csv
cp -p "$AUX_DIR/eval_matrix.csv" "$prev" 2>/dev/null || true

# 2. REGEN combined (multi-base -> writes eval_matrix.csv only).
echo "[rebuild_board] export combined eval_matrix.csv ${SCAN_FLAG:-(incremental)}"
"$VPT_PY" "$EXPORTER" --base-model "$(IFS=,; echo "${BASES[*]}")" $SCAN_FLAG \
  || { echo "[rebuild_board] combined export FAILED" >&2; rc=1; }

# 3. REGEN each per-base file (single-base -> writes eval_matrix_<base>.csv). REQUIRED so the
#    compiler's PRIMARY per-base source can't shadow the combined fix with a stale value.
for b in "${BASES[@]}"; do
  echo "[rebuild_board] export eval_matrix_${b}.csv ${SCAN_FLAG:-(incremental)}"
  "$VPT_PY" "$EXPORTER" --base-model "$b" $SCAN_FLAG --output "$AUX_DIR/eval_matrix_${b}.csv" \
    || { echo "[rebuild_board] per-base export FAILED ($b)" >&2; rc=1; }
done

# 4. STALENESS GUARD: every per-base file must be at least as new as the combined one. If a per-base
#    file is older, it will shadow the combined matrix on the board (the bug this script prevents).
comb_mtime=$(stat -c %Y "$AUX_DIR/eval_matrix.csv" 2>/dev/null || echo 0)
for b in "${BASES[@]}"; do
  pf="$AUX_DIR/eval_matrix_${b}.csv"
  [[ -f "$pf" ]] || continue
  pm=$(stat -c %Y "$pf")
  if (( pm < comb_mtime )); then
    echo "[rebuild_board] WARN: $pf is OLDER than combined matrix -> may shadow fixes on the board" >&2
    rc=1
  fi
done

# 5. RECOMPILE the board.
echo "[rebuild_board] compile master board"
"$VPT_PY" "$COMPILER" || { echo "[rebuild_board] compile FAILED" >&2; rc=1; }

# 6. AFTER diff: which oks_image (the column most prone to silent restatement) changed, and how many.
if [[ -f "$prev" ]]; then
  "$VPT_PY" - "$prev" "$AUX_DIR/eval_matrix.csv" <<'PY'
import csv, sys
def load(p):
    out={}
    with open(p) as f:
        for r in csv.DictReader(f):
            out[(r.get('run_id',''), r.get('model',''))]=r
    return out
a,b=load(sys.argv[1]),load(sys.argv[2])
cols=set()
for k in set(a)&set(b):
    for c in b[k]:
        if a[k].get(c,'')!=b[k].get(c,''):
            cols.add(c)
added=len(set(b)-set(a)); removed=len(set(a)-set(b))
print(f"[rebuild_board] AFTER diff: rows {len(a)}->{len(b)} (+{added}/-{removed}); changed columns: {sorted(cols) or 'none'}")
for c in sorted(cols):
    n=sum(1 for k in set(a)&set(b) if a[k].get(c,'')!=b[k].get(c,''))
    print(f"               · {c}: {n} rows changed")
PY
  rm -f "$prev"
fi

[[ $rc == 0 ]] && echo "[rebuild_board] DONE (board consistent)" || echo "[rebuild_board] DONE WITH WARNINGS (rc=$rc)"
exit "$rc"
