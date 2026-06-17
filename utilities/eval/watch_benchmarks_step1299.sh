#!/usr/bin/env bash
# Notify when the step1299 benchmark run finishes. Completion signal = the Video-MME rating
# JSON (last of the 3 benchmarks). Read-only; logs to a sidecar; exits when done or after a
# safety cap. Does NOT touch any job/process (worker-28 is yours; the SFT job 92391 owns it).
set -uo pipefail
MODEL_DIR=qwen35-4b-mix-12k-1506-sft-step1299-thinkoff
VMME=/home/sgsilva/benchmarks/results/video_mme/$MODEL_DIR
MASTER=/mnt/data/sgsilva/results/master/eval_master.csv
LOG=/mnt/data/sgsilva/logs/watch_benchmarks_step1299.log
MAX_HOURS=12
POLL=300

log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
log "watcher start: waiting for Video-MME rating JSON under $VMME"

deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
while true; do
  rating=$(find "$VMME" -name "*Video-MME_rating.json" 2>/dev/null | head -1)
  if [[ -n "$rating" ]]; then
    log "DONE: Video-MME rating found -> $rating"
    # report the headline numbers from summary.csv (raw; judged only if a judge pass ran)
    for s in summary.csv summary_judge.csv; do
      row=$(grep -i "step1299" /home/sgsilva/benchmarks/results/$s 2>/dev/null | head -1)
      [[ -n "$row" ]] && log "  $s: $row"
    done
    log "  master CSV: $MASTER (re-run compile_eval_results.py to refresh the joined row)"
    break
  fi
  (( $(date +%s) > deadline )) && { log "TIMEOUT after ${MAX_HOURS}h — benchmarks still running or stalled; check manually."; break; }
  sleep "$POLL"
done
log "watcher exit"
