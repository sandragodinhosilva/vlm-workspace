#!/usr/bin/env bash
# oracle_status.sh — at-a-glance progress of every running VObs oracle generation run
# (generate_visual_observations_human.py), plus today's finished ones. Read-only.
#
#   ~/utilities/logs-utils/oracle_status.sh
#   watch -n 30 ~/utilities/logs-utils/oracle_status.sh    # live
#
# For each run: schema file · GT-on/off · K · question-mode · reps written / total ·
# throughput (reps/min) · ETA. Reads the dated oracle run-logs (Collected N tasks, started)
# + counts reps in the run's --output-file JSON. Mirrors eval_status.sh.
set -uo pipefail
LOG_ROOT="/mnt/data/sgsilva/logs/oracle"
PYBIN="/home/sgsilva/vlm-post-training-home-venv/bin/python"
BOLD=$'\e[1m'; DIM=$'\e[2m'; RESET=$'\e[0m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; CYAN=$'\e[36m'; RED=$'\e[31m'

now_epoch=$(date -u +%s)

# count top-level reps in an oracle output JSON (exid -> {rep -> {...}}) without loading via jq
_reps() {
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  "$PYBIN" - "$f" <<'PY' 2>/dev/null || echo 0
import json,sys
try:
    d=json.load(open(sys.argv[1])); print(sum(len(v) for v in d.values()) if isinstance(d,dict) else 0)
except Exception: print(0)
PY
}

# newest oracle log whose cmd contains a given --output-file path
_logfor_output() {
  local out="$1" d
  for d in "$(date -u +%F)" "$(date -u -d yesterday +%F 2>/dev/null)"; do
    grep -ls -- "--output-file $out" "$LOG_ROOT/$d"/*.log 2>/dev/null | while read -r L; do
      printf '%s\t%s\n' "$(stat -c %Y "$L")" "$L"
    done
  done | sort -rn | head -1 | cut -f2-
}

fmt_eta() { local m=$1; (( m < 0 )) && { echo "?"; return; }
  (( m < 90 )) && { echo "${m}m"; return; }
  printf "%.1fh\n" "$(echo "$m/60" | bc -l)"; }

echo
echo "${BOLD}${CYAN}── VObs oracle status $(date '+%H:%M:%S') ──${RESET}  node $(hostname)"
echo "${DIM}────────────────────────────────────────────────────────────${RESET}"

# discover oracle runs from the dated LOGS on shared storage (node-independent — the procs may run
# on any worker node, so local `ps` would miss them when this is run from a login node). Take each
# log's --output-file; a run is "active" if its output file was modified in the last few minutes.
ACTIVE_SECS=300
mapfile -t OUTS < <(
  for d in "$(date -u +%F)" "$(date -u -d yesterday +%F 2>/dev/null)"; do
    grep -hoE -- "--output-file [^ ]+" "$LOG_ROOT/$d"/*.log 2>/dev/null
  done | awk '{print $2}' | grep -E "/oracle_" | sort -u)

if (( ${#OUTS[@]} == 0 )); then
  echo "  ${DIM}no oracle generation run found in today/yesterday's logs${RESET}"
fi

# 397B server load (shared bottleneck) — fetched ONCE up front so the per-run state can use
# "server is running requests" as a secondary liveness signal (pre-first-write). Derive the host
# from the runs' own logs (--server-url) rather than hardcoding, so this follows worker-5/11/etc.
SRV_HOST=$(for d in "$(date -u +%F)" "$(date -u -d yesterday +%F 2>/dev/null)"; do
    grep -hoE -- "--server-url https?://[^ /:]+" "$LOG_ROOT/$d"/*.log 2>/dev/null
  done | grep -oE "://[^ /:]+" | tr -d ':/' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
SRV_HOST="${SRV_HOST:-worker-5}"
# Anchor to the GAUGE line (^vllm:num_requests_running{) and take the TRAILING number — the old
# loose grep matched a stray float elsewhere on the scrape and undercounted in-flight requests.
_metrics=$(curl -s -m 3 "http://${SRV_HOST}:8000/metrics" 2>/dev/null)
running=$(echo "$_metrics" | grep -E "^vllm:num_requests_running\{" | grep -oE "[0-9]+(\.[0-9]+)?$" | head -1)
waiting=$(echo "$_metrics" | grep -E "^vllm:num_requests_waiting\{" | grep -oE "[0-9]+(\.[0-9]+)?$" | head -1)

for out in "${OUTS[@]}"; do
  # skip runs that never produced an output file (killed/aborted before first write — not real runs)
  [[ -f "$out" ]] || continue
  log="$(_logfor_output "$out")"
  name="$(basename "$out" .json)"
  total=0; k="?"; qmode="?"; gt="GT-on"; started=""
  if [[ -n "$log" && -f "$log" ]]; then
    # TOTAL = tasks-remaining ("Collected N tasks") + reps carried over on resume ("resume-skipped M").
    # The banner reads "Collected 10616 tasks (resume-skipped 2100 ...)": 10616 is what's LEFT to do,
    # NOT the grand total — and done_n counts ALL reps in the file (incl. the 2100 carried over). Adding
    # them back is what makes the denominator the true total (e.g. 10616+2100 = 12716) and kills the
    # >100% overflow that falsely read as "done" (the 106%/107% bug, 2026-06-29).
    cline=$(grep -oE "Collected [0-9]+ tasks( \(resume-skipped [0-9]+)?" "$log" 2>/dev/null | tail -1)
    total=$(echo "$cline" | grep -oE "Collected [0-9]+" | grep -oE "[0-9]+")
    skipped=$(echo "$cline" | grep -oE "resume-skipped [0-9]+" | grep -oE "[0-9]+")
    [[ -n "$total" && -n "$skipped" ]] && total=$(( total + skipped ))
    # K from the cmd line (always present; the "Self-consistency: K=" banner only prints when K>1)
    k=$(grep -oE -- "--self-consistency [0-9]+" "$log" 2>/dev/null | head -1 | grep -oE "[0-9]+")
    qmode=$(grep -oE "Question mode: [a-z-]+" "$log" 2>/dev/null | head -1 | awk '{print $3}')
    grep -q -- "--withhold-gt" "$log" 2>/dev/null && gt="GT-off"
    started=$(grep -oE "started  : [0-9T:-]+" "$log" 2>/dev/null | head -1 | sed 's/started  : //')
  fi
  done_n=$(_reps "$out")
  [[ -z "$total" ]] && total=0

  # throughput + ETA from started timestamp
  rate=""; eta=""
  if [[ -n "$started" ]]; then
    s_epoch=$(date -u -d "${started/T/ }" +%s 2>/dev/null || echo 0)
    elapsed_min=$(( (now_epoch - s_epoch) / 60 ))
    if (( elapsed_min > 0 && done_n > 0 )); then
      rate=$(echo "scale=1; $done_n/$elapsed_min" | bc -l)
      remain=$(( total - done_n ))
      (( remain < 0 )) && remain=0
      eta_min=$(echo "$remain/($done_n/$elapsed_min)" | bc 2>/dev/null)
      eta=$(fmt_eta "${eta_min:--1}")
    fi
  fi

  pct=0; (( total > 0 )) && pct=$(( done_n * 100 / total ))
  # active/stalled/done from output-file mtime freshness (node-independent — shared storage).
  # BUT a run that just (re)started spends minutes in a resume-scan + K=5 first-batch BEFORE its
  # first write — that is NOT a stall. Two guards prevent the false "stalled" verdict:
  #   (a) total==0  → the "Collected N tasks" banner hasn't printed yet → still scanning → "starting".
  #   (b) the 397B server is actively running requests → the run is feeding it even pre-write → "active".
  state="${DIM}stalled?${RESET}"; mtime_age=99999
  if [[ -f "$out" ]]; then mtime_age=$(( now_epoch - $(stat -c %Y "$out") )); fi
  srv_running="${running:-0}"; srv_running="${srv_running%.*}"
  # Liveness, node-independent: the clog LOG on shared storage keeps appending while the run lives,
  # so a fresh log mtime means the proc is alive even if it runs on a worker this login can't ps.
  # (ps is a bonus positive signal when the run happens to share this node; never used to prove death.)
  proc_alive=0
  log_age=99999; [[ -n "$log" && -f "$log" ]] && log_age=$(( now_epoch - $(stat -c %Y "$log") ))
  (( log_age <= ACTIVE_SECS )) && proc_alive=1
  if ps -u "$(whoami)" -o cmd= 2>/dev/null | grep -F -- "--output-file $out" | grep -qv "grep"; then proc_alive=1; fi
  # "done" requires BOTH pct>=100 AND no live writes AND proc not seen alive. The stale-denominator
  # overflow (pct>100 from a re-counted resume) MUST NOT alone read as done — that was the 106% bug.
  if (( mtime_age <= ACTIVE_SECS )) || (( proc_alive == 1 )); then
    if (( pct >= 100 )); then
      # writing past the stale task count → denominator is stale, run is NOT finished.
      state="${YELLOW}● active${RESET} ${DIM}(pct>100 = stale task-count; still writing, ${mtime_age}s)${RESET}"
    else
      state="${GREEN}● active${RESET} ${DIM}(${mtime_age}s ago)${RESET}"
    fi
  elif (( pct >= 100 )); then state="${GREEN}done${RESET}"
  elif (( total == 0 )); then state="${YELLOW}◐ starting${RESET} ${DIM}(resume scan; no task count yet)${RESET}"
  elif [[ -n "$srv_running" ]] && (( srv_running > 0 )); then
    state="${GREEN}● active${RESET} ${DIM}(generating; ${srv_running} reqs in-flight, no write ${mtime_age}s)${RESET}"
  else state="${RED}● stalled${RESET} ${DIM}(no write ${mtime_age}s)${RESET}"; fi
  col=$GREEN; (( pct < 100 )) && col=$YELLOW
  echo "  ${BOLD}${name}${RESET}  ${state}"
  echo "    ${gt} · K=${k} · mode=${qmode}"
  echo "    progress: ${col}${done_n}/${total} (${pct}%)${RESET}  rate=${rate:-?} reps/min  ETA=${eta:-?}"
  [[ -n "$log" ]] && echo "    ${DIM}log: $log${RESET}"
done

echo "${DIM}────────────────────────────────────────────────────────────${RESET}"
# 397B server load (the shared bottleneck) — values fetched once up front (SRV_HOST/running/waiting).
[[ -n "$running" ]] && echo "  ${DIM}397B@${SRV_HOST}: ${running%.*} running / ${waiting%.*} waiting${RESET}"
echo
