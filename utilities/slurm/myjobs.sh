#!/usr/bin/env bash
# myjobs.sh — your SLURM jobs with a live log tail, so you see PROGRESS not just "RUNNING".
#
# For each of sgsilva's jobs: jobid · state · node(s) · runtime · name, and for running
# jobs the last line of its most-recent centralized log (/mnt/data/sgsilva/logs/).
#
# Usage:
#   bash ~/utilities/myjobs.sh          # all your jobs + last log line each
#   bash ~/utilities/myjobs.sh -n 3     # last 3 log lines per job
set -euo pipefail

NLINES=1
for ((i=1; i<=$#; i++)); do
  case "${!i}" in -n) j=$((i+1)); NLINES="${!j}" ;; esac
done

BOLD=$'\e[1m'; DIM=$'\e[2m'; RESET=$'\e[0m'
GREEN=$'\e[32m'; YELLOW=$'\e[33m'; CYAN=$'\e[36m'; MAGENTA=$'\e[35m'
LOGROOT=/mnt/data/sgsilva/logs

state_color() { case "$1" in RUNNING) printf '%s' "$GREEN";; PENDING) printf '%s' "$YELLOW";;
  COMPLETING|CONFIGURING) printf '%s' "$CYAN";; *) printf '%s' "$MAGENTA";; esac; }

rows=$(squeue -u sgsilva -h -O 'JobID:.10,State:.12,Name:.40,RunTime:.14,NodeList:.0' 2>/dev/null || true)
if [[ -z "${rows// }" ]]; then echo "${DIM}no jobs for sgsilva${RESET}"; exit 0; fi

echo "${BOLD}${CYAN}── my jobs $(date '+%H:%M:%S') ──${RESET}"
while IFS= read -r line; do
  [[ -z "${line// }" ]] && continue
  jid=$(awk '{print $1}' <<<"$line"); st=$(awk '{print $2}' <<<"$line")
  name=$(awk '{print $3}' <<<"$line"); rt=$(awk '{print $4}' <<<"$line")
  nodes=$(awk '{print $5}' <<<"$line")
  printf '%s%-9s%s %s%-11s%s %s %s%s%s %s\n' \
    "$BOLD" "$jid" "$RESET" "$(state_color "$st")" "$st" "$RESET" \
    "${nodes:-—}" "$DIM" "$rt" "$RESET" "$name"
  if [[ "$st" == RUNNING ]]; then
    # Find the newest log whose name OR body mentions this job id; fall back to newest by name match.
    log=$(grep -rlswF "$jid" "$LOGROOT" --include='*.log' 2>/dev/null \
            | xargs -r ls -t 2>/dev/null | head -1 || true)
    if [[ -n "$log" ]]; then
      tail -n "$NLINES" "$log" 2>/dev/null | sed "s/^/    ${DIM}│${RESET} /"
      printf '    %s└ %s%s\n' "$DIM" "$log" "$RESET"
    else
      printf '    %s│ (no matching log under %s)%s\n' "$DIM" "$LOGROOT" "$RESET"
    fi
  fi
done <<<"$rows"
