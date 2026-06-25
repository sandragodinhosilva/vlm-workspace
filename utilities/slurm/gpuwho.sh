#!/usr/bin/env bash
# gpuwho.sh — what's actually eating a node's GPUs: process → user → GPU-mem.
#
# Use when gpuq flags a rogue ◆ and you want to know WHOSE it is (and whether it's yours)
# BEFORE doing anything. Read-only: this only inspects — it never kills.
# Respects the never-stop-other-users'-processes rule: it tells you the owner so you can decide.
#
# Usage:
#   bash ~/utilities/gpuwho.sh worker-30
#   bash ~/utilities/gpuwho.sh worker-30 worker-31
set -euo pipefail
[[ $# -eq 0 ]] && { echo "usage: gpuwho.sh <node> [node...]"; exit 2; }

BOLD=$'\e[1m'; DIM=$'\e[2m'; RESET=$'\e[0m'; CYAN=$'\e[36m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'
ME=$(id -un)
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ConnectTimeout=5)

for node in "$@"; do
  echo "${BOLD}${CYAN}── $node ──${RESET}"
  # compute-apps gives pid+gpu-mem; map pid→user/cmd via ps on the same host, in one ssh.
  out=$(ssh "${SSH_OPTS[@]}" "$node" '
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | while IFS=, read -r uuid pid mem; do
        pid=$(echo "$pid" | tr -d " ")
        [ -z "$pid" ] && continue
        info=$(ps -o user=,comm= -p "$pid" 2>/dev/null)
        printf "%s\t%s\t%s\n" "$pid" "${mem# } MB" "$info"
      done' 2>/dev/null || true)
  if [[ -z "${out// }" ]]; then
    echo "  ${DIM}no compute processes (idle, or ssh/nvidia-smi unavailable)${RESET}"; continue
  fi
  printf '  %s%-8s %-10s %-12s %s%s\n' "$DIM" "PID" "GPU-MEM" "USER" "CMD" "$RESET"
  while IFS=$'\t' read -r pid mem rest; do
    user=$(awk '{print $1}' <<<"$rest"); cmd=$(awk '{print $2}' <<<"$rest")
    if [[ "$user" == "$ME" ]]; then tag="${GREEN}(you)${RESET}"; else tag="${YELLOW}(someone else — do NOT kill)${RESET}"; fi
    printf '  %-8s %-10s %-12s %s  %b\n' "$pid" "$mem" "${user:-?}" "${cmd:-?}" "$tag"
  done <<<"$out"
done
