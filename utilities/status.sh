#!/usr/bin/env bash
# status.sh — one-shot cluster state: vLLM servers + SLURM jobs + /home space.
#
# Usage:
#   bash ~/utilities/status.sh          # default view
#   bash ~/utilities/status.sh --full   # also show vLLM concurrency + token limits

set -euo pipefail

FULL=0
for arg in "$@"; do
  case "$arg" in --full|-f) FULL=1 ;; esac
done

BOLD=$'\e[1m'; DIM=$'\e[2m'; RESET=$'\e[0m'
GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; CYAN=$'\e[36m'

sep() { printf '%s\n' "${DIM}────────────────────────────────────────${RESET}"; }

# ── 0. Node ──────────────────────────────────────────────────────────────────
echo
echo "${BOLD}${CYAN}── Cluster status $(date '+%H:%M:%S') ──${RESET}"
echo "  node: $(hostname)"
sep

# ── 1. vLLM servers ──────────────────────────────────────────────────────────
echo "${BOLD}vLLM servers${RESET}"

# Get all nodes with running SLURM jobs (these are where vLLM may be running)
job_nodes=$(squeue -u sgsilva --format="%T %R" --noheader 2>/dev/null \
  | awk '$1=="RUNNING" {print $2}' | sort -u || true)

# Also check localhost in case we're already on a worker
all_nodes=$(echo -e "localhost\n${job_nodes}" | sort -u | grep -v '^$')

found_vllm=0
while IFS= read -r node; do
  # Find vLLM processes on this node (ssh for remote, direct for localhost)
  if [ "$node" = "localhost" ] || [ "$node" = "$(hostname)" ]; then
    vllm_procs=$(ps aux 2>/dev/null \
      | grep -E 'vllm\.entrypoints|vllm serve|vllm_server' \
      | grep -v grep | grep sgsilva || true)
  else
    vllm_procs=$(ssh -o ConnectTimeout=3 -o BatchMode=yes "$node" \
      "ps aux 2>/dev/null | grep -E 'vllm\.entrypoints|vllm serve|vllm_server' | grep -v grep | grep sgsilva" \
      2>/dev/null || true)
  fi

  [ -z "$vllm_procs" ] && continue
  found_vllm=1

  while IFS= read -r line; do
    pid=$(echo "$line" | awk '{print $2}')
    port=$(echo "$line" | grep -oE '\-\-port ([0-9]{4,5})' | grep -oE '[0-9]{4,5}' | head -1)
    [ -z "$port" ] && port="8000"

    model_info=$(curl -s --max-time 3 "http://${node}:${port}/v1/models" 2>/dev/null \
      | python3 -c '
import sys, json
try:
  d = json.load(sys.stdin)
  m = d["data"][0]
  mid = m["id"]
  mlen = m.get("max_model_len", "?")
  parts = mid.rstrip("/").split("/")
  short = "/".join(parts[-2:]) if len(parts) >= 2 else mid
  print(f"{short}  max_len={mlen}")
except Exception:
  print("(not ready)")
' 2>/dev/null || echo "(curl failed)")

    echo "  ${node}:${port}  pid=${pid}  ${GREEN}${model_info}${RESET}"

    if [ "$FULL" = "1" ]; then
      metrics=$(curl -s --max-time 3 "http://${node}:${port}/metrics" 2>/dev/null || true)
      if [ -n "$metrics" ]; then
        running=$(echo "$metrics" | grep '^vllm:num_requests_running' | awk '{printf "%.0f", $2}' || echo "?")
        waiting=$(echo "$metrics" | grep '^vllm:num_requests_waiting' | awk '{printf "%.0f", $2}' || echo "?")
        gpu_util=$(echo "$metrics" | grep '^vllm:gpu_cache_usage_perc' | awk '{printf "%.1f%%", $2*100}' || echo "?")
        echo "         running=${running}  waiting=${waiting}  gpu_kv_cache=${gpu_util}"
      fi
    fi
  done <<< "$vllm_procs"
done <<< "$all_nodes"

[ "$found_vllm" = "0" ] && echo "  ${DIM}none running${RESET}"

sep

# ── 2. SLURM jobs (own) ──────────────────────────────────────────────────────
echo "${BOLD}SLURM jobs (sgsilva)${RESET}"

slurm_out=$(squeue -u sgsilva --format="%.10i %.20j %.8T %.10l %.6D %R" 2>/dev/null || true)
if [ -z "$slurm_out" ] || [ "$(echo "$slurm_out" | wc -l)" -le 1 ]; then
  echo "  ${DIM}no jobs in queue${RESET}"
else
  # Print with colour: RUNNING=green, PENDING=yellow, other=dim
  echo "$slurm_out" | head -1 | awk '{printf "  %s\n", $0}'
  echo "$slurm_out" | tail -n +2 | while IFS= read -r row; do
    state=$(echo "$row" | awk '{print $3}')
    case "$state" in
      RUNNING)  color=$GREEN ;;
      PENDING)  color=$YELLOW ;;
      *)        color=$DIM ;;
    esac
    echo "  ${color}${row}${RESET}"
  done
fi

sep

echo "  ${DIM}(to see your own usage: bash ~/utilities/cleanup/cleanup_home.sh)${RESET}"
echo

