#!/usr/bin/env bash
#
# Monitor the current Kimi-K3 workload by scraping vLLM's Prometheus /metrics endpoint.
# Shows how many requests are running vs. waiting in the queue, plus KV-cache usage.
#
# Usage:
#   ./monitor.sh                 # one-shot snapshot
#   ./monitor.sh --watch         # refresh every 2s until Ctrl-C
#   ./monitor.sh --watch 5       # refresh every 5s

set -euo pipefail

KIMI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$KIMI_DIR/lib.sh"

WATCH=0
INTERVAL=2
if [[ "${1:-}" == "--watch" ]]; then
  WATCH=1
  [[ -n "${2:-}" ]] && INTERVAL="$2"
fi

BASE_URL="$(resolve_base_url)" || exit 1
METRICS_URL="${BASE_URL%/v1}/metrics"

# Sum a gauge across all label sets (e.g. multiple engines), printing "n/a" if absent.
sum_metric() {
  local body="$1" name="$2"
  awk -v m="$name" '
    { if (substr($0,1,length(m))==m) {
        c = substr($0, length(m)+1, 1);
        if (c=="{" || c==" ") { s+=$NF; found=1 }
      } }
    END { if (found) print s+0; else print "n/a" }' <<<"$body"
}

snapshot() {
  local body running waiting kv
  body="$(curl -sf "$METRICS_URL" 2>/dev/null || true)"
  if [[ -z "$body" ]]; then
    echo "$(date '+%H:%M:%S')  could not reach $METRICS_URL" >&2
    return 1
  fi
  running="$(sum_metric "$body" vllm:num_requests_running)"
  waiting="$(sum_metric "$body" vllm:num_requests_waiting)"
  kv="$(sum_metric "$body" vllm:kv_cache_usage_perc)"
  # KV usage is a 0..1 fraction; show as a percentage when numeric.
  if [[ "$kv" != "n/a" ]]; then
    kv="$(awk -v v="$kv" 'BEGIN{ printf "%.1f%%", v*100 }')"
  fi
  printf '%s  running=%s  waiting(queue)=%s  kv_cache=%s\n' \
    "$(date '+%H:%M:%S')" "$running" "$waiting" "$kv"
}

echo "metrics: $METRICS_URL"
if [[ "$WATCH" -eq 1 ]]; then
  echo "watching every ${INTERVAL}s (Ctrl-C to stop)"
  while true; do
    snapshot || true
    sleep "$INTERVAL"
  done
else
  snapshot
fi
