#!/usr/bin/env bash
# Shared helpers for the Kimi-K3 serving scripts. Meant to be *sourced*, not executed.
#
# Provides resolve_base_url, which figures out the OpenAI-compatible base URL of the running
# vllm server (e.g. http://worker-5:9000/v1) and echoes it on stdout. Resolution order:
#   1. run/endpoint.env  -- written by serve.sh once the server passed its health check.
#   2. squeue fallback   -- if the endpoint file is missing/stale, look up the node the
#                           `kimi-k3` slurm job is running on and build the URL from PORT.
# On failure it prints a helpful message to stderr and returns non-zero.

# Directory containing this library (= servers/kimi3), regardless of the caller's CWD.
KIMI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENDPOINT_FILE="$KIMI_DIR/run/endpoint.env"

# Must match serve.sh.
JOB_NAME="${JOB_NAME:-kimi-k3}"
PORT="${PORT:-9000}"

resolve_base_url() {
  # Explicit override always wins (handy for testing against another host).
  if [[ -n "${KIMI_BASE_URL:-}" ]]; then
    echo "$KIMI_BASE_URL"
    return 0
  fi

  # Slurm is the source of truth for "is a server actually running, and where".
  # (This runs in a command-substitution subshell, so sourcing the file below can't
  # leak variables back to the caller.)
  local running_node running_jobid
  running_node="$(squeue --name "$JOB_NAME" --states=R -h -o '%N' 2>/dev/null | head -n1)"
  running_jobid="$(squeue --name "$JOB_NAME" --states=R -h -o '%A' 2>/dev/null | head -n1)"

  # 1) Prefer the endpoint file serve.sh wrote -- but only if its JOBID is the job that is
  #    actually running now. This makes a stale file (e.g. left by a hard node crash that
  #    skipped the cleanup trap) harmless: it is simply ignored.
  if [[ -f "$ENDPOINT_FILE" ]]; then
    local BASE_URL="" JOBID=""
    # shellcheck disable=SC1090
    source "$ENDPOINT_FILE"
    if [[ -n "$BASE_URL" && -n "$running_jobid" && "$JOBID" == "$running_jobid" ]]; then
      echo "$BASE_URL"
      return 0
    fi
  fi

  # 2) Fall back to constructing the URL from the node the running job is on.
  if [[ -n "$running_node" ]]; then
    echo "http://${running_node}:${PORT}/v1"
    return 0
  fi

  echo "Could not resolve the Kimi-K3 endpoint: no running slurm job named '$JOB_NAME'." >&2
  echo "  (check: squeue --name $JOB_NAME)" >&2
  echo "Start the server first with: $KIMI_DIR/serve.sh" >&2
  return 1
}
