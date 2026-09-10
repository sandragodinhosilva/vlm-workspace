#!/bin/bash
# Serve ONE Qwen3.8-27B-family policy checkpoint (base or a GRPO step) exactly the way the af6
# policy is served for the eval sets -- same flags, thinking ON, reasoning_effort pinned to MEDIUM
# (the qwen38_medium.jinja every 3WC GRPO recipe trains against; the vendor base defaults to xhigh,
# so the pin is what makes a base arm comparable). Cloned 2026-09-10 from
# phoenix_af6_fp8/serve_phoenix_af6_fp8.sbatch and parameterised by environment:
#
#   MODEL_PATH         checkpoint dir (weights/step_N or the vendor snapshot)   REQUIRED
#   PORT               listen port                                              REQUIRED
#   SERVED_MODEL_NAME  short stable id registered on /v1/models                 REQUIRED
#   ENDPOINT_FILE      where BASE_URL/NODE/PORT/JOBID/MODEL land once healthy   REQUIRED
#   TP (8) MAX_LEN (262144) EFFORT (medium) MAX_NUM_SEQS (64)                   optional
#
#   submit:  cd /home/sgsilva/utilities/serve/qwen38_27b_policy && \
#            MODEL_PATH=... PORT=9003 SERVED_MODEL_NAME=... ENDPOINT_FILE=run/<name>.env \
#            sbatch --job-name=serve-27b-<terse> --export=ALL serve_qwen38_policy.sbatch
#   stop:    scancel <jobid>   (trap tears vllm down and removes the endpoint file)
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --mem=2400G
#SBATCH --time=12:00:00
#SBATCH --exclude=worker-[30-31],worker-0
#SBATCH --output=/mnt/data/sgsilva/logs/serve/slurm-serve-27b-policy-%j.out

set -uo pipefail
: "${MODEL_PATH:?MODEL_PATH required}" "${PORT:?PORT required}" "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME required}" "${ENDPOINT_FILE:?ENDPOINT_FILE required}"
TP="${TP:-8}"; MAX_LEN="${MAX_LEN:-262144}"; EFFORT="${EFFORT:-medium}"
source /home/sgsilva/utilities/logs-utils/log_run.sh
log_start serve "policy_${SERVED_MODEL_NAME}"

NODE="${SLURMD_NODENAME:-$(hostname -s)}"
BASE_URL="http://$NODE:$PORT/v1"
HEALTH_URL="http://$NODE:$PORT/health"

# torch.compile cache MUST be node-local. The default ~/.cache/vllm lives on shared NFS and the
# cache key is per-ARCHITECTURE, so two servers of the same family started together land in the
# SAME directory and deadlock in torch.compile: no writes, no progress, just shm_broadcast "No
# available shared memory broadcast block" once a minute forever (5 servers wedged this way
# 2026-09-10, jobs 157312-157317, ~45 min each before they were killed; weights had loaded in 23s).
# /tmp is node-local ext4, so each job compiles into its own dir and concurrent launches are safe.
# Same reasoning as TRITON_CACHE_DIR in the prime-rl training sbatch.
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$VLLM_CACHE_ROOT"
_cleanup_cache() { rm -rf "/tmp/vllm_cache_${SLURM_JOB_ID:-$$}" 2>/dev/null || true; }

VLLM_PID=""
cleanup() {
  echo "[$(date -Is)] cleanup: stopping vllm (pid ${VLLM_PID:-none}) ..."
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$VLLM_PID" 2>/dev/null || true
  fi
  rm -f "$ENDPOINT_FILE"
  _cleanup_cache 2>/dev/null || true
  echo "[$(date -Is)] cleanup done"
  log_end
}
trap cleanup TERM INT EXIT

echo "[$(date -Is)] node=$NODE  serving $MODEL_PATH as '$SERVED_MODEL_NAME' (TP=$TP, max-model-len $MAX_LEN, thinking ON, effort $EFFORT) on port $PORT"
echo "[$(date -Is)] vllm output -> $_LOG_RUN_FILE"

export ENABLE_THINKING=1
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export STARTUP_HEARTBEAT_SECS=60
export QWEN35_VENV=/home/sgsilva/glm52-serving-home-venv
export SERVED_MODEL_NAME
export EXTRA_VLLM_ARGS="--language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml --generation-config vllm --enable-prefix-caching --default-chat-template-kwargs {\"reasoning_effort\":\"$EFFORT\"}"
/home/sgsilva/utilities/serve/start_vllm_server.sh "$MODEL_PATH" "$TP" "$MAX_LEN" "$PORT" >> "$_LOG_RUN_FILE" 2>&1 &
VLLM_PID=$!

echo "[$(date -Is)] waiting for health at $HEALTH_URL (timeout 3600s) ..."
UP=0
for i in $(seq 1 3600); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date -Is)] !!! vllm exited during startup. Last 100 lines of $_LOG_RUN_FILE:"
    tail -n 100 "$_LOG_RUN_FILE" || true
    exit 1
  fi
  if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    UP=1; echo "[$(date -Is)] vllm healthy after ${i}s"; break
  fi
  sleep 1
done
if [[ "$UP" -ne 1 ]]; then
  echo "[$(date -Is)] !!! vllm not healthy within 3600s. Last 100 lines of $_LOG_RUN_FILE:"
  tail -n 100 "$_LOG_RUN_FILE" || true
  exit 1
fi

{
  echo "BASE_URL=$BASE_URL"
  echo "NODE=$NODE"
  echo "PORT=$PORT"
  echo "JOBID=${SLURM_JOB_ID:-unknown}"
  echo "MODEL=$MODEL_PATH"
  echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
  echo "EFFORT=$EFFORT"
  echo "THINKING=1"
} >"$ENDPOINT_FILE"
echo "[$(date -Is)] endpoint ready: $BASE_URL"
echo "[$(date -Is)] wrote $ENDPOINT_FILE"
curl -s "$BASE_URL/models" | head -c 400; echo

wait "$VLLM_PID"
