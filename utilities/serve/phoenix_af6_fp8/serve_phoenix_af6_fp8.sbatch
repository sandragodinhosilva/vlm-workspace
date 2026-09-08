#!/usr/bin/env bash
# Phoenix-Alpha-27B (af6 step_100, FP8) as the POLICY for the 3wc medication screen/elicitation,
# thinking ENABLED, reasoning_effort medium, published as run/endpoint.env.
# Modeled on serve/qwen35_9b_verifier/serve_qwen35_9b_verifier.sbatch.
#
# The flags reproduce the 2026-09-04 screen server byte-for-byte (log
# serve_Phoenix-27B-Alpha-FP8__worker-26_p9002_3659945.out): TP=8, max-model-len 262144,
# --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml
# --generation-config vllm --enable-prefix-caching, reasoning_effort medium, MAX_NUM_SEQS 64,
# venv glm52-serving-home-venv (transformers 5 + vLLM 0.23.0; see serve_3wc_ckpt.sh for why).
# TP=8 is kept for comparability with the 150 screened rollouts, not because the 27B needs it.
#
# WHY sbatch: a setsid/ssh launch from the Claude pod loses the venv bin from PATH and vLLM's
# Triton JIT dies ~2 min in, after the weights load (observed 2026-09-04).
#
#   submit:  sbatch /home/sgsilva/utilities/serve/phoenix_af6_fp8/serve_phoenix_af6_fp8.sbatch
#   stop:    scancel <jobid>   (trap tears vllm down and removes endpoint.env)
#SBATCH --job-name=serve-27b-policy
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
source /home/sgsilva/utilities/logs-utils/log_run.sh
log_start serve phoenix_af6_fp8_policy

NODE="${SLURMD_NODENAME:-$(hostname -s)}"
PORT=9002
MODEL_PATH=/mnt/data/sgsilva/3wc/phoenix_fp8_mtp/Phoenix-27B-Alpha-FP8
BASE_URL="http://$NODE:$PORT/v1"
HEALTH_URL="http://$NODE:$PORT/health"
ENDPOINT_FILE=/home/sgsilva/utilities/serve/phoenix_af6_fp8/run/endpoint.env

VLLM_PID=""
cleanup() {
  echo "[$(date -Is)] cleanup: stopping vllm (pid ${VLLM_PID:-none}) ..."
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$VLLM_PID" 2>/dev/null || true
  fi
  rm -f "$ENDPOINT_FILE"
  echo "[$(date -Is)] cleanup done"
  log_end
}
trap cleanup TERM INT EXIT

echo "[$(date -Is)] node=$NODE  serving $MODEL_PATH (TP=8, max-model-len 262144, thinking ON, effort medium) on port $PORT"
echo "[$(date -Is)] vllm output -> $_LOG_RUN_FILE"

export ENABLE_THINKING=1
export MAX_NUM_SEQS=64
export STARTUP_HEARTBEAT_SECS=60
export QWEN35_VENV=/home/sgsilva/glm52-serving-home-venv
export EXTRA_VLLM_ARGS='--language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml --generation-config vllm --enable-prefix-caching --default-chat-template-kwargs {"reasoning_effort":"medium"}'
/home/sgsilva/utilities/serve/start_vllm_server.sh "$MODEL_PATH" 8 262144 "$PORT" >> "$_LOG_RUN_FILE" 2>&1 &
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
  echo "THINKING=1"
} >"$ENDPOINT_FILE"
echo "[$(date -Is)] endpoint ready: $BASE_URL"
echo "[$(date -Is)] wrote $ENDPOINT_FILE"
curl -s "$BASE_URL/models" | head -c 400; echo

wait "$VLLM_PID"
