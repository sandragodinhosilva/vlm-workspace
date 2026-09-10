#!/usr/bin/env bash
# SECOND GLM-5.3 instance (2026-09-09, dormancy v3 labelling was decode-bound on the first). Same recipe, publishes run/endpoint_b.env.
# Dedicated GLM-5.3 (FP8, TP=8) on one B300 node via start_vllm_server.sh, published as
# run/endpoint_b.env for client jobs. Modeled on serve/kimi3/run/kimi-k3.sbatch.
#
# Why a dedicated server: the shared worker-31 LB gave ~29.5 s/edge-call under load and its
# retries; a labeler client here runs at RF_MAX_WORKERS=16 without loading anyone else.
# Why sbatch: anything launched from the Claude pod dies with the pod (2026-09-01 19:27 restart
# killed a detached labeler). A SLURM job is independent of the session.
#
#   submit:  sbatch /home/sgsilva/utilities/serve/glm53/serve_glm53.sbatch
#   stop:    scancel <jobid>   (trap tears vllm down and removes endpoint.env)
#SBATCH --job-name=serve-glm53-b
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --mem=2400G
#SBATCH --exclude=worker-[30-31],worker-0
#SBATCH --output=/mnt/data/sgsilva/logs/serve/slurm-serve-glm53-b-%j.out

set -uo pipefail
source /home/sgsilva/utilities/logs-utils/log_run.sh
log_start serve glm53_b_second_instance

NODE="${SLURMD_NODENAME:-$(hostname -s)}"
PORT=9000
BASE_URL="http://$NODE:$PORT/v1"
HEALTH_URL="http://$NODE:$PORT/health"
ENDPOINT_FILE=/home/sgsilva/utilities/serve/glm53/run/endpoint_b.env

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

echo "[$(date -Is)] node=$NODE  serving glm-5.3 (FP8, TP=8, max-model-len 262144) on port $PORT"
echo "[$(date -Is)] vllm output -> $_LOG_RUN_FILE"

# Knobs are the ones Sandra used for glm-5.2-fp8 (bash_history): the glm52 serving venv
# (vllm 0.23.0, proven on the identical GlmMoeDsaForCausalLM/fp8 arch) and 128 concurrent seqs.
export QWEN35_VENV=/home/sgsilva/glm52-serving-home-venv
export MAX_NUM_SEQS=128
export STARTUP_HEARTBEAT_SECS=60
/home/sgsilva/utilities/serve/start_vllm_server.sh glm-5.3 8 262144 9000 >> "$_LOG_RUN_FILE" 2>&1 &
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
  echo "MODEL=glm-5.3"
} >"$ENDPOINT_FILE"
echo "[$(date -Is)] endpoint ready: $BASE_URL  (model id: glm-5.3)"
echo "[$(date -Is)] wrote $ENDPOINT_FILE"
curl -s "$BASE_URL/models" | head -c 400; echo

wait "$VLLM_PID"
