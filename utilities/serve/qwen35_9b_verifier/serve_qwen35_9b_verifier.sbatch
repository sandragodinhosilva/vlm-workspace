#!/usr/bin/env bash
# Qwen3.5-9B as the MEDICATION-CHECK VERIFIER for the 3wc medication-elicitation screen,
# thinking ENABLED, published as run/endpoint.env. Modeled on serve/glm53/serve_glm53.sbatch.
#
# WHY A DIFFERENT FAMILY: this verifier produces the `label` that `k` counts. The elicitation
# mutator and the training judge are both GLM, and elicited rows are selected partly for what
# GLM finds unobjectionable -- so the instrument producing `k` must sit OUTSIDE that family or
# it colludes with what it is meant to check. Qwen is the decided choice (brief Step 0c).
#
# WHY THINKING ON: the medication check asks for reasoning before the label, and agreement
# measured under one thinking mode does not transfer to the other. start_vllm_server.sh
# defaults Qwen3.5 to thinking OFF, so ENABLE_THINKING=1 is REQUIRED here, not tuning.
#
# WHY sbatch: a setsid/ssh launch from the Claude pod dies with the pod, and on this cluster it
# also loses the venv bin from PATH -- vLLM's Triton JIT then dies with
# "[Errno 2] No such file or directory: 'ninja'" ~2 min in, after the weights have loaded
# (observed twice, 2026-09-04). A batch job inherits the login environment and does not.
#
#   submit:  sbatch /home/sgsilva/utilities/serve/qwen35_9b_verifier/serve_qwen35_9b_verifier.sbatch
#   stop:    scancel <jobid>   (trap tears vllm down and removes endpoint.env)
#SBATCH --job-name=serve-9b-verifier
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --exclude=worker-[30-31],worker-0
#SBATCH --output=/mnt/data/sgsilva/logs/serve/slurm-serve-9b-verifier-%j.out

set -uo pipefail
source /home/sgsilva/utilities/logs-utils/log_run.sh
log_start serve qwen35_9b_medication_verifier

NODE="${SLURMD_NODENAME:-$(hostname -s)}"
PORT=9010
BASE_URL="http://$NODE:$PORT/v1"
HEALTH_URL="http://$NODE:$PORT/health"
ENDPOINT_FILE=/home/sgsilva/utilities/serve/qwen35_9b_verifier/run/endpoint.env

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

echo "[$(date -Is)] node=$NODE  serving Qwen3.5-9B verifier (TP=2, max-model-len 65536, thinking ON) on port $PORT"
echo "[$(date -Is)] vllm output -> $_LOG_RUN_FILE"

export ENABLE_THINKING=1
export MAX_NUM_SEQS=64
export STARTUP_HEARTBEAT_SECS=60
/home/sgsilva/utilities/serve/start_vllm_server.sh /mnt/data/shared/models/Qwen3.5-9B 2 65536 "$PORT" >> "$_LOG_RUN_FILE" 2>&1 &
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
  echo "MODEL=/mnt/data/shared/models/Qwen3.5-9B"
  echo "THINKING=1"
} >"$ENDPOINT_FILE"
echo "[$(date -Is)] endpoint ready: $BASE_URL"
echo "[$(date -Is)] wrote $ENDPOINT_FILE"
curl -s "$BASE_URL/models" | head -c 400; echo

wait "$VLLM_PID"
