#!/usr/bin/env bash
#SBATCH --job-name=kimi-k3
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=worker-[30-31]
#SBATCH --cpus-per-task=192
#SBATCH --output=/home/sgsilva/utilities/serve/kimi3/run/slurm-%j.log

set -uo pipefail
cd "/home/sgsilva/utilities/serve/kimi3"

# Put the project venv on PATH. We invoke vllm by absolute path, but its kernel-compilation
# step (torch cpp_extension) locates helper tools like 'ninja' via PATH -- without this it
# dies with "No such file or directory: 'ninja'".
export PATH="/home/sgsilva/utilities/serve/kimi3/.venv/bin:$PATH"

# Slurm sets SLURMD_NODENAME to the compute node's hostname (e.g. "worker-5"), which is
# reachable from the rest of the cluster -- we use it verbatim for the API base URL.
NODE="${SLURMD_NODENAME:-localhost}"
BASE_URL="http://$NODE:9000/v1"
HEALTH_URL="http://$NODE:9000/health"

# K3-specific vLLM tuning (kept from the upstream serve recipe).
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_V2_MODEL_RUNNER=1

VLLM_PID=""
cleanup() {
  echo "[$(date -Is)] cleanup: stopping vllm (pid ${VLLM_PID:-none}) ..."
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$VLLM_PID" 2>/dev/null || true
  fi
  rm -f "/home/sgsilva/utilities/serve/kimi3/run/endpoint.env"
  echo "[$(date -Is)] cleanup done"
}
# scancel sends SIGTERM (then SIGKILL after KillWait); also fire on normal EXIT / Ctrl-C.
trap cleanup TERM INT EXIT

echo "[$(date -Is)] node=$NODE  serving '/mnt/data/shared/models/Kimi-K3' as 'hosted_vllm/kimi-k3' on port 9000"
echo "[$(date -Is)] vllm stdout/stderr -> /home/sgsilva/utilities/serve/kimi3/run/vllm.log"

# Note on --attention-config: fp8 KV cache (--kv-cache-dtype fp8) requires an fp8 prefill query,
# which only the FlashInfer / TRT-LLM-Ragged / TokenSpeed MLA prefill backends support. On
# Blackwell the auto-selected prefill backend is FlashAttention (unsupported), so we pin
# mla_prefill_backend=FLASHINFER; otherwise vllm aborts warm-up with
# "Kimi-K3 fp8 KV cache requires an fp8 prefill query".
"/home/sgsilva/utilities/serve/kimi3/.venv/bin/vllm" serve "/mnt/data/shared/models/Kimi-K3" \
  --served-model-name "kimi-k3" \
  --host 0.0.0.0 \
  --port 9000 \
  --trust-remote-code \
  --moe-backend auto \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 8 \
  --load-format fastsafetensors \
  --no-enable-flashinfer-autotune \
  --max-model-len 1048576 \
  --kv-cache-dtype fp8 \
  --attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"FLASHINFER"}' \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser kimi_k3 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  >"/home/sgsilva/utilities/serve/kimi3/run/vllm.log" 2>&1 &
VLLM_PID=$!

echo "[$(date -Is)] waiting for health at $HEALTH_URL (timeout 3600s) ..."
UP=0
for i in $(seq 1 3600); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date -Is)] !!! vllm exited during startup. Last 200 lines of /home/sgsilva/utilities/serve/kimi3/run/vllm.log:"
    tail -n 200 "/home/sgsilva/utilities/serve/kimi3/run/vllm.log" || true
    exit 1
  fi
  if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    UP=1
    echo "[$(date -Is)] vllm healthy after ${i}s"
    break
  fi
  sleep 1
done
if [[ "$UP" -ne 1 ]]; then
  echo "[$(date -Is)] !!! vllm not healthy within 3600s. Last 200 lines of /home/sgsilva/utilities/serve/kimi3/run/vllm.log:"
  tail -n 200 "/home/sgsilva/utilities/serve/kimi3/run/vllm.log" || true
  exit 1
fi

# Publish the resolved endpoint for the helper scripts.
mkdir -p "/home/sgsilva/utilities/serve/kimi3/run"
{
  echo "BASE_URL=$BASE_URL"
  echo "NODE=$NODE"
  echo "PORT=9000"
  echo "JOBID=${SLURM_JOB_ID:-unknown}"
} >"/home/sgsilva/utilities/serve/kimi3/run/endpoint.env"
echo "[$(date -Is)] endpoint ready: $BASE_URL  (model: hosted_vllm/kimi-k3)"
echo "[$(date -Is)] wrote /home/sgsilva/utilities/serve/kimi3/run/endpoint.env"

# Stay alive alongside vllm; the trap handles teardown on scancel/exit.
wait "$VLLM_PID"
