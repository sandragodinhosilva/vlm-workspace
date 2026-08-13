#!/usr/bin/env bash
#
# Serve Kimi-K3 with vLLM on a single Slurm node (8x B300).
#
# Submits a Slurm batch job that:
#   1. launches `vllm serve` for the local Kimi-K3 checkpoint,
#   2. waits until the OpenAI-compatible server is healthy,
#   3. publishes the endpoint to run/endpoint.env (consumed by health.sh / monitor.sh /
#      check_response.py),
#   4. tears vLLM down cleanly when the job is cancelled (scancel) or ends.
#
# Once up, the model is reachable as:  model=hosted_vllm/kimi-k3  base_url=http://<node>:9000/v1
#
# Usage:
#   ./serve.sh                 # submit the job
#   ./serve.sh --print-only    # print the generated sbatch script and exit (no submission)
#
# Watch progress:  tail -f run/slurm-<jobid>.log   (startup of a 1M-context model can take a while)
# Stop:            scancel <jobid>                  (everything is cleaned up automatically)

set -euo pipefail

KIMI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration (edit here) ---------------------------------------------
MODEL_PATH="/mnt/data/shared/models/Kimi-K3"   # local checkpoint (already downloaded)
SERVED_NAME="kimi-k3"                           # OpenAI model id clients use (hosted_vllm/<this>)
PORT=9000
PARTITION="main"
GRES="gpu:8"                                    # all 8 B300s on the node (tensor-parallel-size 8)
EXCLUDE_NODES="worker-[30-31]"                  # team policy: never use these nodes
CPUS=192                                         # whole node
STARTUP_TIMEOUT=3600                             # seconds to wait for the server to become healthy
# ---------------------------------------------------------------------------

RUN_DIR="$KIMI_DIR/run"
VLLM_LOG="$RUN_DIR/vllm.log"
ENDPOINT_FILE="$RUN_DIR/endpoint.env"
SLURM_LOG="$RUN_DIR/slurm-%j.log"        # %j is expanded by Slurm to the job id
VLLM_BIN="$KIMI_DIR/.venv/bin/vllm"
SBATCH_FILE="$RUN_DIR/kimi-k3.sbatch"

PRINT_ONLY=0
[[ "${1:-}" == "--print-only" ]] && PRINT_ONLY=1

[[ -x "$VLLM_BIN" ]] || { echo "vllm not found at $VLLM_BIN -- set up the .venv first (see README)." >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "model path not found: $MODEL_PATH" >&2; exit 1; }

mkdir -p "$RUN_DIR"

# Render the batch script. Build-time values ($MODEL_PATH, $PORT, ...) are expanded now;
# runtime references (\$NODE, \$VLLM_PID, \$(date) ...) are escaped so they resolve on the node.
cat >"$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$SERVED_NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=$GRES
#SBATCH --exclusive
#SBATCH --exclude=$EXCLUDE_NODES
#SBATCH --cpus-per-task=$CPUS
#SBATCH --output=$SLURM_LOG

set -uo pipefail
cd "$KIMI_DIR"

# Put the project venv on PATH. We invoke vllm by absolute path, but its kernel-compilation
# step (torch cpp_extension) locates helper tools like 'ninja' via PATH -- without this it
# dies with "No such file or directory: 'ninja'".
export PATH="$KIMI_DIR/.venv/bin:\$PATH"

# Slurm sets SLURMD_NODENAME to the compute node's hostname (e.g. "worker-5"), which is
# reachable from the rest of the cluster -- we use it verbatim for the API base URL.
NODE="\${SLURMD_NODENAME:-localhost}"
BASE_URL="http://\$NODE:$PORT/v1"
HEALTH_URL="http://\$NODE:$PORT/health"

# K3-specific vLLM tuning (kept from the upstream serve recipe).
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_V2_MODEL_RUNNER=1

VLLM_PID=""
cleanup() {
  echo "[\$(date -Is)] cleanup: stopping vllm (pid \${VLLM_PID:-none}) ..."
  if [[ -n "\${VLLM_PID:-}" ]] && kill -0 "\$VLLM_PID" 2>/dev/null; then
    kill -TERM "\$VLLM_PID" 2>/dev/null || true
    for _ in \$(seq 1 30); do kill -0 "\$VLLM_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "\$VLLM_PID" 2>/dev/null || true
  fi
  rm -f "$ENDPOINT_FILE"
  echo "[\$(date -Is)] cleanup done"
}
# scancel sends SIGTERM (then SIGKILL after KillWait); also fire on normal EXIT / Ctrl-C.
trap cleanup TERM INT EXIT

echo "[\$(date -Is)] node=\$NODE  serving '$MODEL_PATH' as 'hosted_vllm/$SERVED_NAME' on port $PORT"
echo "[\$(date -Is)] vllm stdout/stderr -> $VLLM_LOG"

# Note on --attention-config: fp8 KV cache (--kv-cache-dtype fp8) requires an fp8 prefill query,
# which only the FlashInfer / TRT-LLM-Ragged / TokenSpeed MLA prefill backends support. On
# Blackwell the auto-selected prefill backend is FlashAttention (unsupported), so we pin
# mla_prefill_backend=FLASHINFER; otherwise vllm aborts warm-up with
# "Kimi-K3 fp8 KV cache requires an fp8 prefill query".
"$VLLM_BIN" serve "$MODEL_PATH" \\
  --served-model-name "$SERVED_NAME" \\
  --host 0.0.0.0 \\
  --port $PORT \\
  --trust-remote-code \\
  --moe-backend auto \\
  --gpu-memory-utilization 0.95 \\
  --tensor-parallel-size 8 \\
  --load-format fastsafetensors \\
  --no-enable-flashinfer-autotune \\
  --max-model-len 1048576 \\
  --kv-cache-dtype fp8 \\
  --attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"FLASHINFER"}' \\
  --enable-prefix-caching \\
  --language-model-only \\
  --reasoning-parser kimi_k3 \\
  --enable-auto-tool-choice \\
  --tool-call-parser kimi_k3 \\
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=\$!

echo "[\$(date -Is)] waiting for health at \$HEALTH_URL (timeout ${STARTUP_TIMEOUT}s) ..."
UP=0
for i in \$(seq 1 $STARTUP_TIMEOUT); do
  if ! kill -0 "\$VLLM_PID" 2>/dev/null; then
    echo "[\$(date -Is)] !!! vllm exited during startup. Last 200 lines of $VLLM_LOG:"
    tail -n 200 "$VLLM_LOG" || true
    exit 1
  fi
  if curl -sf "\$HEALTH_URL" >/dev/null 2>&1; then
    UP=1
    echo "[\$(date -Is)] vllm healthy after \${i}s"
    break
  fi
  sleep 1
done
if [[ "\$UP" -ne 1 ]]; then
  echo "[\$(date -Is)] !!! vllm not healthy within ${STARTUP_TIMEOUT}s. Last 200 lines of $VLLM_LOG:"
  tail -n 200 "$VLLM_LOG" || true
  exit 1
fi

# Publish the resolved endpoint for the helper scripts.
mkdir -p "$RUN_DIR"
{
  echo "BASE_URL=\$BASE_URL"
  echo "NODE=\$NODE"
  echo "PORT=$PORT"
  echo "JOBID=\${SLURM_JOB_ID:-unknown}"
} >"$ENDPOINT_FILE"
echo "[\$(date -Is)] endpoint ready: \$BASE_URL  (model: hosted_vllm/$SERVED_NAME)"
echo "[\$(date -Is)] wrote $ENDPOINT_FILE"

# Stay alive alongside vllm; the trap handles teardown on scancel/exit.
wait "\$VLLM_PID"
EOF

if [[ "$PRINT_ONLY" -eq 1 ]]; then
  cat "$SBATCH_FILE"
  exit 0
fi

JOBID="$(sbatch --parsable "$SBATCH_FILE")"
echo "Submitted Slurm job $JOBID (name: $SERVED_NAME)"
echo "  logs:     tail -f $RUN_DIR/slurm-$JOBID.log"
echo "  health:   $KIMI_DIR/health.sh   (once the job is running)"
echo "  stop:     scancel $JOBID"
