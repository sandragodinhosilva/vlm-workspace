#!/bin/bash
# Script to start vLLM server with proper configuration for different models
#
# Canonical serving envs now live under /home/sgsilva.
# Qwen 3.5 prefers /home/sgsilva/qwen3.5-serving-home-venv.
# Kimi prefers /home/sgsilva/kimi-serving-home-venv when present.
#
# Usage examples:
#   cd /path/to/vlm-evaluation
#
#   # Predefined models (looked up in MODELS dict):
#   ./start_vllm_server.sh qwen3.5-4b 8 262144 8000
#   ./start_vllm_server.sh qwen3.5-122b-a10b 8 262144 8000
#   ./start_vllm_server.sh glm-4.7 8 202752 8000        # capped at model max
#   ./start_vllm_server.sh qwen3-vl-4b-instruct 8 262144 8000
#
#   # Local finetuned checkpoint (direct path, contains "/"):
#   ./start_vllm_server.sh /mnt/data/sgsilva/models/qwen35-4b-3epochs-mcqa-video-0603-step50 8 262144 8000
#
# If the first argument contains "/" it's treated as a direct path to a model.
# Otherwise, it's looked up in the predefined MODELS dictionary.

# Model configurations
declare -A MODELS
MODELS["qwen3-vl-30b-a3b-instruct"]="Qwen/Qwen3-VL-30B-A3B-Instruct"
MODELS["qwen3-vl-30b-a3b-thinking"]="Qwen/Qwen3-VL-30B-A3B-Thinking"
MODELS["qwen3-vl-235b-a22b-thinking"]="Qwen/Qwen3-VL-235B-A22B-Thinking"
MODELS["qwen3-vl-235b-a22b-instruct"]="Qwen/Qwen3-VL-235B-A22B-Instruct"
MODELS["qwen3-vl-4b-instruct"]="Qwen/Qwen3-VL-4B-Instruct"
MODELS["qwen3-vl-4b-thinking"]="Qwen/Qwen3-VL-4B-Thinking"
MODELS["qwen3-vl-8b-instruct"]="Qwen/Qwen3-VL-8B-Instruct"
MODELS["qwen3-vl-8b-thinking"]="Qwen/Qwen3-VL-8B-Thinking"
MODELS["qwen3-vl-32b-instruct"]="Qwen/Qwen3-VL-32B-Instruct"
MODELS["qwen3-vl-32b-thinking"]="Qwen/Qwen3-VL-32B-Thinking"
MODELS["glm-4.5v"]="zai-org/GLM-4.5V"
MODELS["glm-4.6v"]="/mnt/data/shared/models/GLM-4.6V"
MODELS["glm-4.7"]="/mnt/data/shared/models/GLM-4.7"
MODELS["internvl3.5-241b"]="OpenGVLab/InternVL3_5-241B-A28B-HF"
MODELS["kimi-k2.5"]="/mnt/data/shared/models/Kimi-K2.5/"
MODELS["qwen3.5-397b-a17b"]="/mnt/data/shared/models/Qwen3.5-397B-A17B/"
MODELS["qwen3.5-122b-a10b"]="/mnt/data/shared/models/Qwen3.5-122B-A10B"
MODELS["qwen3.5-27b"]="/mnt/data/shared/models/Qwen3.5-27B"
MODELS["qwen3.5-35b-a3b"]="Qwen/Qwen3.5-35B-A3B"
MODELS["qwen3.5-4b"]="Qwen/Qwen3.5-4B"
MODELS["qwen3.5-9b"]="Qwen/Qwen3.5-9B"

MODEL_INPUT=${1:-"qwen3-vl-instruct"}
TENSOR_PARALLEL_SIZE=${2:-8}
MAX_MODEL_LEN=${3:-128192}
PORT=${4:-8000}
# Set ENABLE_THINKING=1 to enable Qwen3.5 thinking mode (for reasoning trace generation).
# Default is off - Qwen3.5 thinking is disabled by default per the model card.
ENABLE_THINKING="${ENABLE_THINKING:-0}"
GPU_BUSY_THRESHOLD_MIB="${GPU_BUSY_THRESHOLD_MIB:-4096}"
# Set STARTUP_HEARTBEAT_SECS to a positive integer to print periodic startup
# progress messages while vLLM is coming up. Default is disabled.
STARTUP_HEARTBEAT_SECS="${STARTUP_HEARTBEAT_SECS:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
QWEN35_VENV="${QWEN35_VENV:-}"
QWEN35_VENV_HOME_DEFAULT="${HOME:-/home/sgsilva}/qwen3.5-serving-home-venv"
KIMI_VENV="${KIMI_VENV:-}"
KIMI_VENV_HOME_DEFAULT="${HOME:-/home/sgsilva}/kimi-serving-home-venv"

port_in_use() {
    local port="$1"
    local ss_status=0

    if timeout 5s bash -lc "ss -ltnH '( sport = :$port )' 2>/dev/null | grep -q ." ; then
        return 0
    fi
    ss_status=$?

    if [ "$ss_status" -ne 1 ]; then
        echo "WARNING: 'ss' port probe timed out or failed for port $port; falling back to a Python bind probe." >&2
    fi

    python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("", port))
except OSError:
    raise SystemExit(0)
finally:
    sock.close()
raise SystemExit(1)
PY
}

check_local_model_path() {
    if [[ "$MODEL_PATH" == /* ]] && [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Local model not found at: $MODEL_PATH"
        exit 1
    fi
}

check_port_preflight() {
    if port_in_use "$PORT"; then
        echo "ERROR: Port $PORT is already in use."
        echo "Pick a different port, for example:"
        echo "  PORT=$((PORT + 1))"
        exit 1
    fi
}

check_gpu_preflight() {
    if [ "${SKIP_GPU_PREFLIGHT:-0}" = "1" ]; then
        echo "Skipping GPU preflight because SKIP_GPU_PREFLIGHT=1"
        return
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return
    fi

    local gpu_query_timeout_secs="${GPU_PREFLIGHT_TIMEOUT_SECS:-10}"
    local free_gpus=0
    local total_gpus=0
    local busy_report=""
    local idx used util
    local gpu_query_output

    echo "Checking GPU availability..."
    if ! gpu_query_output="$(timeout "${gpu_query_timeout_secs}s" nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
        local query_status=$?
        if [ "$query_status" -eq 124 ]; then
            echo "WARNING: GPU preflight timed out after ${gpu_query_timeout_secs}s; continuing without GPU availability checks."
        else
            echo "WARNING: GPU preflight failed; continuing without GPU availability checks."
        fi
        return
    fi

    while IFS=, read -r idx used util; do
        idx=$(echo "$idx" | xargs)
        used=$(echo "$used" | xargs)
        util=$(echo "$util" | xargs)
        total_gpus=$((total_gpus + 1))

        if [ "${used:-0}" -gt "$GPU_BUSY_THRESHOLD_MIB" ]; then
            busy_report+="  - GPU $idx: ${used} MiB used, ${util}% util"$'\n'
        else
            free_gpus=$((free_gpus + 1))
        fi
    done <<< "$gpu_query_output"

    if [ -n "$busy_report" ]; then
        echo "GPU preflight detected busy devices:"
        printf "%s" "$busy_report"
        echo ""
    fi

    if [ "$total_gpus" -gt 0 ] && [ "$free_gpus" -lt "$TENSOR_PARALLEL_SIZE" ] && [ "${ALLOW_BUSY_GPUS:-0}" != "1" ]; then
        echo "ERROR: Requested tensor parallel size $TENSOR_PARALLEL_SIZE, but only $free_gpus/$total_gpus GPUs look mostly free."
        echo "If you intentionally want to try anyway, rerun with ALLOW_BUSY_GPUS=1."
        exit 1
    fi
}

run_with_startup_heartbeat() {
    env PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" "$@" &
    local child_pid=$!
    local waited=0
    local heartbeat_secs="${STARTUP_HEARTBEAT_SECS:-0}"

    trap 'kill -INT "$child_pid" 2>/dev/null' INT TERM

    if [ "$heartbeat_secs" -gt 0 ]; then
        while kill -0 "$child_pid" 2>/dev/null; do
            sleep "$heartbeat_secs"
            waited=$((waited + heartbeat_secs))
            if kill -0 "$child_pid" 2>/dev/null; then
                echo "[startup] vLLM is still initializing... ${waited}s elapsed."
            fi
        done
    fi

    wait "$child_pid"
    local exit_code=$?
    trap - INT TERM
    return "$exit_code"
}

print_startup_heartbeat_status() {
    if [ "${STARTUP_HEARTBEAT_SECS:-0}" -gt 0 ]; then
        echo "Startup heartbeat: every ${STARTUP_HEARTBEAT_SECS}s"
    else
        echo "Startup heartbeat: disabled"
    fi
}

resolve_venv_path() {
    local label="$1"
    local requested="$2"
    shift 2
    local candidate

    if [ -n "$requested" ]; then
        if [ -x "$requested/bin/python" ]; then
            echo "$requested"
            return 0
        fi
        echo "ERROR: Requested $label env is missing a usable python binary: $requested" >&2
        return 1
    fi

    for candidate in "$@"; do
        if [ -x "$candidate/bin/python" ]; then
            echo "$candidate"
            return 0
        fi
    done

    echo "ERROR: Could not find a usable $label env." >&2
    for candidate in "$@"; do
        echo "  - checked: $candidate" >&2
    done
    return 1
}

run_qwen35_vllm() {
    local resolved_venv
    resolved_venv="$(resolve_venv_path "Qwen 3.5" "$QWEN35_VENV" "$QWEN35_VENV_HOME_DEFAULT")" || return 1
    echo "Serving env: $resolved_venv"
    run_with_startup_heartbeat "$resolved_venv/bin/python" -m vllm.entrypoints.cli.main "$@"
}

run_kimi_vllm() {
    local resolved_venv
    resolved_venv="$(resolve_venv_path "Kimi" "$KIMI_VENV" "$KIMI_VENV_HOME_DEFAULT")" || return 1
    echo "Serving env: $resolved_venv"
    run_with_startup_heartbeat "$resolved_venv/bin/python" -m vllm.entrypoints.cli.main "$@"
}

# Check if input is a path (contains /) or a model name
if [[ "$MODEL_INPUT" == *"/"* ]]; then
    # Direct path provided
    MODEL_PATH="$MODEL_INPUT"
    MODEL_NAME="$(basename "$MODEL_PATH")"
    echo "Using direct model path: $MODEL_PATH"
else
    # Look up in predefined models
    MODEL_NAME="$MODEL_INPUT"
    MODEL_PATH="${MODELS[$MODEL_NAME]}"
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: Unknown model name: $MODEL_NAME"
        echo "Available predefined models:"
        for key in "${!MODELS[@]}"; do
            echo "  - $key: ${MODELS[$key]}"
        done
        echo ""
        echo "Or provide a direct path to a model directory (must contain '/')"
        exit 1
    fi
fi

echo "Starting vLLM server with:"
echo "  Model Name: $MODEL_NAME"
echo "  Model Path: $MODEL_PATH"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  Port: $PORT"
echo ""

check_local_model_path
check_port_preflight
check_gpu_preflight

# Check model type and apply specific configuration
if [[ "$MODEL_PATH" == *"Qwen3.5"* ]] || [[ "$MODEL_PATH" == *"Qwen/Qwen3.5"* ]] || [[ "$MODEL_PATH" == *"qwen35"* ]]; then
    echo "Detected Qwen 3.5 model"
    echo "ENABLE_THINKING=$ENABLE_THINKING"
    echo ""
    print_startup_heartbeat_status
    THINKING_ARGS=()
    if [[ "$ENABLE_THINKING" == "1" ]]; then
        THINKING_ARGS+=(--reasoning-parser qwen3)
        echo "Thinking mode: ENABLED (--reasoning-parser qwen3)"
    else
        THINKING_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
        echo "Thinking mode: DISABLED (--default-chat-template-kwargs enable_thinking=false)"
    fi
    # Optional short alias: register the model under SERVED_MODEL_NAME instead of the (possibly
    # 225-char) path. Lets clients use a short id — avoids VLMEvalKit's >255-char filename slug
    # (Errno 36) for long external ckpt paths. Unset => identical to before (served id = path).
    SERVED_NAME_ARGS=()
    [[ -n "${SERVED_MODEL_NAME:-}" ]] && { SERVED_NAME_ARGS+=(--served-model-name "$SERVED_MODEL_NAME"); echo "Served model name: $SERVED_MODEL_NAME (alias for $MODEL_PATH)"; }
    run_qwen35_vllm serve "$MODEL_PATH" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
        --mm-encoder-tp-mode data \
        --mm-processor-cache-type shm \
        --media-io-kwargs '{"video": {"num_frames": 2048}}' \
        --port "$PORT" \
        --gpu-memory-utilization 0.85 \
        --enable-chunked-prefill \
        --no-enable-prefix-caching \
        "${SERVED_NAME_ARGS[@]}" \
        "${THINKING_ARGS[@]}"

elif [[ "$MODEL_PATH" == *"Kimi-K2.5"* ]]|| [[ "$MODEL_PATH" == *"kimi"* ]]; then
    echo "Detected Kimi K2.5 - using the preferred Kimi serving env"
    echo ""
    print_startup_heartbeat_status
    run_kimi_vllm serve "$MODEL_PATH" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
        --tool-call-parser kimi_k2 \
        --reasoning-parser kimi_k2 \
        --mm-encoder-tp-mode data \
        --gpu-memory-utilization 0.9 \
        --port "$PORT" \
        --no-enable-prefix-caching

elif [[ "$MODEL_PATH" == *"GLM-4.7"* ]]; then
    echo "Detected GLM-4.7 model - using GLM-4.7-specific configuration"
    echo ""
    # GLM-4.7 max_position_embeddings=202752; cap context to that value
    GLM47_MAX_LEN=$MAX_MODEL_LEN
    if [ "$GLM47_MAX_LEN" -gt 202752 ]; then GLM47_MAX_LEN=202752; fi
    print_startup_heartbeat_status
    run_qwen35_vllm serve "$MODEL_PATH" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$GLM47_MAX_LEN" \
        --tool-call-parser glm45 \
        --reasoning-parser glm45 \
        --enable-auto-tool-choice \
        --mm-encoder-tp-mode data \
        --mm_processor_cache_type shm \
        --media-io-kwargs '{"video": {"num_frames": 2048}}' \
        --allowed-local-media-path / \
        --port "$PORT"

else
    echo "Using standard vLLM configuration"
    echo "ENABLE_THINKING=$ENABLE_THINKING"
    echo ""
    print_startup_heartbeat_status
    THINKING_ARGS=()
    if [[ "$ENABLE_THINKING" == "1" ]]; then
        THINKING_ARGS+=(--reasoning-parser qwen3)
        echo "Thinking mode: ENABLED"
    else
        THINKING_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
        echo "Thinking mode: DISABLED (pass ENABLE_THINKING=1 to enable)"
    fi
    run_qwen35_vllm serve "$MODEL_PATH" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
        --mm-encoder-tp-mode data \
        --mm-processor-cache-type shm \
        --media-io-kwargs '{"video": {"num_frames": 2048}}' \
        --port "$PORT" \
        --gpu-memory-utilization 0.90 \
        --enable-chunked-prefill \
        --no-enable-prefix-caching \
        "${THINKING_ARGS[@]}"
fi
