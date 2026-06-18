#!/bin/bash
#SBATCH --job-name=serve-only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --output=/mnt/data/sgsilva/logs/eval/slurm/serve_only-%j.out
#SBATCH --error=/mnt/data/sgsilva/logs/eval/slurm/serve_only-%j.err
#SBATCH --time=12:00:00
#
# Serve ONE vLLM model and keep it up for the job's walltime, so MANY eval drivers can attach via
# --base-url WITHOUT each re-reserving a node / re-paying the serve cost. The server outlives any
# single eval (unlike eval_all --serve, whose server dies when that job's eval finishes).
#
# Submit (export vars first; nothing hardcoded to a node):
#   export MODEL=/mnt/data/pmartins/.../step_492/hf BASE_MODEL=qwen3.5-27b THINKING=on
#   export SERVE_VENV=/home/sgsilva/vlm-post-training-home-venv   # pmartins tokenizer
#   sbatch --export=ALL /home/sgsilva/utilities/eval/serve_only.sbatch
#
# Then find the node (squeue -u sgsilva), and attach evals from the LOGIN node:
#   eval_all.sh --model $MODEL --base-model qwen3.5-27b --stages benchmarks \
#     --base-url http://<node>:8000/v1 --thinking on \
#     --bench-extra "--skip-vsibench --skip-mmmu --max-tokens 8192"   # e.g. capped Video-MME only
#   (NO --serve: it reuses THIS server. Repeat for as many evals as you want.)
#
# Kill when done:  scancel <this job id>   (frees the node)
set -uo pipefail
: "${MODEL:?set MODEL=/path/to/ckpt}"
: "${BASE_MODEL:?set BASE_MODEL=qwen3.5-27b|qwen3.5-4b|qwen3.5-9b}"
: "${THINKING:?set THINKING=on|off (must match the SFT target)}"
PORT="${PORT:-8000}"
SERVE_VENV="${SERVE_VENV:-/home/sgsilva/qwen3.5-serving-home-venv}"
case "$BASE_MODEL" in *27b*) TP="${TP:-8}"; MAX_LEN="${MAX_LEN:-65536}" ;; *) TP="${TP:-8}"; MAX_LEN="${MAX_LEN:-32768}" ;; esac

# long external path -> short symlink (same trick as eval_all, so served id stays short)
SERVED="$MODEL"
if [[ ${#MODEL} -gt 120 ]]; then
  sn="$(basename "$(dirname "$MODEL")")_$(basename "$MODEL")"; sn="$(printf '%s' "${RUN_ID:-$sn}" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-60)"
  SERVED="/mnt/data/sgsilva/models/_ext/${sn}"
  mkdir -p /mnt/data/sgsilva/models/_ext
  find /mnt/data/sgsilva/models/_ext -maxdepth 1 -xtype l -delete 2>/dev/null
  ln -sfn "$MODEL" "$SERVED"
fi
ENABLE_BIT="$([[ "$THINKING" == on ]] && echo 1 || echo 0)"

echo "===== serve_only ====="
echo "  node    : $(hostname)   job: ${SLURM_JOB_ID:-interactive}"
echo "  serving : $SERVED  (real: $MODEL)"
echo "  TP=$TP  max_len=$MAX_LEN  port=$PORT  thinking=$THINKING  venv=$SERVE_VENV"
echo "  ATTACH  : eval_all.sh --base-url http://$(hostname):${PORT}/v1   (no --serve)"
echo "  kill    : scancel ${SLURM_JOB_ID:-<jobid>}"
echo "======================"

# foreground: the server runs until the job walltime / scancel. No teardown — that's the point.
exec env ENABLE_THINKING="$ENABLE_BIT" QWEN35_VENV="$SERVE_VENV" \
  /home/sgsilva/vlm-evaluation/start_vllm_server.sh "$SERVED" "$TP" "$MAX_LEN" "$PORT"
