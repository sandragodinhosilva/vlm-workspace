#!/bin/bash
#SBATCH --job-name=eval-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --output=/mnt/data/sgsilva/logs/eval/slurm/eval_all_slurm-%j.out
#SBATCH --error=/mnt/data/sgsilva/logs/eval/slurm/eval_all_slurm-%j.err
#
# Serve a checkpoint, run all requested eval stages, compile the unified master CSV, then
# tear down the server — all in ONE sbatch job. The node frees when the job ends.
# eval_all.sh --serve owns the vLLM lifecycle (launch + health-wait + trap-kill on exit; it
# kills ONLY the PID it started, never pkill-by-pattern).
#
# SUBMIT VIA THE WRAPPER (preferred — puts .out/.err in a dated subdir):
#   /home/sgsilva/utilities/eval/sbatch_eval_all.sh
# Or manually: export vars first, then:
#   export MODEL=/mnt/data/sgsilva/models/qwen35-4b-mix-12k-1506-sft-step1299
#   export BASE_MODEL=qwen3.5-4b STAGES=aux,visualobs,benchmarks THINKING=off
#   export TRAIN_GROUP_ID=mix_12k_1506 RUN_ID=step1299_full_thinkoff
#   sbatch --export=ALL /home/sgsilva/utilities/eval/eval_all.sbatch
# (--export=ALL forwards your whole exported env, so STAGES' commas are safe.)
#
# Required env: MODEL, BASE_MODEL, STAGES, THINKING (on|off — must match the SFT target).
# aux stage also needs TRAIN_GROUP_ID + RUN_ID. Optional: TP, MAX_LEN, PORT, MAX_SAMPLES,
# JUDGE_BASE_URL, JUDGE_MODEL, SERVE_WAIT, TESTSET.
set -uo pipefail

# ---- required ----
: "${MODEL:?set MODEL=/mnt/data/sgsilva/models/<ckpt>}"
: "${BASE_MODEL:?set BASE_MODEL=qwen3.5-4b|qwen3.5-9b|qwen3.5-27b}"
: "${STAGES:?set STAGES=aux,visualobs,benchmarks (any subset)}"
: "${THINKING:?set THINKING=on|off (must match the SFT target)}"
# ---- optional (defaults) ----
PORT="${PORT:-8000}"
SERVE_WAIT="${SERVE_WAIT:-1800}"
BASE_URL="http://localhost:${PORT}/v1"

echo "===== eval_all.sbatch ====="
echo "  node      : $(hostname)"
echo "  job id    : ${SLURM_JOB_ID:-<interactive>}"
echo "  model     : $MODEL"
echo "  base      : $BASE_MODEL   stages: $STAGES   thinking: $THINKING"
echo "  serve     : port $PORT  TP=${TP:-default}  max_len=${MAX_LEN:-default}  wait=${SERVE_WAIT}s"
echo "  group/run : ${TRAIN_GROUP_ID:-<none>} / ${RUN_ID:-<none>}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/    gpu /'

# ---- build the eval_all.sh command (only pass flags that are set) ----
CMD=( /home/sgsilva/utilities/eval/eval_all.sh
  --model "$MODEL" --base-model "$BASE_MODEL" --stages "$STAGES"
  --serve --thinking "$THINKING" --base-url "$BASE_URL" --serve-wait "$SERVE_WAIT" )
[[ -n "${TP:-}" ]]              && CMD+=( --tp "$TP" )
[[ -n "${MAX_LEN:-}" ]]         && CMD+=( --max-len "$MAX_LEN" )
[[ -n "${SERVE_VENV:-}" ]]      && CMD+=( --serve-venv "$SERVE_VENV" )  # pmartins -> vlm-post-training-home-venv
[[ "${KEEP_SERVER:-0}" == 1 ]]  && CMD+=( --keep-server )  # leave vLLM up after eval (reuse; rarely useful in sbatch since node frees at job end)
[[ -n "${BENCH_MAX_TOKENS:-}" ]] && CMD+=( --bench-max-tokens "$BENCH_MAX_TOKENS" )  # cap thinkon runaways (8192) so Video-MME finishes in hours not days
[[ -n "${TRAIN_GROUP_ID:-}" ]]  && CMD+=( --train-group-id "$TRAIN_GROUP_ID" )
[[ -n "${RUN_ID:-}" ]]          && CMD+=( --run-id "$RUN_ID" )
[[ -n "${TAG:-}" ]]             && CMD+=( --tag "$TAG" )
[[ -n "${TESTSET:-}" ]]         && CMD+=( --testset "$TESTSET" )
[[ -n "${MAX_SAMPLES:-}" ]]     && CMD+=( --max-samples "$MAX_SAMPLES" )
[[ -n "${JUDGE_BASE_URL:-}" ]]  && CMD+=( --judge-base-url "$JUDGE_BASE_URL" )
[[ -n "${JUDGE_MODEL:-}" ]]     && CMD+=( --judge-model "$JUDGE_MODEL" )

echo "  exec: ${CMD[*]}"
echo "==========================="
"${CMD[@]}"
RC=$?
echo "===== eval_all.sbatch done (rc=$RC) — server torn down by eval_all --serve; node releasing ====="
exit $RC
