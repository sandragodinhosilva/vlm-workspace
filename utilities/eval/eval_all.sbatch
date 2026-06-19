#!/bin/bash
#SBATCH --job-name=eval-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=1200G
#SBATCH --output=/mnt/data/sgsilva/logs/eval/slurm/eval_all_slurm-%j.out
#SBATCH --error=/mnt/data/sgsilva/logs/eval/slurm/eval_all_slurm-%j.err
#
# DEFAULT = HALF NODE (gpu:4, 96 CPU, 1200G) so TWO gpu:4 evals PACK onto one 8-GPU/192-CPU/
# 2489G node (partition OverSubscribe=YES:4). The old full-node default left half the GPUs idle:
# --cpus-per-task=192 + no --mem grabbed ALL CPUs+RAM, so nothing else could land even with 4
# free GPUs.  PORT is auto-derived from the allocated GPUs below so two co-located jobs DON'T
# collide on 8000 (no manual PORT= needed when packing).
#   - gpu:4 job (default)          : just submit; two pack per node automatically.
#   - FULL-NODE job (TP=8 / 397B)  : sbatch --gres=gpu:8 --cpus-per-task=192 --mem=2400G ...
#     (the 27B/397B aux+serve need all 8 GPUs; ask for the whole node explicitly.)
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
# PORT: derived from SLURM_JOB_ID so two co-located packed jobs CANNOT collide. The old GPU-bank
# heuristic was BROKEN for packing — SLURM remaps CUDA_VISIBLE_DEVICES to 0-based PER JOB (both
# packed gpu:4 jobs see "0,1,2,3" → both picked 8000 → the 2nd eval queried the 1st's server →
# WRONG-MODEL garbage scores; war story: 94198/94199 on worker-4, 4B-on hit the 27B server → video
# 2%). job-id parity gives each job a distinct port with no manual bookkeeping. Explicit PORT= still
# overrides (e.g. to attach a known port). Interactive (no SLURM_JOB_ID) falls back to 8000.
# LEAK GUARD: under SLURM, an INHERITED PORT (almost always a stale export leaked via --export=ALL,
# NOT a deliberate pin) silently overrides the job-id scheme → two packed jobs both reuse the leaked
# 8000 → wrong-model collision (the F2.2 preflight catches it, but the run is wasted). So under SLURM
# we IGNORE an inherited PORT and always use the job-id port, warning loudly. To deliberately pin a
# port (rare — e.g. attach a known server), set PORT_PIN=<n>, which wins.
if [[ -n "${PORT_PIN:-}" ]]; then
  PORT="$PORT_PIN"
  echo "  PORT: pinned via PORT_PIN=$PORT"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _jobport=$(( 8000 + (SLURM_JOB_ID % 100) ))   # 8000-8099, distinct per job → collision-free when packed
  if [[ -n "${PORT:-}" && "${PORT}" != "${_jobport}" ]]; then
    echo "  ⚠ WARN: ignoring INHERITED PORT=${PORT} (stale --export=ALL leak?) — using job-id port ${_jobport}." >&2
    echo "          (a deliberate pin must use PORT_PIN=<n>; an inherited PORT under SLURM is treated as a leak.)" >&2
  fi
  PORT="$_jobport"
else
  PORT="${PORT:-8000}"   # interactive (no SLURM_JOB_ID): honor an explicit PORT, else 8000
fi
SERVE_WAIT="${SERVE_WAIT:-1800}"
BASE_URL="http://localhost:${PORT}/v1"

echo "===== eval_all.sbatch ====="
echo "  node      : $(hostname)"
echo "  job id    : ${SLURM_JOB_ID:-<interactive>}"
echo "  model     : $MODEL"
echo "  base      : $BASE_MODEL   stages: $STAGES   thinking: $THINKING"
echo "  serve     : port $PORT  TP=${TP:-default}  max_len=${MAX_LEN:-default}  wait=${SERVE_WAIT}s"
echo "  group/run : ${TRAIN_GROUP_ID:-<none>} / ${RUN_ID:-<none>}"
# --export=ALL forwards the WHOLE env, so a stale exported value silently re-targets this run.
# Echo the leak-prone vars so a leftover (esp. SERVE_VENV/TP/TESTSET/BENCH_MAX_TOKENS) is visible
# in the log BEFORE the run starts. SERVE_VENV must be UNSET for bare Qwen (pmartins-only).
echo "  env-leak watch: SERVE_VENV=${SERVE_VENV:-<unset>}  TESTSET=${TESTSET:-<default 1506>}  BENCH_MAX_TOKENS=${BENCH_MAX_TOKENS:-<default>}  TAG=${TAG:-<none>}  MAX_SAMPLES=${MAX_SAMPLES:-<none>}"
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
[[ -n "${BENCH_EXTRA:-}" ]]      && CMD+=( --bench-extra "$BENCH_EXTRA" )  # e.g. "--skip-videomme" (skip the slow runaway-tail benchmark)
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
