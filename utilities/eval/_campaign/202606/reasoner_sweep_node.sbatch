#!/bin/bash
# Self-contained: serve ONE thinkON sft2812 reasoner across all 8 GPUs (TP8) on a node, wait for
# it warm, then run ONE reasoner sweep over the given obs list against it. Survives disconnect
# (sbatch). Mirrors the discipline of eval_all.sbatch (set -uo pipefail, all-guarded vars, dated
# slurm logs, config echo + nvidia-smi). 2026-06-23.
#
# Submit (pick a free 8-GPU node; the ONLY list is a SINGLE quoted arg):
#   sbatch --job-name=eval-reasoner-sweep-tp8 --nodelist=worker-4 \
#     /home/sgsilva/utilities/eval/_campaign/202606/reasoner_sweep_node.sbatch \
#     "Qwen3.5-4B_thinkon grpo_step492 oracle-obs-cat-union5 ..."
# Arg $1 = the ONLY obs list (space-separated substrings). Omit to sweep ALL discoverable obs.
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --mem=2000G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/sgsilva/logs/eval/slurm/reasoner_sweep_tp8_slurm-%j.out
#SBATCH --error=/mnt/data/sgsilva/logs/eval/slurm/reasoner_sweep_tp8_slurm-%j.err
set -uo pipefail

REASONER=/mnt/data/pmartins/vlm_ckpts/sft_qwen_35_27b_1805_not_reviewed_diverse_reasoning_merged_rep_fe_comparison_3_mix_reas_12k_full_2005/step_2812/hf
SERVE=/home/sgsilva/vlm-evaluation/start_vllm_server.sh
SWEEP=/home/sgsilva/utilities/eval/reasoner_sweep.sh
VENV=/home/sgsilva/vlm-post-training-home-venv
HOST="$(hostname)"
# Clear any leaked env (PORT/TAG/etc from --export=ALL) BEFORE setting our own, so a stale exported
# value can't override us and set -u doesn't trip later (eval_all.sbatch's leak-guard discipline).
unset PORT TAG SERVE_VENV BENCH_EXTRA MAX_SAMPLES TESTSET 2>/dev/null || true
PORT="${PORT_PIN:-8013}"
MAX_LEN="${MAX_LEN:-65536}"
MAX_WORKERS="${MAX_WORKERS:-8}"
RESUME_ATTEMPTS="${RESUME_ATTEMPTS:-12}"
SERVE_WAIT_TICKS="${SERVE_WAIT_TICKS:-150}"   # ×10s = ~25 min max for TP8 load
ONLY_LIST="${1:-}"

mkdir -p /mnt/data/sgsilva/logs/eval/slurm
echo "===== reasoner_sweep_node.sbatch (TP8) ====="
echo "  node      : $HOST"
echo "  job id    : ${SLURM_JOB_ID:-<interactive>}"
echo "  reasoner  : $REASONER"
echo "  serve     : port $PORT  TP=8  max_len=$MAX_LEN  (thinkON)"
echo "  sweep     : MAX_WORKERS=$MAX_WORKERS  RESUME_ATTEMPTS=$RESUME_ATTEMPTS"
echo "  obs ONLY  : ${ONLY_LIST:-<ALL discoverable>}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    gpu /'
echo "============================================"

# ---- serve ONE thinkON reasoner across all 8 GPUs (TP8), backgrounded ----
LOGDIR=/mnt/data/sgsilva/logs/serve/$(date -u +%Y-%m-%d); mkdir -p "$LOGDIR"
SERVE_LOG="$LOGDIR/serve_sft2812_tp8_${SLURM_JOB_ID:-x}_${PORT}.log"
ENABLE_THINKING=1 QWEN35_VENV="$VENV" \
  "$SERVE" "$REASONER" 8 "$MAX_LEN" "$PORT" > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
echo "  serving (TP8) on :$PORT  pid=$SERVE_PID  log=$SERVE_LOG"

# ---- wait for WARM (cold-start guard — never start the sweep against a loading server) ----
ok=0
for i in $(seq 1 "$SERVE_WAIT_TICKS"); do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "[FAIL] serve process (pid $SERVE_PID) died during startup — see $SERVE_LOG"; exit 1
  fi
  if curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    echo "  :$PORT WARM after ~$((i*10))s"; ok=1; break
  fi
  sleep 10
done
[[ "$ok" == 1 ]] || { echo "[FAIL] :$PORT never came up after $((SERVE_WAIT_TICKS*10))s — see $SERVE_LOG"; exit 1; }

# ---- run the sweep against the warm TP8 server (env-driven, like reasoner_sweep.sh expects) ----
# ONE big server feeds MAX_WORKERS=8; RESUME_ATTEMPTS high so the ~20% thinkON runaway rate still
# converges to 1181 (avoids the overnight [INCOMPLETE] tails). ONLY exported on its own line so a
# multi-word list reaches the sweep intact (inline VAR=val prefix splits on spaces — the 127 bug).
echo "==== launching sweep ($(date)) ===="
export REASONER
export REASONER_URL="http://$HOST:$PORT/v1"
export MAX_WORKERS RESUME_ATTEMPTS
[[ -n "$ONLY_LIST" ]] && export ONLY="$ONLY_LIST"
"$SWEEP"
SWEEP_RC=$?
echo "  sweep exited (rc=$SWEEP_RC)"

# ---- recompile board so the new vo_s2 cells land ----
echo "==== recompiling board ($(date)) ===="
"$VENV/bin/python" /home/sgsilva/utilities/eval/compile_eval_results.py 2>&1 | grep -E 'wrote' | head -1

# ---- tear down our server (kill ONLY the PID we started — never pkill-by-pattern) ----
echo "==== tearing down serve pid $SERVE_PID ===="
kill "$SERVE_PID" 2>/dev/null || true
echo "==== reasoner_sweep_node COMPLETE ($(date), sweep rc=$SWEEP_RC) ===="
exit "$SWEEP_RC"
