#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# eval_grpo_steps.sh — evaluate ALL exported GRPO checkpoint steps sequentially.
#
# For each model dir matching a prefix, this:
#   1. serves it on a vLLM server (thinkoff ENABLE_THINKING=0 / thinkon =1)
#   2. waits until /v1/models is ready
#   3. runs stage-1 visual-obs (categorical, v1 schema)
#   4. runs agreement-vs-human  -> reads error_relevant.vs_gt.a.overall.micro_f1
#   5. stops the server, frees the port, moves to the next step
#   6. prints a per-step trajectory table at the end
#
# Sequential ON PURPOSE: one model on the node at a time (no GPU contention).
#
# Usage (run ON the eval node — verify hostname first; serving binds to it):
#   bash eval_grpo_steps.sh thinkoff
#   bash eval_grpo_steps.sh thinkon
#
# Outputs -> /mnt/data/sgsilva/results/visual_obs_runs/   (per CLAUDE.md)
# Eval venv = vlm-post-training-home-venv. Serve venv handled by start_vllm_server.sh.
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail   # NOT -e: a single bad step must not abort the whole sweep

# ---- logging ----
source /home/sgsilva/utilities/logs-utils/log_run.sh
_GRPO_LOG=$(log_start eval "eval_grpo_steps_${1:-unknown}")
exec > >(tee -a "$_GRPO_LOG") 2>&1
# ---- end logging ----

MODE="${1:-}"
if [[ "$MODE" != "thinkoff" && "$MODE" != "thinkon" ]]; then
  echo "Usage: bash eval_grpo_steps.sh <thinkoff|thinkon>" >&2
  exit 2
fi

# ── literal paths (no $VAR-built training paths handed around) ────────────────
PY=/home/sgsilva/vlm-post-training-home-venv/bin/python
VLM=/home/sgsilva/vlm-post-training
SERVE=/home/sgsilva/vlm-evaluation/start_vllm_server.sh
MODELS=/mnt/data/sgsilva/models
RESULTS=/mnt/data/sgsilva/results/visual_obs_runs
SYMLINKS=/mnt/data/sgsilva/datasets/1105_test_processed_symlinks
OBSFILE=/home/sgsilva/vlm-post-training/visual_observations_categorical.json
GT=/mnt/data/sgsilva/results/visual_obs_runs/oracle_397b_1105_categorical.json
EXPECT_REPS=1181   # stage-1 record count MUST equal this (per CLAUDE.md verify rule)

HOST="$(hostname)"   # never hardcode the node
PORT=4108            # change if taken (showmodels / ss -ltn)

# ── mode-specific knobs ───────────────────────────────────────────────────────
if [[ "$MODE" == "thinkoff" ]]; then
  PREFIX="qwen35-4b-oracle-obs-cat-sft-grpo-1105"
  ENABLE_THINKING=0
  THINK_FLAG="--disable-thinking"
  MAXTOK=16384
  TAG="thinkoff"
else
  PREFIX="qwen35-4b-oracle-obs-cat-reasoning-sft-grpo-1105"
  ENABLE_THINKING=1
  THINK_FLAG=""                    # thinkon = NO flag (script has only --disable-thinking; thinking comes from server ENABLE_THINKING=1)
  MAXTOK=32768                     # thinkon needs the big budget (real <think> trace)
  TAG="thinkon"
fi

mkdir -p "$RESULTS"
SUMMARY="$RESULTS/grpo_sweep_${TAG}_summary.tsv"
echo -e "step\tmicro_f1\tn_records\tstatus" > "$SUMMARY"

# ── preflight ─────────────────────────────────────────────────────────────────
echo "=== eval_grpo_steps: MODE=$MODE  HOST=$HOST  PORT=$PORT ==="
for p in "$SERVE" "$SYMLINKS" "$OBSFILE" "$GT"; do
  [[ -e "$p" ]] || { echo "FATAL: missing required path: $p" >&2; exit 3; }
done

# discover steps, numerically sorted
mapfile -t MODEL_DIRS < <(ls -d "$MODELS/${PREFIX}-step"* 2>/dev/null | sort -V)
if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
  echo "FATAL: no exported models match $MODELS/${PREFIX}-step*  (export first)" >&2
  exit 3
fi
echo "Found ${#MODEL_DIRS[@]} step(s): $(for d in "${MODEL_DIRS[@]}"; do basename "$d" | grep -oE 'step[0-9]+'; done | tr '\n' ' ')"

# ── helper: stop whatever is serving on $PORT (only our own procs) ─────────────
stop_server() {
  local pids
  pids=$(lsof -ti:"$PORT" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    # verify ownership before kill (never touch another user's proc)
    for pid in $pids; do
      if [[ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" == "sgsilva" ]]; then
        kill "$pid" 2>/dev/null || true
      fi
    done
    sleep 8
    # force any stragglers we own
    for pid in $(lsof -ti:"$PORT" 2>/dev/null || true); do
      [[ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" == "sgsilva" ]] && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3
  fi
}

# ── helper: wait until the server answers /v1/models (or time out) ────────────
wait_ready() {
  local tries=0
  while (( tries < 120 )); do   # ~20 min cap
    if curl -s --max-time 5 "http://$HOST:$PORT/v1/models" 2>/dev/null | grep -q '"id"'; then
      return 0
    fi
    sleep 10; ((tries++))
  done
  return 1
}

# ── main loop ─────────────────────────────────────────────────────────────────
for MD in "${MODEL_DIRS[@]}"; do
  MODEL="$MD"
  NAME="$(basename "$MD")"
  STEP="$(echo "$NAME" | grep -oE 'step[0-9]+')"
  STAGE1="$RESULTS/${NAME}_stage1_${TAG}.json"
  AGREE="$RESULTS/agreement_${NAME}_testonly_${TAG}.json"
  echo ""
  echo "──────── $NAME ($STEP) ────────"

  stop_server   # ensure port is free from the previous step

  echo "[$STEP] serving (ENABLE_THINKING=$ENABLE_THINKING) on $HOST:$PORT ..."
  ENABLE_THINKING=$ENABLE_THINKING "$SERVE" "$MODEL" 2 262144 "$PORT" \
    > "$RESULTS/serve_${NAME}_${TAG}.log" 2>&1 &
  if ! wait_ready; then
    echo "[$STEP] ERROR: server never became ready -> skip"; stop_server
    echo -e "${STEP}\tNA\t0\tserve_failed" >> "$SUMMARY"; continue
  fi
  echo "[$STEP] server ready."

  echo "[$STEP] stage-1 ..."
  cd "$VLM"
  "$PY" data_preparation/generate_visual_observations_human.py \
    --processed-dir "$SYMLINKS" \
    --model "$MODEL" \
    --server-url "http://$HOST:$PORT/v1" \
    --visual-obs-variant categorical \
    --visual-obs-file "$OBSFILE" \
    $THINK_FLAG --max-tokens "$MAXTOK" --max-workers 16 \
    --output-file "$STAGE1" --resume \
    > "$RESULTS/stage1_${NAME}_${TAG}.log" 2>&1
  if [[ ! -f "$STAGE1" ]]; then
    echo "[$STEP] ERROR: stage-1 produced no output -> skip"; stop_server
    echo -e "${STEP}\tNA\t0\tstage1_failed" >> "$SUMMARY"; continue
  fi

  # verify record count == EXPECT_REPS (per CLAUDE.md: count must equal N)
  NREC=$("$PY" -c "import json,sys;d=json.load(open('$STAGE1'));print(sum(len(v) if isinstance(v,dict) else 1 for v in (d.values() if isinstance(d,dict) else d)))" 2>/dev/null || echo 0)

  echo "[$STEP] agreement ..."
  "$PY" data_preparation/analyze_observation_agreement.py \
    --output "$AGREE" \
    --gt-source "$GT" \
    --label-a model --label-b oracle \
    --a "$STAGE1" --b "$GT" \
    > "$RESULTS/agree_${NAME}_${TAG}.log" 2>&1
  F1=$("$PY" -c "import json;print('%.4f'%json.load(open('$AGREE'))['error_relevant']['vs_gt']['a']['overall']['micro_f1'])" 2>/dev/null || echo "NA")

  STATUS="ok"
  [[ "$F1" == "NA" ]] && STATUS="agree_failed"
  [[ "$NREC" != "$EXPECT_REPS" ]] && STATUS="count_mismatch_${NREC}"
  echo "[$STEP] micro_f1=$F1  n=$NREC  status=$STATUS"
  echo -e "${STEP}\t${F1}\t${NREC}\t${STATUS}" >> "$SUMMARY"

  stop_server
done

# ── trajectory table ──────────────────────────────────────────────────────────
echo ""
echo "════════ GRPO $TAG trajectory (vs-human micro_f1, v1 schema) ════════"
column -t -s $'\t' "$SUMMARY"
echo ""
echo "SFT baseline for reference: thinkoff step357 = 0.4753"
echo "Summary saved: $SUMMARY"
