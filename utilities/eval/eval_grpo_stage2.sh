#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# eval_grpo_stage2.sh — STAGE-2 (two-stage) eval for selected GRPO checkpoints.
#
# Stage-1 (visual observations) is already done (eval_grpo_steps.sh). Stage-2 feeds
# each model's stage-1 observations to the 27B reasoner -> error-detection F1 + severity.
#
# DECIDED RECIPE (POST_GRPO_EVAL_PLAN.md 2026-06-04): serve ONE thinkOFF 27B reasoner
# (ENABLE_THINKING=0) and reuse it for ALL two-stage runs — reasoner-OFF beat reasoner-ON
# on every metric for both thinkon and thinkoff stage-1. "Match the branch" was REJECTED.
#
# Scope (per request): thinkoff step190 (final) + step180 (best stage-1);
#                      thinkon  step116 (final) + step90  (worst stage-1).
#
# Run ON a GPU node (serves the 27B reasoner). Verify hostname first.
# Usage:  bash eval_grpo_stage2.sh
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail
source /home/sgsilva/utilities/logs-utils/log_run.sh
_STAGE2_LOG=$(log_start eval "eval_grpo_stage2")
exec > >(tee -a "$_STAGE2_LOG") 2>&1

PY=/home/sgsilva/vlm-post-training-home-venv/bin/python
VLM=/home/sgsilva/vlm-post-training
SERVE=/home/sgsilva/utilities/serve/start_vllm_server.sh
R=/mnt/data/sgsilva/results/visual_obs/runs
REASONER=/mnt/data/shared/models/Qwen3.5-27B
TEST=/mnt/data/shared/vlm/data/human_annotation_datasets/1105_not_reviewed/repetitions_test
HOST="$(hostname)"
PORT=4099   # reasoner thinkOFF slot (per the eval plan)

# Each item: "stage1_basename  out_tag". stage1 JSONs already exist under $R.
RUNS=(
  "qwen35-4b-oracle-obs-cat-sft-grpo-1105-step190_stage1_thinkoff.json|grpo_thinkoff_step190"
  "qwen35-4b-oracle-obs-cat-sft-grpo-1105-step180_stage1_thinkoff.json|grpo_thinkoff_step180"
  "qwen35-4b-oracle-obs-cat-reasoning-sft-grpo-1105-step116_stage1_thinkon.json|grpo_thinkon_step116"
  "qwen35-4b-oracle-obs-cat-reasoning-sft-grpo-1105-step90_stage1_thinkon.json|grpo_thinkon_step90"
)

SUMMARY="$R/grpo_stage2_summary.tsv"
echo -e "model\terr_detection_f1\tn\tstatus" > "$SUMMARY"
echo "=== eval_grpo_stage2: HOST=$HOST PORT=$PORT (reasoner thinkOFF) ==="

# preflight: every stage-1 input must exist (no silent skip)
for item in "${RUNS[@]}"; do
  s1="${item%%|*}"
  [[ -f "$R/$s1" ]] || { echo "FATAL: missing stage-1 input: $R/$s1" >&2; exit 3; }
done
[[ -x "$SERVE" ]] || { echo "FATAL: serve script missing: $SERVE" >&2; exit 3; }

# serve the 27B reasoner ONCE, thinkOFF
echo "[serve] 27B reasoner (ENABLE_THINKING=0) on $HOST:$PORT ..."
ENABLE_THINKING=0 "$SERVE" "$REASONER" 2 262144 "$PORT" > "$R/serve_reasoner27b_thinkoff.log" 2>&1 &
ready=0
for i in $(seq 1 150); do   # ~25 min cap (27B is big)
  if curl -s --max-time 5 "http://$HOST:$PORT/v1/models" 2>/dev/null | grep -q '"id"'; then ready=1; break; fi
  sleep 10
done
[[ "$ready" == 1 ]] || { echo "FATAL: reasoner never became ready" >&2; exit 4; }
echo "[serve] reasoner ready."

# loop the GRPO stage-1 JSONs through evaluate.py --two-stage
for item in "${RUNS[@]}"; do
  s1="${item%%|*}"; tag="${item##*|}"
  OUT="$R/twostage_${tag}_thinkoff.json"
  echo ""
  echo "──── $tag ────"
  cd "$VLM"
  "$PY" data_preparation/evaluate.py \
    --test-dataset-dir "$TEST" \
    --two-stage \
    --precomputed-visual-obs "$R/$s1" \
    --model "$REASONER" \
    --server-url "http://$HOST:$PORT/v1" \
    --max-tokens 16384 --max-workers 16 \
    --output-file "$OUT" --resume \
    > "$R/stage2_${tag}.log" 2>&1
  if [[ ! -f "$OUT" ]]; then
    echo "  ERROR: no stage-2 output"; echo -e "${tag}\tNA\t0\tstage2_failed" >> "$SUMMARY"; continue
  fi
  F1=$("$PY" -c "import json;d=json.load(open('$OUT'));print('%.4f'%d['metrics']['error_detection_f1'])" 2>/dev/null || echo NA)
  N=$("$PY" -c "import json;print(json.load(open('$OUT'))['metrics'].get('total_samples',0))" 2>/dev/null || echo 0)
  st="ok"; [[ "$F1" == NA ]] && st="parse_failed"; [[ "$N" != 1181 ]] && st="count_${N}"
  echo "  err_detection_f1=$F1  n=$N  status=$st"
  echo -e "${tag}\t${F1}\t${N}\t${st}" >> "$SUMMARY"
done

# stop the reasoner (only our own proc)
for pid in $(lsof -ti:"$PORT" 2>/dev/null || true); do
  [[ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" == "sgsilva" ]] && kill "$pid" 2>/dev/null || true
done

echo ""
echo "════════ GRPO stage-2 (err-detection F1, reasoner thinkOFF) ════════"
column -t -s $'\t' "$SUMMARY"
echo ""
echo "SFT baselines (reasoner thinkOFF): thinkoff step357 err-F1 0.4733 | thinkon step336 0.4313"
echo "Summary: $SUMMARY"
