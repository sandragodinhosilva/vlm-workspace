#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# chain_sft_export_1805_merged_reasoning.sh
#
# Wait for the 1805 merged-reasoning SFT job to finish, export all checkpoints
# to HF format, then delete raw Megatron step_N dirs only after verifying each
# HF export is complete (config.json + model.safetensors.index.json + all shards).
#
# Usage:  bash chain_sft_export_1805_merged_reasoning.sh <slurm_job_id>
#   e.g.  bash chain_sft_export_1805_merged_reasoning.sh 12345
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail

JOBID="${1:-}"
[[ -z "$JOBID" ]] && { echo "Usage: bash chain_sft_export_1805_merged_reasoning.sh <slurm_job_id>" >&2; exit 2; }

# ---- logging ----
source /home/sgsilva/utilities/logs-utils/log_run.sh
_CHAIN_LOGDIR=$(log_start --dir sft "chain_sft_export_1805_merged_reasoning_j${JOBID}")
exec > >(tee -a "$_CHAIN_LOGDIR/run.log") 2>&1
# ---- end logging ----

CKPT_DIR=/mnt/data/sgsilva/checkpoints/sft_qwen35_27b_oracle_obs_merged_reasoning_1805
MODELS=/mnt/data/sgsilva/models
PREFIX=qwen35-27b-oracle-obs-merged-reasoning-1805-sft
EXPORT_SCRIPT=/home/sgsilva/nemo-rl-vlm/scripts/export_all_checkpoints.sh
# export_all_checkpoints.sh auto-selects nemo-rl-vlm/.venv (has megatron.bridge).
# Do NOT set PYTHON_BIN here — vlm-post-training-home-venv lacks megatron.bridge.
LOGDIR=/mnt/data/sgsilva/logs/export_logs/chain_sft_1805_merged_reasoning
mkdir -p "$LOGDIR"
LOG="$LOGDIR/chain.log"

log() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }

log "=== chain_sft_export_1805_merged_reasoning: waiting for job $JOBID ==="

# 1. Poll until job leaves the queue (clean finish OR crash)
while squeue -j "$JOBID" -h -o "%T" 2>/dev/null | grep -q .; do
  NCKPT=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | wc -l)
  log "  job $JOBID still running ($NCKPT ckpts saved so far) ..."
  sleep 1800   # check every 30 min
done
log "=== job $JOBID left the queue ==="

# Give SLURM epilog + final tmp_step_N -> step_N rename time to settle
sleep 60
for _w in $(seq 1 20); do
  if ls -d "$CKPT_DIR"/tmp_step_* >/dev/null 2>&1; then
    log "  waiting for in-flight checkpoint to finalize: $(ls -d "$CKPT_DIR"/tmp_step_* 2>/dev/null | sed 's#.*/##' | tr '\n' ' ')"
    sleep 30
  else
    break
  fi
done
log "  final checkpoints: $(ls -d "$CKPT_DIR"/step_* 2>/dev/null | sed 's#.*/##' | sort -V | tr '\n' ' ')"

# 2. Export all checkpoints to HF
log "=== exporting all checkpoints to HF ==="
cd /home/sgsilva/nemo-rl-vlm
bash "$EXPORT_SCRIPT" \
  "$CKPT_DIR" "$MODELS" "$PREFIX" \
  > "$LOGDIR/export_all.log" 2>&1
EXPORT_EXIT=$?
log "  export exit=$EXPORT_EXIT | models: $(ls -d "$MODELS/${PREFIX}-step"* 2>/dev/null | sed 's#.*/##' | grep -oE 'step[0-9]+' | sort -V | tr '\n' ' ')"

# 3. Delete raw Megatron ckpts only after full HF verification
log "=== verifying and deleting raw Megatron checkpoints ==="

verify_hf() {
  local hf="$1"
  [[ -f "$hf/config.json" ]] || return 1
  [[ -f "$hf/model.safetensors.index.json" ]] || return 1
  /home/sgsilva/nemo-rl-vlm/.venv/bin/python - "$hf" <<'PYV' 2>/dev/null
import json, os, sys
hf = sys.argv[1]
idx = json.load(open(os.path.join(hf, "model.safetensors.index.json")))
shards = set(idx["weight_map"].values())
missing = [s for s in shards if not os.path.exists(os.path.join(hf, s))]
sys.exit(1 if missing else 0)
PYV
}

for SD in "$CKPT_DIR"/step_*; do
  [[ -d "$SD" ]] || continue
  STEP=$(basename "$SD" | grep -oE '[0-9]+')
  HF="$MODELS/${PREFIX}-step${STEP}"
  if verify_hf "$HF"; then
    SIZE=$(du -sh "$SD" 2>/dev/null | cut -f1)
    log "  step${STEP}: VERIFIED -> rm -rf $SD (~${SIZE})"
    rm -rf "$SD"
  else
    log "  step${STEP}: NOT fully verified at $HF -> KEEPING raw ckpt (no delete)"
  fi
done

REMAINING=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | wc -l)
log "  remaining raw ckpts: $REMAINING"
log "=== chain_sft_export_1805_merged_reasoning DONE ==="
log "  exported models: $(ls -d "$MODELS/${PREFIX}-step"* 2>/dev/null | sed 's#.*/##' | sort -V | tr '\n' ' ')"
