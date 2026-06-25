#!/usr/bin/env bash
# launch_sft_with_export.sh — submit an SFT/GRPO training job AND start the
# export-before-evict watcher in ONE command, so the watcher can never be forgotten.
#
# WHY: nemo-rl keeps only keep_top_k raw checkpoints; intermediate step_N dirs are
# rmtree'd mid-training. The fix is watch_export_checkpoints.sh, but it must run on a
# DEV node (worker-30/31) — NOT inside the compute sbatch (which --excludes 30/31 and
# would waste a B300). So the watcher is a SEPARATE process; this wrapper launches both
# and derives all the watcher's args from the config, so there's nothing to remember.
#
# USAGE — run ON worker-30 or worker-31 (the watcher pins a free LOCAL GPU here):
#   launch_sft_with_export.sh <config.yaml> [sbatch_launcher.sh] [poll_s]
#
#   <config.yaml>         the training config (its checkpoint_dir + derived prefix drive the watcher)
#   [sbatch_launcher.sh]  optional; the slurm launcher to sbatch. If omitted, AUTO-DERIVED:
#                         examples/configs/recipes/vlm/slurm_multinode_worker_<ckptbasename-minus-sft_>.sh
#                         (only set this if your launcher name doesn't follow that convention)
#   [poll_s]              watcher poll interval (default 300)
#
# Example:
#   ssh worker-30 ; tmux ; then:
#   bash /home/sgsilva/utilities/chains/launch_sft_with_export.sh \
#     examples/configs/sft_vlm_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union_reasoning_megatron.yaml
#
# It: (1) lints the export name, (2) sbatch's the training job, (3) starts the watcher
# in the background on THIS dev node (logged via clog). The watcher self-stops when the
# training job's checkpoint dir stops growing and no squeue job owns it.

set -uo pipefail

NEMO_DIR="/home/sgsilva/nemo-rl-vlm"
LINT="/home/sgsilva/utilities/chains/lint_model_name.sh"
WATCHER="/home/sgsilva/utilities/chains/watch_export_checkpoints.sh"
OUT_BASE="/mnt/data/sgsilva/models"

CONFIG="${1:?usage: launch_sft_with_export.sh <config.yaml> [sbatch_launcher.sh] [poll_s]}"
LAUNCHER="${2:-}"
POLL="${3:-300}"

# Resolve config path (accept repo-relative or absolute).
case "$CONFIG" in
  /*) CFG_ABS="$CONFIG" ;;
  *)  CFG_ABS="$NEMO_DIR/$CONFIG" ;;
esac
[ -f "$CFG_ABS" ] || { echo "FATAL: config not found: $CFG_ABS" >&2; exit 1; }

# Dev-node guard (watcher needs a free LOCAL GPU; compute nodes are wrong).
HOST="$(hostname)"
case "$HOST" in
  worker-30|worker-31) : ;;
  *) echo "[launch+export] ⚠ on $HOST, not worker-30/31. The watcher pins a LOCAL GPU — run this on a dev node, or the export will fight for a compute GPU." >&2 ;;
esac

# 1) Lint + derive checkpoint_dir and prefix from the config (single source of truth).
echo "[launch+export] linting export name from $CFG_ABS ..."
LINT_OUT="$(bash "$LINT" --config "$CFG_ABS" 2>&1)" || { echo "$LINT_OUT" >&2; echo "FATAL: lint failed — fix checkpoint_dir basename before launching." >&2; exit 1; }
echo "$LINT_OUT"
# checkpoint_dir straight from the yaml; prefix from the lint output ("   prefix   : <X>" with -step<N> placeholder)
CKPT_DIR="$(grep -E '^\s*checkpoint_dir:' "$CFG_ABS" | head -1 | sed -E 's/.*checkpoint_dir:\s*//; s/\s*#.*//; s/^["'\'']//; s/["'\'']\s*$//')"
[ -n "$CKPT_DIR" ] || { echo "FATAL: no checkpoint_dir in $CFG_ABS" >&2; exit 1; }
# PREFIX = the exporter's derived prefix WITHOUT the -step<N>[_think] suffix (the watcher re-appends per step).
PREFIX="$(echo "$LINT_OUT" | grep -E '^\s*prefix\s*:' | head -1 | sed -E 's/.*prefix\s*:\s*//; s/-step<N>.*//')"
[ -n "$PREFIX" ] || { echo "FATAL: could not derive prefix from lint output" >&2; exit 1; }

# 2) Resolve the sbatch launcher (auto-derive from ckpt basename if not given).
if [ -z "$LAUNCHER" ]; then
  BASE="$(basename "$CKPT_DIR")"; BASE="${BASE#sft_}"   # ckpt dir = sft_<name> ; launcher = ..._worker_<name>.sh
  LAUNCHER="$NEMO_DIR/examples/configs/recipes/vlm/slurm_multinode_worker_${BASE}.sh"
fi
case "$LAUNCHER" in /*) : ;; *) LAUNCHER="$NEMO_DIR/$LAUNCHER" ;; esac
[ -f "$LAUNCHER" ] || { echo "FATAL: sbatch launcher not found: $LAUNCHER  (pass it as arg 2 if non-standard)" >&2; exit 1; }

echo "[launch+export] config     : $CFG_ABS"
echo "[launch+export] launcher   : $LAUNCHER"
echo "[launch+export] ckpt_dir   : $CKPT_DIR"
echo "[launch+export] prefix     : $PREFIX"
echo "[launch+export] out_base   : $OUT_BASE   poll=${POLL}s"

# 3) Submit training.
cd "$NEMO_DIR"
SB_OUT="$(sbatch "$LAUNCHER")" || { echo "FATAL: sbatch failed" >&2; exit 1; }
echo "[launch+export] $SB_OUT"
JOBID="$(echo "$SB_OUT" | grep -oE '[0-9]+' | head -1)"

# 4) Start the watcher on THIS dev node (logged via clog). It waits for step_N dirs +
#    a free local GPU; self-stops when the ckpt dir stops growing and no squeue job owns it.
source /home/sgsilva/utilities/logs-utils/log_run.sh
RUN_NAME="watch_export_$(basename "$CKPT_DIR" | sed 's/^sft_//')"
echo "[launch+export] starting watcher (background) → clog export $RUN_NAME"
nohup bash -c "source /home/sgsilva/utilities/logs-utils/log_run.sh && clog export '$RUN_NAME' -- bash '$WATCHER' '$CKPT_DIR' '$OUT_BASE' '$PREFIX' '$POLL'" >/dev/null 2>&1 &
WPID=$!
echo "[launch+export] watcher pid=$WPID  (exports each committed step_N to $OUT_BASE before keep_top_k prunes it)"
echo "[launch+export] DONE. training job=$JOBID ; watcher pid=$WPID on $HOST."
echo "  monitor training: squeue -j $JOBID ; tail -F $NEMO_DIR/slurm_logs/$(date +%Y%m%d)/training_*node_0_*.log"
echo "  monitor export  : tail -F \$(ls -t /mnt/data/sgsilva/logs/export/$(date +%Y-%m-%d)/${RUN_NAME}*.log | head -1)"
