#!/usr/bin/env bash
#
# prune_sweep_checkpoints.sh — safely delete SUPERSEDED megatron checkpoints
# for the reasoning-trace sweep (visual-obs-sft). Each step_N is ~357 GB.
#
# Safety rules (all enforced):
#   - Never delete a step dir unless a STRICTLY LATER step_M (M>N) exists in the
#     same variant dir (so we never remove the most recent checkpoint).
#   - Never delete a dir modified within --min-age-min minutes (default 15) — a
#     dir still being written is left alone.
#   - DRY-RUN by default; prints what it WOULD delete. Pass --run to delete.
#   - Optional --keep-ep1: keep the earliest step (ep1) even if superseded
#     (use until the ep1 sniff eval is done).
#
# Usage:
#   ./prune_sweep_checkpoints.sh                 # dry-run, all variants
#   ./prune_sweep_checkpoints.sh --keep-ep1      # dry-run, but never touch ep1
#   ./prune_sweep_checkpoints.sh --run --keep-ep1
#   ./prune_sweep_checkpoints.sh --run --variant A
#
set -euo pipefail
source /home/sgsilva/utilities/logs-utils/log_run.sh

CKROOT="/mnt/data/sgsilva/checkpoints"
PREFIX="sft_qwen35_27b_oracle_obs_cat_reasoning"
VARIANTS=(A B C D baseline_rerun)

DRY_RUN=1
KEEP_EP1=0
MIN_AGE_MIN=15
ONLY_VARIANT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) DRY_RUN=0 ;;
    --keep-ep1) KEEP_EP1=1 ;;
    --min-age-min) MIN_AGE_MIN="$2"; shift ;;
    --variant) ONLY_VARIANT="$2"; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
  shift
done

[[ -n "$ONLY_VARIANT" ]] && VARIANTS=("$ONLY_VARIANT")
if [[ $DRY_RUN -eq 0 ]]; then
    _PRUNE_LOG=$(log_start misc "prune_sweep_checkpoints")
    exec > >(tee -a "$_PRUNE_LOG") 2>&1
fi
echo "mode: $([[ $DRY_RUN -eq 1 ]] && echo DRY-RUN || echo RUN)  keep-ep1=$KEEP_EP1  min-age=${MIN_AGE_MIN}min"
echo

total_freed=0
for V in "${VARIANTS[@]}"; do
  d="${CKROOT}/${PREFIX}_${V}"
  [[ -d "$d" ]] || { echo "$V: no checkpoint dir"; continue; }
  # sorted step numbers present
  mapfile -t steps < <(ls "$d" 2>/dev/null | sed -n 's/^step_\([0-9]\+\)$/\1/p' | sort -n)
  [[ ${#steps[@]} -eq 0 ]] && { echo "$V: no step_* dirs"; continue; }
  latest="${steps[-1]}"
  echo "$V: steps present = ${steps[*]}  (latest = step_${latest}, KEPT)"
  for i in "${!steps[@]}"; do
    s="${steps[$i]}"
    [[ "$s" == "$latest" ]] && continue            # never delete the latest
    if [[ $KEEP_EP1 -eq 1 && $i -eq 0 ]]; then
      echo "    step_${s}: KEEP (ep1, --keep-ep1)"; continue
    fi
    sd="${d}/step_${s}"
    # age guard: skip if modified within MIN_AGE_MIN
    if [[ -n "$(find "$sd" -maxdepth 0 -mmin -"$MIN_AGE_MIN" 2>/dev/null)" ]]; then
      echo "    step_${s}: SKIP (modified < ${MIN_AGE_MIN}min ago — may be live)"; continue
    fi
    sz=$(du -sh "$sd" 2>/dev/null | cut -f1)
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "    step_${s}: WOULD DELETE ($sz)  superseded by step_${latest}"
    else
      echo "    step_${s}: DELETING ($sz)  superseded by step_${latest}"
      rm -rf "$sd"
    fi
  done
done
echo
[[ $DRY_RUN -eq 1 ]] && echo "DRY-RUN — nothing deleted. Re-run with --run to apply."