#!/usr/bin/env bash
# Submit eval_all.sbatch with SLURM .out/.err routed to a dated subdir.
# Usage: same as direct sbatch — export vars first, then run this wrapper.
#
#   export MODEL=... BASE_MODEL=... STAGES=... THINKING=...
#   [export TRAIN_GROUP_ID=... RUN_ID=... TAG=... TESTSET=... ...]
#   /home/sgsilva/utilities/eval/sbatch_eval_all.sh
#
# Why a wrapper: #SBATCH --output can't expand $(date) at submit time; this
# script pre-creates the dated dir and passes --output/--error on the CLI,
# which overrides the #SBATCH defaults.

set -euo pipefail

SLURM_DIR="/mnt/data/sgsilva/logs/eval/slurm/$(date -u +%Y-%m-%d)"
mkdir -p "$SLURM_DIR"

# "$@" (e.g. --gres=gpu:2 --job-name=...) MUST come BEFORE the script path so SLURM treats them as
# sbatch OPTIONS that override the #SBATCH defaults. Anything AFTER the script path is passed as a
# positional ARG to eval_all.sbatch (which ignores it) — so a trailing --gres was silently dropped
# and the job fell back to the #SBATCH --gres=gpu:4 default (mis-sized 4B jobs). Keep "$@" here.
sbatch \
  --output="${SLURM_DIR}/eval_all_slurm-%j.out" \
  --error="${SLURM_DIR}/eval_all_slurm-%j.err" \
  --export=ALL \
  "$@" \
  /home/sgsilva/utilities/eval/eval_all.sbatch
