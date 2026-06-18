#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# chain_thinkoff_export_eval.sh — wait for the thinkoff GRPO run to finish, then
# export ALL its checkpoints and run the full sequential eval sweep. Unattended.
#
# Sequence:
#   1. poll until thinkoff job (JOBID) leaves the queue (done or crashed)
#   2. export every step_* checkpoint -> HF
#   2b. reclaim disk: rm -rf each raw megatron ckpt (~51G/step) ONLY after its HF
#       export is verified (config.json + >=1 safetensors). Partial/missing export
#       => keep the raw ckpt (never delete the only copy). Safe because the training
#       job already finished in step 1, so nothing needs the raw ckpt for --resume.
#   3. run eval_grpo_steps.sh thinkoff  (serves + stage-1 + agreement per step)
#
# Run this ON an eval node (it serves locally in step 3). Verify hostname first.
# Usage:  bash chain_thinkoff_export_eval.sh <thinkoff_jobid>
#   e.g.  bash chain_thinkoff_export_eval.sh 78085
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail

JOBID="${1:-}"
[[ -z "$JOBID" ]] && { echo "Usage: bash chain_thinkoff_export_eval.sh <thinkoff_jobid>" >&2; exit 2; }

# ---- logging ----
source /home/sgsilva/utilities/logs-utils/log_run.sh
LOGDIR=$(log_start --dir export "chain_thinkoff_export_eval_j${JOBID}")
exec > >(tee -a "$LOGDIR/run.log") 2>&1
# ---- end logging ----

CKPT_DIR=/mnt/data/sgsilva/checkpoints/grpo_visual_obs_cat_1105_4b
MODELS=/mnt/data/sgsilva/models
PREFIX=qwen35-4b-oracle-obs-cat-sft-grpo-1105
GRPO_VENV=/home/sgsilva/nemo-rl-vlm-grpo-home-venv/bin/python
EXPORT_SCRIPT=/home/sgsilva/nemo-rl-vlm/scripts/export_all_checkpoints.sh
EVAL_SCRIPT=/home/sgsilva/utilities/eval/eval_grpo_steps.sh

echo "=== chain_thinkoff: waiting for thinkoff job $JOBID to finish ==="
# 1. wait for the training job to leave the queue (covers clean finish AND crash)
while squeue -j "$JOBID" -h -o "%T" 2>/dev/null | grep -q .; do
  echo "  $(date '+%F %T')  job $JOBID still RUNNING ($(ls -d "$CKPT_DIR"/step_* 2>/dev/null | wc -l) ckpts saved) ..."
  sleep 300   # check every 5 min (training is slow, ~750s/step)
done
echo "=== job $JOBID left the queue at $(date '+%F %T') ==="
# The job leaving squeue can precede SLURM epilog + the final checkpoint's tmp_step_N -> step_N
# rename. Settle, then wait for any in-flight tmp_step_* to finalize so export reads only
# complete checkpoints (never a half-written final step).
sleep 30
for _w in $(seq 1 20); do   # up to ~10 min for a final save to finalize
  if ls -d "$CKPT_DIR"/tmp_step_* >/dev/null 2>&1; then
    echo "  waiting for in-flight checkpoint to finalize: $(ls -d "$CKPT_DIR"/tmp_step_* 2>/dev/null | sed 's#.*/##' | tr '\n' ' ')"
    sleep 30
  else
    break
  fi
done
echo "  final checkpoints: $(ls -d "$CKPT_DIR"/step_* 2>/dev/null | sed 's#.*/##' | sort -V | tr '\n' ' ')"

# 2. export every checkpoint (export_all_checkpoints does all step_* under the dir)
#    Safe to delete the raw megatron ckpts here: the training job has ALREADY finished
#    (we waited in step 1), so nothing needs them for resume anymore.
echo "=== exporting all thinkoff checkpoints -> HF ==="
cd /home/sgsilva/nemo-rl-vlm
PYTHON_BIN="$GRPO_VENV" bash "$EXPORT_SCRIPT" \
  "$CKPT_DIR" "$MODELS" "$PREFIX" \
  > "$LOGDIR/export_thinkoff_all.log" 2>&1
echo "  export exit=$?  | models: $(ls -d "$MODELS/${PREFIX}-step"* 2>/dev/null | sed 's#.*/##' | grep -oE 'step[0-9]+' | sort -V | tr '\n' ' ')"

# 2b. delete each raw megatron ckpt ONLY after its HF export is verified COMPLETE.
#     Strong gate (closes the partial-export data-loss path): config.json AND
#     model.safetensors.index.json AND every shard the index references must exist
#     on disk. A presence-only check (config + any .safetensors) would pass a
#     half-written export and then rm the only raw copy. Reclaims ~51G/step.
#     NEVER delete on a missing/partial/unverifiable export.
echo "=== deleting raw megatron checkpoints whose HF export is FULLY verified ==="
verify_hf() {  # returns 0 only if the HF dir is a complete, loadable export
  local hf="$1"
  [[ -f "$hf/config.json" ]] || return 1
  [[ -f "$hf/model.safetensors.index.json" ]] || return 1
  # every shard named in the index must exist (catches a truncated/aborted export)
  "$GRPO_VENV" - "$hf" <<'PYV' 2>/dev/null
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
    echo "  step${STEP}: HF export VERIFIED (config+index+all shards) -> rm -rf $SD (~$(du -sh "$SD" 2>/dev/null | cut -f1))"
    rm -rf "$SD"
  else
    echo "  step${STEP}: HF export NOT fully verified at $HF -> KEEPING raw ckpt $SD (no safe delete)"
  fi
done
echo "  remaining raw ckpts: $(ls -d "$CKPT_DIR"/step_* 2>/dev/null | wc -l)"

# 3. run the sequential eval sweep over all exported steps.
#    The sweep SERVES models, so it must run on a GPU node — grab a fresh idle node
#    NOW (not held during the long wait). Find one idle, then srun the sweep onto it.
echo "=== finding an idle node for the eval sweep ==="
SWEEP_NODE=""
for _try in $(seq 1 60); do   # retry up to ~30 min for a free node
  SWEEP_NODE=$(sinfo -h -N -o "%N %t" 2>/dev/null | awk '$2=="idle"{print $1; exit}')
  [[ -n "$SWEEP_NODE" ]] && break
  echo "  no idle node yet, retrying in 30s ..."; sleep 30
done
if [[ -z "$SWEEP_NODE" ]]; then
  echo "  no idle node found — run manually: bash $EVAL_SCRIPT thinkoff (on a free node)"; exit 4
fi
echo "=== running eval sweep (thinkoff) on $SWEEP_NODE ==="
srun --nodelist="$SWEEP_NODE" --gres=gpu:8 --time=03:00:00 --job-name=grpo-eval-thinkoff \
  bash "$EVAL_SCRIPT" thinkoff
echo "=== chain_thinkoff DONE at $(date '+%F %T') ==="
echo "  summary: /mnt/data/sgsilva/results/visual_obs_runs/grpo_sweep_thinkoff_summary.tsv"
