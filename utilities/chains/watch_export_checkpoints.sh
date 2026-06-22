#!/usr/bin/env bash
# watch_export_checkpoints.sh — export-BEFORE-evict for an in-flight SFT/GRPO run.
#
# THE PROBLEM: nemo-rl keeps only `keep_top_k` raw megatron checkpoints on disk — when a new
# step commits, the oldest-beyond-top-k is shutil.rmtree'd MID-TRAINING
# (nemo_rl/utils/checkpoint.py:267). So intermediate steps are lost before you can export them.
#
# THIS WATCHER: polls the checkpoint dir during training and, the moment a COMMITTED step_N
# appears that isn't yet exported to HF, exports it — so every saved step gets a durable HF copy
# even though only `keep_top_k` raw DCPs ever survive on disk. No training-code change (works on
# the SWORD-origin nemo-rl repo). Reuses scripts/export_all_checkpoints.sh as the export engine
# (it already skips already-exported steps + verifies config.json) per [[feedback_script_discipline]].
#
# ⚠ RUN THIS ON A DEV NODE — worker-30 or worker-31 (both IDLE+CLOUD+DRAIN, i.e. excluded from the
#   scheduler's compute pool). Don't waste a schedulable B300 node on export. The megatron→HF
#   conversion needs 1 GPU; before each export the watcher AUTO-PICKS a FREE LOCAL GPU and pins
#   CUDA_VISIBLE_DEVICES to it, WAITING (polling) until one frees — never steals a busy GPU (other
#   users share 30/31). "Free" = <FREE_GPU_MEM_MIB MiB used (default 1024).
# ⚠ SAFE TO EXPORT only a COMMITTED checkpoint: we skip tmp_step_* (still being written).
# ⚠ This NEVER deletes a raw checkpoint — eviction stays nemo-rl's job. We only ADD HF copies.
#   (The reverse — delete raw after verified export — is [[feedback_verified_delete_after_export]].)
#
# Usage (run ON worker-30 or 31, e.g. inside tmux):
#   watch_export_checkpoints.sh <checkpoint_dir> <output_base> <prefix> [poll_s]
# Stops automatically when the training job's checkpoint dir stops growing AND no squeue job owns
# it — or kill it (Ctrl-C). Re-runnable: skips already-exported steps. Override the free-GPU
# threshold with FREE_GPU_MEM_MIB; pin a fixed GPU with CUDA_VISIBLE_DEVICES (skips auto-pick).
#
# Example (ssh worker-31; tmux; then):
#   bash /home/sgsilva/utilities/chains/watch_export_checkpoints.sh \
#     /mnt/data/sgsilva/checkpoints/sft_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union \
#     /mnt/data/sgsilva/models \
#     qwen35-27b-oracle-obs-merged-1805-binary-aux12k-union-sft  300

set -uo pipefail

CKPT_DIR="${1:?usage: watch_export_checkpoints.sh <checkpoint_dir> <output_base> <prefix> [poll_s]}"
OUT_BASE="${2:?need output_base}"
PREFIX="${3:?need model_name_prefix}"
POLL="${4:-300}"   # seconds between scans (default 5 min; the evict window is save_period*step_time, usually hours)

EXPORTER="/home/sgsilva/nemo-rl-vlm/scripts/export_all_checkpoints.sh"
[ -f "$EXPORTER" ] || { echo "FATAL: exporter not found: $EXPORTER" >&2; exit 1; }
[ -d "$CKPT_DIR" ] || { echo "FATAL: checkpoint dir not found: $CKPT_DIR" >&2; exit 1; }

FREE_GPU_MEM_MIB="${FREE_GPU_MEM_MIB:-1024}"   # a GPU with < this many MiB used counts as free

# Guard: this is meant to run on a DEV node (worker-30/31). Warn if elsewhere — don't hard-block
# (user may have a free GPU somewhere), but make the misuse visible.
HOST="$(hostname)"
case "$HOST" in
  worker-30|worker-31) : ;;
  *) echo "[watch-export] ⚠ running on $HOST, not worker-30/31 — intended to run on a DEV node so a compute node isn't wasted." >&2 ;;
esac

# wait_for_free_gpu: echo the index of a free LOCAL GPU (< FREE_GPU_MEM_MIB used), polling until one
# frees. If CUDA_VISIBLE_DEVICES is already pinned, respect it (echo it, no wait). Never picks a busy
# GPU (other users share 30/31).
wait_for_free_gpu() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "$CUDA_VISIBLE_DEVICES"; return 0; fi
  while true; do
    local g
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
          | awk -v t="$FREE_GPU_MEM_MIB" -F', *' '$2 < t {print $1; exit}')
    if [ -n "$g" ]; then echo "$g"; return 0; fi
    echo "[watch-export] $(date '+%H:%M:%S') no free GPU on $HOST (<${FREE_GPU_MEM_MIB}MiB) — waiting ${POLL}s..." >&2
    sleep "$POLL"
  done
}

echo "[watch-export] host=$HOST  ckpt_dir=$CKPT_DIR"
echo "[watch-export] out_base=$OUT_BASE  prefix=$PREFIX  poll=${POLL}s  free_gpu_thresh=${FREE_GPU_MEM_MIB}MiB"
echo "[watch-export] exporting EVERY committed step before keep_top_k can evict it. Never deletes raw ckpts."

# A step is EXPORTABLE iff committed (step_N, not tmp_step_N) and not yet FULLY exported.
# ⚠ config.json lands BEFORE the safetensors shards finish — so "config.json exists" does NOT mean
# the export is complete. A robust "done" check needs the shard INDEX (model.safetensors.index.json,
# written last and listing every shard) AND that all listed shards are present. This prevents the
# watcher from marking a half-written step "done" and never finishing it. (export_all_checkpoints.sh
# itself only checks config.json — so we pass EXPORT_CLEAN_PARTIAL=1 below to make it retry partials.)
hf_done() {  # $1 = step number ; return 0 only if the HF export is COMPLETE
  local d="$OUT_BASE/${PREFIX}-step$1"
  local idx="$d/model.safetensors.index.json"
  [ -f "$d/config.json" ] || return 1
  if [ -f "$idx" ]; then
    # every shard named in the index must exist on disk
    local missing
    missing=$(/home/sgsilva/vlm-post-training-home-venv/bin/python - "$d" "$idx" <<'PY' 2>/dev/null
import json,sys,os
d,idx=sys.argv[1],sys.argv[2]
shards=set(json.load(open(idx)).get("weight_map",{}).values())
print(sum(1 for s in shards if not os.path.exists(os.path.join(d,s))))
PY
)
    [ "${missing:-1}" = "0" ]
  else
    # no index yet -> shards still being written (or single-file model w/o index): treat as NOT done
    # unless there's exactly one safetensors and no index is expected. Be conservative: not done.
    return 1
  fi
}

idle_scans=0
prev_mtime="$(stat -c '%Y' "$CKPT_DIR" 2>/dev/null || echo 0)"
while true; do
  # committed steps currently on disk (skip tmp_step_*)
  mapfile -t steps < <(find "$CKPT_DIR" -maxdepth 1 -type d -name 'step_*' 2>/dev/null \
                         | sed -E 's#.*/step_##' | sort -n)
  pending=()
  for s in "${steps[@]}"; do
    hf_done "$s" || pending+=("$s")
  done

  if [ "${#pending[@]}" -gt 0 ]; then
    echo "[watch-export] $(date '+%H:%M:%S') committed=${steps[*]:-none}  to-export=${pending[*]}"
    GPU="$(wait_for_free_gpu)"   # blocks until a local GPU is free (or honors a pinned CUDA_VISIBLE_DEVICES)
    echo "[watch-export] $(date '+%H:%M:%S') using GPU $GPU on $HOST for export"
    # export_all_checkpoints.sh exports every committed step_* and SKIPS those whose HF config.json
    # exists. We pass EXPORT_CLEAN_PARTIAL=1 so a step whose export died mid-write (config.json but
    # missing shards) is REMOVED and retried rather than skipped — pairs with the strict hf_done().
    # clog-wrap for the artifact log; pin the chosen free GPU.
    # shellcheck disable=SC1090
    source /home/sgsilva/utilities/logs-utils/log_run.sh
    CUDA_VISIBLE_DEVICES="$GPU" EXPORT_CLEAN_PARTIAL=1 \
      clog export "watchexport_${PREFIX}" -- bash "$EXPORTER" "$CKPT_DIR" "$OUT_BASE" "$PREFIX"
    idle_scans=0
  else
    # Nothing new to export. Decide: training still alive (keep watching) or done (exit)?
    # Robust, self-contained signal (NO fragile job-name match — squeue %j uses hyphens, %Z is the
    # repo dir, neither equals the underscore ckpt basename): training is "alive" if a save is in
    # progress (a tmp_step_* exists) OR the ckpt dir's mtime advanced since the prior scan (new
    # step_*/tmp_step_* landed). Only when the dir is quiet for STALE_SCANS consecutive scans AND no
    # tmp_step_* exists do we conclude the run finished + everything is exported, and exit.
    STALE_SCANS=3
    tmp_present=$(find "$CKPT_DIR" -maxdepth 1 -type d -name 'tmp_step_*' 2>/dev/null | head -1)
    cur_mtime=$(stat -c '%Y' "$CKPT_DIR" 2>/dev/null || echo 0)
    if [ -n "$tmp_present" ] || [ "${cur_mtime:-0}" != "${prev_mtime:-x}" ]; then
      echo "[watch-export] $(date '+%H:%M:%S') all committed steps exported; run still active (tmp=${tmp_present:+yes} mtime-moved) — waiting."
      idle_scans=0
    else
      idle_scans=$((idle_scans + 1))
      echo "[watch-export] $(date '+%H:%M:%S') all committed steps exported; ckpt dir quiet (idle $idle_scans/$STALE_SCANS)"
      [ "$idle_scans" -ge "$STALE_SCANS" ] && { echo "[watch-export] run finished + all steps exported — exiting."; break; }
    fi
    prev_mtime="$cur_mtime"
  fi
  sleep "$POLL"
done

echo "[watch-export] DONE. HF exports under: $OUT_BASE/${PREFIX}-step*"
find "$OUT_BASE" -maxdepth 1 -type d -name "${PREFIX}-step*" 2>/dev/null | sort -V
