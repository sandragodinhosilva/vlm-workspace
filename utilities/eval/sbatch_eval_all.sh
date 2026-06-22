#!/usr/bin/env bash
# Submit eval_all.sbatch with SLURM .out/.err routed to a dated subdir.
# Usage: same as direct sbatch — export vars first, then run this wrapper.
#
#   export MODEL=... BASE_MODEL=... STAGES=... THINKING=...
#   [export TRAIN_GROUP_ID=... RUN_ID=... TAG=... TESTSET=... ...]
#   /home/sgsilva/utilities/eval/sbatch_eval_all.sh [--pack] [sbatch opts...]
#
# Why a wrapper: #SBATCH --output can't expand $(date) at submit time; this
# script pre-creates the dated dir and passes --output/--error on the CLI,
# which overrides the #SBATCH defaults.
#
# --pack (dynamic node packing): SLURM SPREADS by default — a small gpu:2 job lands on a
# fresh idle node instead of packing onto a node where you ALREADY have a gpu:4 eval with 4
# GPUs free, wasting a whole node on a shared cluster. With --pack, the wrapper finds one of
# YOUR currently-running nodes that has >= the requested GPUs free (tightest fit) and pins the
# job there via --nodelist. Falls back to normal scheduling (loud msg) if none fits. The job-id
# PORT scheme keeps co-packed jobs on distinct ports, so packing is collision-safe.

set -euo pipefail

SLURM_DIR="/mnt/data/sgsilva/logs/eval/slurm/$(date -u +%Y-%m-%d)"
mkdir -p "$SLURM_DIR"

# DEV nodes — always excluded (reserved for development, not eval/batch jobs).
EXCLUDE_NODES="worker-30,worker-31"

# ---- consume our own flags (NOT passed to sbatch); collect the rest as sbatch opts ----
PACK=0
SBATCH_ARGS=()
for a in "$@"; do
  case "$a" in
    --pack) PACK=1 ;;
    *) SBATCH_ARGS+=("$a") ;;
  esac
done

# ---- requested GPU count: from --gres=gpu:N in the sbatch args, else the #SBATCH default (4) ----
req_gpu=4
for a in "${SBATCH_ARGS[@]}"; do
  case "$a" in --gres=gpu:*) req_gpu="${a##*:}";; esac
done

# ---- --pack: pick one of MY running nodes with >= req_gpu free (tightest fit), pin via --nodelist.
# Skip if the caller already pinned a node. A dev node can't be picked (it's never one of mine, and
# --exclude guards anyway). Tightest fit (smallest sufficient free) preserves big gaps for big jobs.
PACK_NODE=""
if [[ "$PACK" == 1 ]]; then
  already_pinned=0
  for a in "${SBATCH_ARGS[@]}"; do case "$a" in --nodelist=*|-w) already_pinned=1;; esac; done
  if [[ "$already_pinned" == 1 ]]; then
    echo "[pack] caller already passed --nodelist; honoring it (no auto-pack)."
  else
    my_nodes="$(squeue -u "$USER" -h -t RUNNING -o '%N' 2>/dev/null | tr ',' '\n' | sort -u | grep -E '^worker-[0-9]+$' || true)"
    # SAME-BATCH RACE FIX (2026-06-21): scontrol AllocTRES LAGS — a job I packed onto a node seconds
    # ago isn't accounted yet, so a second --pack reads the node as still-free and double-pins it
    # (→ PENDING ReqNodeNotAvail forever). So ALSO subtract GPUs my OWN not-yet-RUNNING jobs have
    # already pinned via --nodelist (their ReqNodeList): build node -> pending-gpu-demand from squeue.
    declare -A pending_gpu=()
    while IFS='|' read -r pj pb prnl; do
      [[ -z "$pj" ]] && continue
      pn="$(printf '%s' "$prnl" | grep -oE 'worker-[0-9]+' | head -1)"   # a pinned pending job names ONE node
      [[ -z "$pn" ]] && continue
      pg="$(printf '%s' "$pb" | grep -oE 'gpu:[0-9]+' | grep -oE '[0-9]+$')"; pg="${pg:-0}"
      pending_gpu[$pn]=$(( ${pending_gpu[$pn]:-0} + pg ))
    done < <(squeue -u "$USER" -h -t PENDING,CONFIGURING -o '%i|%b|%R' 2>/dev/null)
    best_node=""; best_free=99
    while read -r n; do
      [[ -z "$n" ]] && continue
      case ",$EXCLUDE_NODES," in *",$n,"*) continue;; esac   # never a dev node
      info="$(scontrol show node "$n" 2>/dev/null)" || continue
      [[ "$info" == *State=*DOWN* || "$info" == *State=*DRAIN* ]] && continue
      cfg="$(printf '%s' "$info" | grep -oE 'CfgTRES=[^ ]*' | grep -oE 'gres/gpu=[0-9]+' | grep -oE '[0-9]+$')"
      alloc="$(printf '%s' "$info" | grep -oE 'AllocTRES=[^ ]*' | grep -oE 'gres/gpu=[0-9]+' | grep -oE '[0-9]+$')"
      free=$(( ${cfg:-8} - ${alloc:-0} - ${pending_gpu[$n]:-0} ))   # minus my own pinned-pending demand
      # tightest fit: smallest free that still satisfies the request
      if (( free >= req_gpu && free < best_free )); then best_free=$free; best_node="$n"; fi
    done <<< "$my_nodes"
    if [[ -n "$best_node" ]]; then
      PACK_NODE="$best_node"
      echo "[pack] packing onto $PACK_NODE ($best_free GPUs free, need $req_gpu) — a node you're already on."
    else
      echo "[pack] no running node of yours has >= $req_gpu GPUs free → normal scheduling (may land on a fresh node)."
    fi
  fi
fi
PACK_OPT=()
[[ -n "$PACK_NODE" ]] && PACK_OPT=( --nodelist="$PACK_NODE" )

# "$@"/SBATCH_ARGS (e.g. --gres=gpu:2 --job-name=...) and our PACK_OPT MUST come BEFORE the script
# path so SLURM treats them as sbatch OPTIONS, not positional args to eval_all.sbatch (which ignores
# them → a trailing --gres was silently dropped, mis-sizing jobs). --nodelist (PACK_OPT) is placed
# AFTER SBATCH_ARGS so an explicit caller --nodelist still wins.
sbatch \
  --output="${SLURM_DIR}/eval_all_slurm-%j.out" \
  --error="${SLURM_DIR}/eval_all_slurm-%j.err" \
  --export=ALL \
  --exclude="${EXCLUDE_NODES}" \
  "${SBATCH_ARGS[@]}" \
  "${PACK_OPT[@]}" \
  /home/sgsilva/utilities/eval/eval_all.sbatch
