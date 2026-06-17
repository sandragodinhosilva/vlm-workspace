#!/usr/bin/env bash
# Sync cluster docs to a local Obsidian vault via rsync.
#
# USAGE (from your laptop):
#   bash sync_obsidian.sh [--dry-run]
#
# SETUP (once):
#   1. Edit CLUSTER and VAULT below, or export them before running.
#   2. chmod +x sync_obsidian.sh
#   3. Optional cron every 15 min:
#      */15 * * * * CLUSTER=sgsilva@<host> VAULT=~/obsidian/sword-vlm bash ~/utilities/sync_obsidian.sh >> ~/obsidian-sync.log 2>&1
#
# VAULT LAYOUT (mirrors cluster paths, all links resolve):
#   memory/                                    ← ~/.claude/projects/-home-sgsilva/memory/
#   reports/                                   ← ~/.claude/reports/
#   vlm-post-training/docs/                    ← ~/vlm-post-training/docs/
#   vlm-post-training/aux_tasks/docs/          ← includes q3d/mix/mcqa investigation docs
#   vlm-post-training/aux_tasks/docs/visual_obs/ ← visual-obs-sft + POST_GRPO docs
#   vlm-post-training/aux_tasks/transcripts/docs/
#   vlm-post-training/scripts_regen/docs/
#   monitoring-app/docs/
#   video-sft-vlm/docs/
#   vlm-evaluation/docs/
#   sft-data-vlm/docs/

set -euo pipefail

CLUSTER="${CLUSTER:-sgsilva@cluster}"
VAULT="${VAULT:-$HOME/obsidian/sword-vlm}"
DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

RSYNC="rsync -avz --delete --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' $DRY"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] syncing to $VAULT"

# --- create all dirs ---
mkdir -p \
  "$VAULT/memory" \
  "$VAULT/reports" \
  "$VAULT/vlm-post-training/docs" \
  "$VAULT/vlm-post-training/aux_tasks/docs" \
  "$VAULT/vlm-post-training/aux_tasks/transcripts/docs" \
  "$VAULT/vlm-post-training/scripts_regen/docs" \
  "$VAULT/monitoring-app/docs" \
  "$VAULT/video-sft-vlm/docs" \
  "$VAULT/vlm-evaluation/docs" \
  "$VAULT/sft-data-vlm/docs"

# --- sync all doc dirs ---
$RSYNC "$CLUSTER:/home/sgsilva/.claude/projects/-home-sgsilva/memory/"         "$VAULT/memory/"
$RSYNC "$CLUSTER:/home/sgsilva/.claude/reports/"                               "$VAULT/reports/"
$RSYNC "$CLUSTER:/home/sgsilva/vlm-post-training/docs/"                        "$VAULT/vlm-post-training/docs/"
$RSYNC "$CLUSTER:/home/sgsilva/vlm-post-training/aux_tasks/docs/"              "$VAULT/vlm-post-training/aux_tasks/docs/"
$RSYNC "$CLUSTER:/home/sgsilva/vlm-post-training/aux_tasks/transcripts/docs/"  "$VAULT/vlm-post-training/aux_tasks/transcripts/docs/"
$RSYNC "$CLUSTER:/home/sgsilva/vlm-post-training/scripts_regen/docs/"          "$VAULT/vlm-post-training/scripts_regen/docs/"
$RSYNC "$CLUSTER:/home/sgsilva/monitoring-app/docs/"                           "$VAULT/monitoring-app/docs/"
$RSYNC "$CLUSTER:/home/sgsilva/video-sft-vlm/docs/"                            "$VAULT/video-sft-vlm/docs/"
$RSYNC "$CLUSTER:/home/sgsilva/vlm-evaluation/docs/"                           "$VAULT/vlm-evaluation/docs/"
$RSYNC "$CLUSTER:/home/sgsilva/sft-data-vlm/docs/"                             "$VAULT/sft-data-vlm/docs/"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
