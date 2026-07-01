#!/usr/bin/env bash
# vault_freshness_check.sh — weekly NUDGE (not a doer) for ~/.claude vault hygiene.
#
# WHY a nudge and not a runner: /digest, /plan-status, /report-status are Claude skills that
# spawn agents/workflows and make judgment calls — they need a real Claude session, and an
# unattended headless run can hang on permission prompts. So this script only DETECTS staleness
# and writes a reminder; you run the skills in a session. Mirrors the hook-nudge philosophy.
#
# DRIVER: the login node has no cron, so this is invoked by the Stop hook
# (session-consolidate.sh → check_vault_freshness), which rate-limits it to once every ~7 days.
# It can also be run by hand anytime. Reminder lands in $NUDGE_FILE (surfaced by the hook at
# session end) + the weekly log. Exit 0 = fresh (nudge cleared); prints STALE + writes nudge otherwise.
set -uo pipefail

CLAUDE_DIR="/home/sgsilva/.claude"
REPORTS="$CLAUDE_DIR/reports"
DOC_INDEX="$CLAUDE_DIR/DOC_INDEX.md"
PLANS="$CLAUDE_DIR/plans"
ACTIVE_PIPELINES="/home/sgsilva/utilities/ACTIVE_PIPELINES.md"
NUDGE_FILE="$CLAUDE_DIR/.vault-freshness-nudge"        # next session reads this; deleted once acted on
STALE_DAYS="${STALE_DAYS:-8}"                          # ACTIVE_PIPELINES older than this = stale

now=$(date +%s)
today=$(date +%F)
issues=""

# 1) DOC_INDEX coverage — reports on disk not in the index (excl living *_LIVE twins)
if [[ -f "$DOC_INDEX" ]]; then
  unindexed=0
  while IFS= read -r f; do
    b="$(basename "$f")"
    case "$b" in *_LIVE.md) continue;; esac
    grep -q "$b" "$DOC_INDEX" 2>/dev/null || unindexed=$((unindexed+1))
  done < <(find "$REPORTS" -maxdepth 1 -name '*.md' 2>/dev/null)
  [[ "$unindexed" -gt 0 ]] && issues+="  • $unindexed report(s) not in DOC_INDEX → run /report-status"$'\n'
fi

# 2) plans without a STATUS banner in their first 3 lines (unstamped by /plan-status)
if [[ -d "$PLANS" ]]; then
  unstamped=0
  for p in "$PLANS"/*.md; do
    [[ -e "$p" ]] || continue
    head -3 "$p" | grep -q "STATUS" || unstamped=$((unstamped+1))
  done
  [[ "$unstamped" -gt 0 ]] && issues+="  • $unstamped plan(s) missing a STATUS banner → run /plan-status"$'\n'
fi

# 3) ACTIVE_PIPELINES dashboard age
if [[ -f "$ACTIVE_PIPELINES" ]]; then
  age_days=$(( (now - $(stat -c %Y "$ACTIVE_PIPELINES")) / 86400 ))
  [[ "$age_days" -ge "$STALE_DAYS" ]] && issues+="  • ACTIVE_PIPELINES is ${age_days}d old → run /digest"$'\n'
fi

if [[ -n "$issues" ]]; then
  {
    echo "=== Vault freshness nudge ($today) ==="
    echo "The weekly hygiene triad is overdue. In a Claude session, run:"
    echo "$issues"
    echo "Or just run /clean (Mode F covers reports; also sweeps memory/cruft)."
  } > "$NUDGE_FILE"
  # also log it (own line via the standard log dir; no log_run wrapper needed — this is a cheap check)
  logdir="/mnt/data/sgsilva/logs/misc/$today"; mkdir -p "$logdir" 2>/dev/null
  cat "$NUDGE_FILE" >> "$logdir/vault_freshness_check.log" 2>/dev/null
  echo "vault freshness: STALE — nudge written to $NUDGE_FILE"
else
  rm -f "$NUDGE_FILE" 2>/dev/null   # all fresh → clear any old nudge
  echo "vault freshness: all fresh ($today)"
fi
