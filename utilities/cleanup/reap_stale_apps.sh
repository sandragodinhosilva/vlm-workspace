#!/usr/bin/env bash
# reap_stale_apps.sh — find (and optionally kill) STALE Gradio/app processes owned by sgsilva.
#
# A "stale app" = a long-lived python app process that escaped the launch_app.sh lifecycle
# (bare `nohup python app.py` from the old manual workflow) and is now squatting a port / holding
# a venv after its repo was archived. The 2026-06-30 case: a 14-day-old video-sft app.py still
# running from /home/sgsilva/video-sft-vlm (deleted) on port 7863.
#
# Detection signals (any one flags a candidate):
#   1. cwd is "(deleted)"  — the launch dir no longer exists (archived/removed repo). STRONGEST.
#   2. ELAPSED > STALE_DAYS (default 7) — apps are meant to be relaunched, not run for weeks.
# Only sgsilva-owned processes are ever touched (verified per-pid). Dry-run by default.
#
# Usage:
#   reap_stale_apps.sh            # dry-run: list candidates, kill nothing
#   reap_stale_apps.sh --kill     # kill the flagged candidates (sgsilva-owned only)
#   STALE_DAYS=14 reap_stale_apps.sh   # change the age threshold
set -uo pipefail

STALE_DAYS="${STALE_DAYS:-7}"
KILL=0
[ "${1:-}" = "--kill" ] && KILL=1

ME=$(id -un)
# app process patterns (gradio apps + the named viewers/dashboards)
PAT='app\.py|_viewer\.py|vibe_test\.py|grpo_dashboard|sft_dashboard|claude-tracker|gradio'

# elapsed-seconds helper (ps etime → seconds)
etime_secs() {  # $1 = pid
  local e; e=$(ps -o etime= -p "$1" 2>/dev/null | tr -d ' ') || return 1
  [ -z "$e" ] && { echo 0; return; }
  local d=0 hms="$e"
  case "$e" in *-*) d=${e%%-*}; hms=${e#*-};; esac
  local s=0 IFS=:; set -- $hms
  if   [ $# -eq 3 ]; then s=$((10#$1*3600 + 10#$2*60 + 10#$3))
  elif [ $# -eq 2 ]; then s=$((10#$1*60 + 10#$2))
  else s=$((10#$1)); fi
  echo $((d*86400 + s))
}

threshold=$((STALE_DAYS*86400))
found=0
printf '%-8s %-12s %-7s %s\n' PID ELAPSED FLAG "CMD / CWD"
echo "--------------------------------------------------------------------------"
for pid in $(pgrep -u "$ME" -f "$PAT" 2>/dev/null); do
  # never touch a non-sgsilva proc (defensive — pgrep -u already scopes it)
  [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" = "$ME" ] || continue
  cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || echo "?")
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-80)
  el=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  secs=$(etime_secs "$pid")
  flag=""
  case "$cwd" in *'(deleted)'*) flag="DELETED-CWD";; esac
  [ -z "$flag" ] && [ "$secs" -gt "$threshold" ] && flag="OLD>${STALE_DAYS}d"
  [ -z "$flag" ] && continue   # not stale → skip
  found=$((found+1))
  printf '%-8s %-12s %-7s %s\n' "$pid" "$el" "$flag" "$cmd"
  printf '%-30s cwd=%s\n' "" "$cwd"
  if [ "$KILL" -eq 1 ]; then
    kill "$pid" 2>/dev/null && echo "   -> SIGTERM sent" || echo "   -> kill failed"
  fi
done

echo "--------------------------------------------------------------------------"
if [ "$found" -eq 0 ]; then
  echo "No stale app processes. ✓"
elif [ "$KILL" -eq 0 ]; then
  echo "$found stale candidate(s). Re-run with --kill to terminate (sgsilva-owned only)."
  echo "Then relaunch any you still want via: ~/utilities/apps/launch_app.sh <name>"
else
  echo "$found candidate(s) signalled. Verify ports are freed: ss -tlnp | grep python"
fi
