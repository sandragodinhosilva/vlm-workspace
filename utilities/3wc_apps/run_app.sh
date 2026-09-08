#!/usr/bin/env bash
# 3WC trace-viewer launcher.
#
# Self-contained on purpose: this repo is SWORDHealth's, and the VLM workstream's
# launch_app.sh / apps_registry.yaml assume VLM paths and venvs. Nothing here imports
# them, so the two stay independent (see CLAUDE.local.md "membrane").
#
#   /home/sgsilva/utilities/3wc_apps/run_app.sh                 # trace viewer, full merged corpus, port 7860
#   /home/sgsilva/utilities/3wc_apps/run_app.sh medconv         # medication-mention viewer, port 7883
#   /home/sgsilva/utilities/3wc_apps/run_app.sh elicit          # behaviour viewer (dormancy / PT handback), port 7884
#   /home/sgsilva/utilities/3wc_apps/run_app.sh -p 7865         # pick a port
#   /home/sgsilva/utilities/3wc_apps/run_app.sh --era 0907      # index ONLY the 0907 export (skip 2206/2606)
#   /home/sgsilva/utilities/3wc_apps/run_app.sh --fg            # run in foreground (Ctrl-C to stop)
#   /home/sgsilva/utilities/3wc_apps/run_app.sh --status        # is it up? where is it logging?
#   /home/sgsilva/utilities/3wc_apps/run_app.sh --stop          # stop the instance on this port
#
# First launch scans the corpus to build byte-offset indexes (~34 GB for the full
# merged set, a few minutes); every later launch loads them from .trace_index/ and
# starts in seconds. Cache is keyed on (size, mtime, INDEX_VERSION) per file, so
# only a changed/new file is ever re-scanned.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # this launcher (utilities/3wc_apps)
APP_DIR="${APP_DIR:-/home/sgsilva/dawn-research/3wc}"           # the app itself, in the SWORD repo
DATA_ROOT="${DATA_DIR:-/mnt/data/shared/3wc}"                   # READ-ONLY: never write here
                                                                # (moved from /mnt/data/pmartins/3-way-chat-thrive, 2026-07-28)
PYTHON="${PYTHON:-/mnt/data/sgsilva/.venvs/uv/bin/python3}"
MEDCONV_PYTHON="${MEDCONV_PYTHON:-/home/sgsilva/dawn-research-3wc-home-venv/bin/python}"
LOG_DIR="${LOG_DIR:-/mnt/data/sgsilva/logs/3wc}"

PORT="${PORT:-}"
HOST="${HOST:-0.0.0.0}"
ERA=""
MODE="bg"

# App registry: name -> "script relative to APP_DIR | default port".
# `trace` is the original viewer; add a row here rather than writing a second launcher,
# so start/stop/status/ownership checks stay in ONE place.
APP="trace"
declare -A APP_SCRIPT=( [trace]="app.py"        [medconv]="apps/med_conversations.py"  [elicit]="apps/elicit_behavior.py" )
declare -A APP_PORT=(   [trace]="7860"          [medconv]="7883"                       [elicit]="7884" )
declare -A APP_DESC=(
  [trace]="Langfuse trace viewer — Member / Phoenix / Specialist"
  [medconv]="medication-mention conversations (member + send_message)"
  [elicit]="behaviours: policy vs Sonnet per turn (dormancy escalation, PT handback)"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port) PORT="$2"; shift 2 ;;
    trace|medconv|elicit) APP="$1"; shift ;;
    --host)    HOST="$2"; shift 2 ;;
    --era)     ERA="$2";  shift 2 ;;
    --fg)      MODE="fg"; shift ;;
    --status)  MODE="status"; shift ;;
    --stop)    MODE="stop"; shift ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# The PYTHON process, not a wrapper shell. `pgrep -f` also matches any `bash -c '…'`
# whose command STRING contains "python3 app.py --port N" (nohup wrappers, this
# launcher itself), and those report ~0% CPU — misleading when checking health.
# argv[0] is the discriminator: for the real process it IS the interpreter.
pid_on_port() {
  # Match the SELECTED app's script (app.py or apps/med_conversations.py), not a literal
  # app.py -- otherwise --status/--stop silently report "not running" for every app but
  # the trace viewer. Ownership stays enforced by `pgrep -u "$USER"`.
  local _leaf="${SCRIPT##*/}"
  pgrep -u "$USER" -f "${_leaf//./\\.}" 2>/dev/null | while read -r p; do
    mapfile -d '' -t argv < "/proc/$p/cmdline" 2>/dev/null || continue
    [[ "${argv[0]:-}" == *python* ]] || continue          # argv[0] must be the interpreter
    [[ "${argv[1]:-}" == *"$_leaf" ]] || continue         # argv[1] must be the script
    for ((i = 2; i < ${#argv[@]}; i++)); do
      [[ "${argv[i]}" == "--port" && "${argv[i+1]:-}" == "$PORT" ]] && { echo "$p"; break; }
    done
  done | head -1
}

# Resolve the app BEFORE any mode runs: --status/--stop need SCRIPT + the app's default port.
SCRIPT="${APP_SCRIPT[$APP]}"
PORT="${PORT:-${APP_PORT[$APP]}}"

case "$MODE" in
  status)
    pid="$(pid_on_port || true)"
    if [[ -n "$pid" ]]; then
      echo "3wc app RUNNING  pid=$pid  port=$PORT  host=$(hostname)"
      ps -o pid=,etime=,%cpu=,rss= -p "$pid"
      ls -t "$LOG_DIR"/app_${PORT}_*.log 2>/dev/null | head -1 | sed 's/^/log: /'
    else
      echo "3wc app NOT running on port $PORT"
    fi
    exit 0 ;;
  stop)
    pid="$(pid_on_port || true)"
    # Only ever our own process — never another user's (shared cluster).
    if [[ -n "$pid" ]] && [[ "$(ps -o user= -p "$pid" | tr -d ' ')" == "$USER" ]]; then
      kill "$pid" && echo "stopped pid=$pid on port $PORT"
    else
      echo "nothing of yours to stop on port $PORT"
    fi
    exit 0 ;;
esac

# --- preflight -------------------------------------------------------------- #
[[ -d "$DATA_ROOT" ]] || { echo "FATAL: DATA_DIR not found: $DATA_ROOT" >&2; exit 1; }
[[ -f "$APP_DIR/$SCRIPT" ]] || { echo "FATAL: not found: $APP_DIR/$SCRIPT" >&2; exit 1; }
echo "app:   $APP — ${APP_DESC[$APP]}"
[[ -x "$PYTHON"    ]] || { echo "FATAL: python not executable: $PYTHON" >&2; exit 1; }
"$PYTHON" -c 'import gradio' 2>/dev/null || { echo "FATAL: gradio missing in $PYTHON" >&2; exit 1; }

if existing="$(pid_on_port || true)"; [[ -n "$existing" ]]; then
  echo "FATAL: port $PORT already serving (pid=$existing). Use --stop, or -p <other>." >&2
  exit 1
fi
if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "FATAL: port $PORT is in use by another process." >&2
  exit 1
fi

# Corpus scope. Default = .index_files (merged 0907 + 2206/2606). --era 0907
# restricts to that export, which is what the eval/test sets are built from.
export DATA_DIR="$DATA_ROOT"
if [[ -n "$ERA" ]]; then
  conv="$DATA_ROOT/raw/3_agents_${ERA}.jsonl"
  poc_candidates=("$DATA_ROOT/raw/precision_of_care_${ERA}.jsonl" "$DATA_ROOT"/raw/precision_of_care_*.jsonl)
  poc=""
  for c in "${poc_candidates[@]}"; do [[ -f "$c" ]] && { poc="$c"; break; }; done
  [[ -f "$conv" ]] || { echo "FATAL: no conv export for era '$ERA': $conv" >&2; exit 1; }
  [[ -n "$poc"  ]] || { echo "FATAL: no PoC export for era '$ERA'" >&2; exit 1; }
  export DATA_FILES="$conv,$poc"
  echo "corpus: era $ERA -> $(basename "$conv"), $(basename "$poc")"
else
  echo "corpus: merged (.index_files)"
fi

# medconv reads a multi-unit dump (thrive/bloom/pulse) and must not be unit-scoped;
# it also runs on the 3wc venv, which is what core/ + the screen vocabulary import against.
if [[ "$APP" == "medconv" || "$APP" == "elicit" ]]; then
  PYTHON="$MEDCONV_PYTHON"
  export ONLY_UNIT="${ONLY_UNIT:-All}"
else
  export ONLY_UNIT="${ONLY_UNIT:-thrive}"
fi
: "${ONLY_UNIT:?}"
# Keep every write under /home/sgsilva — /tmp is a shared mount. Overriding
# unconditionally is deliberate: a `GRADIO_TEMP_DIR=/tmp/...` inherited from the
# ambient shell would otherwise win over a `:-` default and silently put temp
# files back on /tmp. Set TMP_3WC to choose a different location.
export GRADIO_TEMP_DIR="${TMP_3WC:-/home/sgsilva/tmp/gradio_3wc}"
mkdir -p "$GRADIO_TEMP_DIR" "$LOG_DIR"

echo "host:  $(hostname)"
echo "index: $APP_DIR/.trace_index/  (git-excluded; delete to force a rebuild)"
echo "tmp:   $GRADIO_TEMP_DIR"

cd "$APP_DIR"
if [[ "$MODE" == "fg" ]]; then
  exec "$PYTHON" "$SCRIPT" --port "$PORT" --host "$HOST"
fi

LOG="$LOG_DIR/${APP}_${PORT}_$(date +%Y%m%d_%H%M%S).log"
nohup "$PYTHON" "$SCRIPT" --port "$PORT" --host "$HOST" > "$LOG" 2>&1 &
pid=$!
echo "started pid=$pid  port=$PORT"
echo "log:   $LOG"
echo
echo "watch:   tail -F $LOG"
echo "status:  $HERE/run_app.sh --status -p $PORT"
echo "stop:    $HERE/run_app.sh --stop   -p $PORT"
echo "tunnel:  ssh -N -L ${PORT}:$(hostname):${PORT} new-login-0   # then http://localhost:${PORT}"
