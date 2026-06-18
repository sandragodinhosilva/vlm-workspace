#!/usr/bin/env bash
# log_run.sh — canonical run-logging helper for /mnt/data/sgsilva/logs.
# SOURCE this file; do not execute it.  Honors: no /tmp, no node assumptions,
# distinct failure sentinels, NFS-atomic index appends.
#
#   source /home/sgsilva/utilities/logs-utils/log_run.sh
#   LOG=$(log_start <category> <run_name>)        # single file -> echoes .log path
#   RUNDIR=$(log_start --dir <category> <name>)   # multi-component -> echoes run dir
#   log_end "$LOG" "$?"                            # footer + flips index status
#
# Categories (closed allowlist): grpo sft sam3d dataset export eval oracle serve claude misc
#
# Claude usage (after sourcing):
#   clog <category> <run_name> -- <command ...>
#   e.g. source /home/sgsilva/utilities/logs-utils/log_run.sh && clog dataset build_nonreasoning_mix_aux12k -- /home/sgsilva/vlm-post-training-home-venv/bin/python build_mix.py --foo bar

LOG_ROOT="/mnt/data/sgsilva/logs"
LOG_INDEX="$LOG_ROOT/index.jsonl"
LOG_LOCK="$LOG_ROOT/index.lock"
_LOG_CATS=" grpo sft sam3d dataset export oracle eval serve claude misc "
_LOG_ROTATE_BYTES=$(( 2 * 1024 * 1024 * 1024 ))   # 2 GiB per segment

_log_err()  { printf '[log_run] %s\n' "$*" >&2; }
_log_now()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
_log_epoch(){ date -u +%s; }

# JSON-escape a scalar string (handles \, ", control chars minimally).
_log_jesc() { local s=${1//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/ }; printf '%s' "$s"; }

# Atomic append one line to the central index under flock; fall back to per-run pending file.
_log_index_append() {
    local line="$1" rundir="$2"
    { exec 9>"$LOG_LOCK"; } 2>/dev/null
    if flock -w 10 9 2>/dev/null; then
        printf '%s\n' "$line" >>"$LOG_INDEX"
        flock -u 9
    else
        _log_err "index lock timeout (NFS?) -> buffering to .index_pending.jsonl"
        printf '%s\n' "$line" >>"$rundir/.index_pending.jsonl"
    fi
}

# Background watcher: rotate $1 when it exceeds the segment size.
_log_size_watch() {
    local f="$1"
    while [[ -e "$f" ]]; do
        sleep 60
        local sz; sz=$(stat -c%s "$f" 2>/dev/null) || break
        if (( sz > _LOG_ROTATE_BYTES )); then
            local i=1; while [[ -e "$f.$i" ]]; do i=$((i+1)); done
            mv "$f" "$f.$i" 2>/dev/null && : >"$f"
            _log_err "rotated $(basename "$f") -> .$i"
        fi
    done
}

log_start() {
    local as_dir=0
    if [[ "$1" == "--dir" ]]; then as_dir=1; shift; fi
    local cat="$1" name="$2"
    if [[ -z "$cat" || -z "$name" ]]; then _log_err "usage: log_start [--dir] <category> <run_name>"; return 2; fi
    case "$_LOG_CATS" in *" $cat "*) ;; *) _log_err "unknown category '$cat' (allowed:$_LOG_CATS)"; return 2;; esac

    # sanitize name: lowercase, allow [a-z0-9_.-], collapse others to _, cap 80
    local rn; rn=$(printf '%s' "$name" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_.-' '_' | cut -c1-80)

    # origin: SLURM job > claude > interactive node+pid.  Node from hostname (never assumed).
    local origin node pid="$$"
    node=$(hostname -s 2>/dev/null || echo unknown)
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        origin="j${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
        origin="j${SLURM_JOB_ID}"
    elif [[ "${LOG_ORIGIN:-}" == "claude" ]]; then
        origin="claude_p${pid}"
    else
        origin="${node}_p${pid}"
    fi

    local date; date=$(date -u +%Y-%m-%d)
    local stamp; stamp=$(date -u +%Y-%m-%dT%H%M)
    local base="${rn}__${origin}__${stamp}"
    local dir="$LOG_ROOT/$cat/$date"
    mkdir -p "$dir" || { _log_err "mkdir failed: $dir"; return 1; }

    local logfile rundir meta
    if (( as_dir )); then
        rundir="$dir/$base"; mkdir -p "$rundir"
        logfile="$rundir/run.log"; meta="$rundir/meta.json"
    else
        rundir="$dir"; logfile="$dir/$base.log"; meta="$dir/$base.meta.json"
    fi
    : >"$logfile"

    local start_epoch; start_epoch=$(_log_epoch)
    local started; started=$(_log_now)
    # meta.json: full record (superset of index line)
    cat >"$meta" <<EOF
{"category":"$cat","run_name":"$(_log_jesc "$rn")","origin":"$origin","node":"$node","pid":$pid,
 "slurm_job_id":"${SLURM_JOB_ID:-}","log_origin":"${LOG_ORIGIN:-shell}",
 "started":"$started","start_epoch":$start_epoch,"status":"running",
 "logfile":"$(_log_jesc "$logfile")","cwd":"$(_log_jesc "$PWD")",
 "cmd":"$(_log_jesc "${LOG_CMD:-$0 $*}")"}
EOF

    # header in the log itself
    {   echo "==== RUN START ===="
        echo "category : $cat"
        echo "run_name : $rn"
        echo "origin   : $origin   (node=$node pid=$pid jobid=${SLURM_JOB_ID:-none})"
        echo "started  : $started"
        echo "cmd      : ${LOG_CMD:-$0 $*}"
        echo "cwd      : $PWD"
        echo "logfile  : $logfile"
        echo "==================="
    } >>"$logfile"

    # index: running line
    _log_index_append \
      "{\"ts\":\"$started\",\"epoch\":$start_epoch,\"category\":\"$cat\",\"run_name\":\"$(_log_jesc "$rn")\",\"origin\":\"$origin\",\"node\":\"$node\",\"status\":\"running\",\"logfile\":\"$(_log_jesc "$logfile")\",\"exit\":null,\"dur_s\":null}" \
      "$rundir"

    # state for trap + size watcher
    export _LOG_RUN_META="$meta" _LOG_RUN_FILE="$logfile" _LOG_RUN_RUNDIR="$rundir" _LOG_RUN_START="$start_epoch" _LOG_RUN_ENDED=0 _LOG_RUN_PID="$$"
    _log_size_watch "$logfile" & disown 2>/dev/null
    export _LOG_RUN_WATCH=$!
    trap '_log_run_trap' EXIT INT TERM

    # echo the canonical path (dir for --dir, else the .log)
    if (( as_dir )); then printf '%s\n' "$rundir"; else printf '%s\n' "$logfile"; fi
}

# Internal: flip the LAST index line for this logfile to a terminal status.
# Idempotent: if _LOG_RUN_ENDED is already 1 or PID doesn't match, skip silently.
_log_finalize() {
    [[ "${_LOG_RUN_ENDED:-0}" == "1" ]] && return
    [[ -n "${_LOG_RUN_PID:-}" && "${_LOG_RUN_PID}" != "$$" ]] && return
    local status="$1" code="$2" logfile="$3" meta="$4" rundir="$5"
    local end_epoch dur ended
    end_epoch=$(_log_epoch); ended=$(_log_now)
    dur=$(( end_epoch - ${_LOG_RUN_START:-end_epoch} ))
    # footer in the log
    {   echo "==== RUN END ===="
        echo "status   : $status"
        echo "exit     : $code"
        echo "ended    : $ended"
        echo "duration : ${dur}s"
        echo "================="
    } >>"$logfile" 2>/dev/null
    # rewrite meta status (sed in place is fine; small file, single writer)
    sed -i "s/\"status\":\"running\"/\"status\":\"$status\"/" "$meta" 2>/dev/null
    # extract category from meta.json for the terminal index line
    local meta_cat; meta_cat=$(/bin/grep -o '"category":"[^"]*"' "$meta" 2>/dev/null | head -1 | cut -d'"' -f4)
    # append a terminal index line (append-only; query takes the LAST line per logfile)
    _log_index_append \
      "{\"ts\":\"$ended\",\"epoch\":$end_epoch,\"category\":\"$meta_cat\",\"run_name\":\"finalize\",\"origin\":\"\",\"node\":\"\",\"status\":\"$status\",\"logfile\":\"$(_log_jesc "$logfile")\",\"exit\":$code,\"dur_s\":$dur}" \
      "$rundir"
}

log_end() {
    local logfile="$1" code="${2:-0}"
    local status; if [[ "$code" == "0" ]]; then status="done"; else status="failed"; fi
    _log_finalize "$status" "$code" "$logfile" "$_LOG_RUN_META" "$_LOG_RUN_RUNDIR"
    export _LOG_RUN_ENDED=1
    [[ -n "${_LOG_RUN_WATCH:-}" ]] && kill "$_LOG_RUN_WATCH" 2>/dev/null
    trap - EXIT INT TERM
}

# Trap: only fires if log_end was NOT called (crash / kill / set -e abort).
_log_run_trap() {
    local rc=$?
    [[ "${_LOG_RUN_ENDED:-0}" == "1" ]] && return
    # Guard: only fire for the shell that called log_start (PID check prevents stale inherited traps)
    [[ "${_LOG_RUN_PID:-}" != "$$" ]] && return
    local status; if [[ $rc -eq 130 || $rc -eq 143 ]]; then status="killed"; else status="failed"; fi
    export _LOG_RUN_ENDED=1   # mark done BEFORE finalize so log_end is a no-op if called after
    _log_finalize "$status" "$rc" "$_LOG_RUN_FILE" "$_LOG_RUN_META" "$_LOG_RUN_RUNDIR"
    [[ -n "${_LOG_RUN_WATCH:-}" ]] && kill "$_LOG_RUN_WATCH" 2>/dev/null
}

# clog <category> <run_name> -- <command...>
# Runs the command with LOG_ORIGIN=claude, tees combined output to the canonical log,
# records cmd/args/exit/duration. Echoes the log path on stderr for the user.
clog() {
    local cat="$1" name="$2"; shift 2
    [[ "$1" == "--" ]] && shift
    export LOG_ORIGIN=claude
    export LOG_CMD="$*"
    local LOG; LOG=$(log_start "$cat" "$name")
    _log_err "logging to: $LOG"
    # Redirect stdout+stderr directly to the log (no tee — avoids process-substitution
    # EXIT-trap side effects in bash). Output is captured; watch live with: tail -F "$LOG"
    "$@" >> "$LOG" 2>&1
    local rc=$?
    log_end "$LOG" "$rc"
    unset LOG_ORIGIN LOG_CMD
    return "$rc"
}
