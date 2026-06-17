#!/usr/bin/env bash
# launch_app.sh — launch a Gradio inspection app by registry name.
#
# Usage:
#   launch_app.sh <app-name> [extra args passed to the script...]
#   launch_app.sh --list
#   launch_app.sh --status
#
# Examples:
#   launch_app.sh video-sft
#   launch_app.sh video-sft DEFAULT_JSONL=/mnt/data/sgsilva/datasets/app_video_datasets/mcqa_1405_skel_partition_inspect.jsonl
#   launch_app.sh vo-compare --results "SFT=/mnt/data/sgsilva/results/..."
#   launch_app.sh --list
#   launch_app.sh --status
#
# After launch, open in browser:
#   http://localhost:1<PORT>/   (local port = 10000 + remote port)
# e.g. remote port 7862 → http://localhost:17862/
#
# Registry: ~/utilities/apps/apps_registry.yaml

set -euo pipefail

REGISTRY="$(dirname "$(realpath "$0")")/apps_registry.yaml"

# ── helpers ───────────────────────────────────────────────────────────────────

die() { echo "ERROR: $*" >&2; exit 1; }

# Minimal YAML parser — reads a single scalar under apps.<name>.<key>
yaml_get() {
    local file=$1 app=$2 key=$3
    python3 - "$file" "$app" "$key" <<'PYEOF'
import sys, re
path, app, key = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(path).readlines()
in_app = False
for line in lines:
    stripped = line.rstrip()
    if re.match(r'^  ' + re.escape(app) + r'\s*:', stripped):
        in_app = True
        continue
    if in_app:
        if re.match(r'^  \w', stripped) and not re.match(r'^   ', stripped):
            break
        m = re.match(r'^    ' + re.escape(key) + r':\s*(.*)', stripped)
        if m:
            print(m.group(1).strip())
            sys.exit(0)
PYEOF
}

yaml_get_list() {
    local file=$1 app=$2 key=$3
    python3 - "$file" "$app" "$key" <<'PYEOF'
import sys, re
path, app, key = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(path).readlines()
in_app = False
in_key = False
for line in lines:
    stripped = line.rstrip()
    if re.match(r'^  ' + re.escape(app) + r'\s*:', stripped):
        in_app = True; continue
    if in_app:
        if re.match(r'^  \w', stripped) and not re.match(r'^   ', stripped):
            break
        if re.match(r'^    ' + re.escape(key) + r'\s*:', stripped):
            in_key = True; continue
        if in_key:
            m = re.match(r'^      - (.*)', stripped)
            if m:
                print(m.group(1).strip())
            elif stripped and not stripped.startswith('      '):
                break
PYEOF
}

yaml_get_env() {
    local file=$1 app=$2
    python3 - "$file" "$app" <<'PYEOF'
import sys, re
path, app = sys.argv[1], sys.argv[2]
lines = open(path).readlines()
in_app = False
in_env = False
for line in lines:
    stripped = line.rstrip()
    if re.match(r'^  ' + re.escape(app) + r'\s*:', stripped):
        in_app = True; continue
    if in_app:
        if re.match(r'^  \w', stripped) and not re.match(r'^   ', stripped):
            break
        if re.match(r'^    env\s*:', stripped):
            in_env = True; continue
        if in_env:
            m = re.match(r'^      (\w+):\s*(.*)', stripped)
            if m:
                print(f"{m.group(1)}={m.group(2).strip()}")
            elif stripped and not stripped.startswith('      '):
                break
PYEOF
}

list_apps() {
    python3 - "$REGISTRY" <<'PYEOF'
import sys, re
lines = open(sys.argv[1]).readlines()
in_apps = False
current = None
label = None
port = None
for line in lines:
    s = line.rstrip()
    if s == 'apps:':
        in_apps = True; continue
    if in_apps:
        m = re.match(r'^  (\S+):\s*$', s)
        if m:
            if current and label:
                p = f":{port}" if port else ""
                print(f"  {current:<22} {label}{p}")
            current = m.group(1); label = None; port = None
        m2 = re.match(r'^    label:\s*"?(.*?)"?\s*$', s)
        if m2: label = m2.group(1)
        m3 = re.match(r'^    port:\s*(\d+)', s)
        if m3: port = m3.group(1)
if current and label:
    p = f"  → port {port}" if port else ""
    print(f"  {current:<22} {label}{p}")
PYEOF
}

# Collect all (name, port) pairs from the registry
all_app_ports() {
    python3 - "$REGISTRY" <<'PYEOF'
import sys, re
lines = open(sys.argv[1]).readlines()
in_apps = False
current = None
port = None
label = None
for line in lines:
    s = line.rstrip()
    if s == 'apps:':
        in_apps = True; continue
    if in_apps:
        m = re.match(r'^  (\S+):\s*$', s)
        if m:
            if current and port:
                print(f"{current} {port} {label or current}")
            current = m.group(1); port = None; label = None
        m2 = re.match(r'^    port:\s*(\d+)', s)
        if m2: port = m2.group(1)
        m3 = re.match(r'^    label:\s*"?(.*?)"?\s*$', s)
        if m3: label = m3.group(1)
if current and port:
    print(f"{current} {port} {label or current}")
PYEOF
}

status_apps() {
    # Run on the login node — check which registered ports have a live process.
    # If called from a worker, SSH to login-1 to check there.
    local check_host=""
    if [[ "$(hostname)" == worker-* ]]; then
        check_host="${LOGIN_NODE}"
    fi

    echo ""
    echo "App status on ${check_host:-$(hostname)}:"
    echo ""
    printf "  %-22s %-6s %-10s %s\n" "NAME" "PORT" "STATUS" "LABEL"
    printf "  %-22s %-6s %-10s %s\n" "----" "----" "------" "-----"

    while read -r name port label; do
        local local_port=$(( port + 10000 ))
        local status pid
        if [[ -n "$check_host" ]]; then
            pid=$(ssh "$check_host" "lsof -ti:${port} 2>/dev/null || true")
        else
            pid=$(lsof -ti:"${port}" 2>/dev/null || true)
        fi
        if [[ -n "$pid" ]]; then
            status="✓ running"
        else
            status="· stopped"
        fi
        printf "  %-22s %-6s %-10s %s\n" "$name" "$port" "$status" "$label"
    done < <(all_app_ports)
    echo ""
}

# Health-check: poll the Gradio app until it responds or timeout.
# Gradio serves a root page; a 200 means it's up.
health_check() {
    local port=$1 label=$2
    local timeout=30 interval=2 elapsed=0
    echo -n "  Waiting for ${label} to be ready"
    while (( elapsed < timeout )); do
        if curl -sf --max-time 2 "http://localhost:${port}/" -o /dev/null 2>/dev/null; then
            echo " ✓"
            return 0
        fi
        echo -n "."
        sleep "$interval"
        (( elapsed += interval ))
    done
    echo " ✗ (did not respond within ${timeout}s — check logs)"
    return 1
}

# ── main ──────────────────────────────────────────────────────────────────────

LOGIN_NODE="login-1"
TMUX_SESSION="app"

if [[ $# -eq 0 || "$1" == "--list" || "$1" == "-l" ]]; then
    echo ""
    echo "Available apps (registry: $REGISTRY):"
    echo ""
    list_apps
    echo ""
    echo "Usage: launch_app.sh <app-name> [KEY=VALUE...] [--extra-arg value...]"
    echo "       launch_app.sh --status"
    exit 0
fi

if [[ "$1" == "--status" || "$1" == "-s" ]]; then
    status_apps
    exit 0
fi

APP_NAME="$1"; shift

# ── worker-node redirect ───────────────────────────────────────────────────────
# If running on a compute node (hostname = worker-*), the app must run on the
# login node where the SSH tunnels land. Transparently re-dispatch via SSH into
# the persistent 'app' tmux session on login-1 and exit.
if [[ "$(hostname)" == worker-* ]]; then
    WINDOW_NAME="$APP_NAME"
    # Look up registry fields locally (shared /home) so we can print the banner
    _PORT=$(yaml_get "$REGISTRY" "$APP_NAME" "port")
    _LOCAL_PORT=$(( _PORT + 10000 ))
    _LABEL=$(yaml_get "$REGISTRY" "$APP_NAME" "label")
    _GOAL=$(yaml_get "$REGISTRY" "$APP_NAME" "goal")
    _SCRIPT=$(yaml_get "$REGISTRY" "$APP_NAME" "script")
    _REPO=$(yaml_get "$REGISTRY" "$APP_NAME" "repo")
    _VENV=$(yaml_get "$REGISTRY" "$APP_NAME" "venv")
    # Rebuild a safely-quoted arg string to pass through SSH
    ARGS=""
    for arg in "$@"; do
        ARGS="${ARGS} $(printf '%q' "$arg")"
    done
    ssh "${LOGIN_NODE}" bash << REMOTE
set -euo pipefail
if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${TMUX_SESSION}"
fi
# Kill existing window of same name if present (clean restart)
tmux kill-window -t "${TMUX_SESSION}:${WINDOW_NAME}" 2>/dev/null || true
tmux new-window -t "${TMUX_SESSION}" -n "${WINDOW_NAME}" \
    "/home/sgsilva/utilities/apps/launch_app.sh ${APP_NAME}${ARGS}; echo ''; echo '[app exited — press Enter to close]'; read"
REMOTE
    echo ""
    echo "┌─────────────────────────────────────────────────────────────"
    echo "│  ${_LABEL}"
    echo "│  ${_GOAL}"
    echo "│  Script  : ${_REPO}/${_SCRIPT}"
    echo "│  Venv    : ${_VENV}"
    echo "│  Port    : ${_PORT} (remote) → ${_LOCAL_PORT} (local)"
    echo "│  Browser : http://localhost:${_LOCAL_PORT}/"
    echo "│  Logs    : ssh ${LOGIN_NODE} -t 'tmux attach -t ${TMUX_SESSION}'"
    echo "└─────────────────────────────────────────────────────────────"
    echo ""
    # Health-check from the worker via SSH tunnel to login-1
    if ssh "${LOGIN_NODE}" "curl -sf --max-time 2 http://localhost:${_PORT}/ -o /dev/null 2>/dev/null"; then
        echo "  ✓ App is already up (previous instance was restarted)"
    else
        echo -n "  Waiting for ${_LABEL} to be ready"
        timeout=30; interval=2; elapsed=0
        while (( elapsed < timeout )); do
            if ssh "${LOGIN_NODE}" "curl -sf --max-time 2 http://localhost:${_PORT}/ -o /dev/null 2>/dev/null"; then
                echo " ✓  ready"
                break
            fi
            echo -n "."
            sleep "$interval"
            (( elapsed += interval ))
        done
        if (( elapsed >= timeout )); then
            echo " ✗  did not respond within ${timeout}s"
            echo "  Check logs: ssh ${LOGIN_NODE} -t 'tmux attach -t ${TMUX_SESSION}'"
        fi
    fi
    exit 0
fi

# ── local launch (on login node) ──────────────────────────────────────────────

# Validate app exists in registry
REPO=$(yaml_get "$REGISTRY" "$APP_NAME" "repo")
[[ -n "$REPO" ]] || die "Unknown app '$APP_NAME'. Run 'launch_app.sh --list' to see available apps."

SCRIPT=$(yaml_get "$REGISTRY" "$APP_NAME" "script")
PORT=$(yaml_get "$REGISTRY" "$APP_NAME" "port")
VENV=$(yaml_get "$REGISTRY" "$APP_NAME" "venv")
LABEL=$(yaml_get "$REGISTRY" "$APP_NAME" "label")
GOAL=$(yaml_get "$REGISTRY" "$APP_NAME" "goal")

[[ -d "$REPO" ]]   || die "Repo directory not found: $REPO"
[[ -f "$REPO/$SCRIPT" ]] || die "Script not found: $REPO/$SCRIPT"

# Separate KEY=VALUE overrides from extra args
ENV_OVERRIDES=()
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" =~ ^[A-Z_][A-Z0-9_]*= ]]; then
        ENV_OVERRIDES+=("$arg")
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# Kill any existing process on the port
if lsof -ti:"$PORT" &>/dev/null; then
    echo "  Killing existing process on port $PORT..."
    lsof -ti:"$PORT" | xargs -r kill -9
    sleep 0.3
fi

# Build env prefix from registry env block + command-line overrides
ENV_PREFIX=""
while IFS= read -r kv; do
    [[ -n "$kv" ]] && ENV_PREFIX="$kv $ENV_PREFIX"
done < <(yaml_get_env "$REGISTRY" "$APP_NAME")
for kv in "${ENV_OVERRIDES[@]}"; do
    ENV_PREFIX="$kv $ENV_PREFIX"
done

# Build registry args
REGISTRY_ARGS=()
while IFS= read -r item; do
    [[ -n "$item" ]] && REGISTRY_ARGS+=("$item")
done < <(yaml_get_list "$REGISTRY" "$APP_NAME" "args")

# Compose and launch
LOCAL_PORT=$((PORT + 10000))
echo ""
echo "┌─────────────────────────────────────────────────────────────"
echo "│  $LABEL"
echo "│  $GOAL"
echo "│  Script  : $REPO/$SCRIPT"
echo "│  Venv    : $VENV"
echo "│  Port    : $PORT (remote) → $LOCAL_PORT (local)"
echo "│  Browser : http://localhost:$LOCAL_PORT/"
echo "└─────────────────────────────────────────────────────────────"
echo ""

cd "$REPO"
CMD="$VENV $SCRIPT ${REGISTRY_ARGS[*]+"${REGISTRY_ARGS[@]}"} ${EXTRA_ARGS[*]+"${EXTRA_ARGS[@]}"}"
[[ -n "$ENV_PREFIX" ]] && echo "  env: $ENV_PREFIX"
echo "  cmd: $CMD"
echo ""

eval "$ENV_PREFIX $CMD"
