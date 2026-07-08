"""
Shared cluster-scan for live vLLM servers — find-live-server + owner-attribution
logic used by any app that lets Sandra pick a served model to query.

Extracted from vibe_test.py (2026-07-08 homogenization pass — see
GRADIO_APPS_REPORT.md § Priority 1). vibe_test.py was already the de facto
shared source (gradio_app_multiturn.py imported straight from it); this module
formalizes that into scripts/, matching nav_widgets.py's home, so a future app
doesn't have to import a launched Gradio app's module to get this feature.

Model-CALLING (vLLM chat completion / Vertex-gemini routing) is a separate,
larger shared surface already centralized in vlm-post-training/inference/query_server.py
— reasoning-prompt and multiturn-tools both use it. vibe_test.py doesn't (its
venv lacks the GCloud SDK, so it subprocesses to the eval venv instead via
scripts/_vertex_call.py) — that's a real, venv-forced divergence, not
duplication worth merging here.

Pure functions + one small SSH/HTTP-probing surface — no gr.State wiring, no
Gradio imports. Caller owns state and wires .click()/.change() events.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import requests

WORKER_NODES = [f"worker-{i}" for i in range(32)]  # worker-0 … worker-31
VLLM_PORT = 8000
VLLM_PORTS = [8000, 8001, 8002, 8003]  # scan a small range so non-8000 servers show up
SCAN_TIMEOUT = 2.0  # seconds per node


def get_vllm_owner(node: str, port: int = VLLM_PORT) -> str:
    """SSH to node and return the user OWNING the server on this PORT.

    A node can host several vLLM servers on different ports (e.g. worker-30 with
    afarinhas on :8000 and sgsilva on :8001), so resolve the owner of the process
    actually LISTENING on `port` — not just the first vllm process on the node.

    KNOWN LIMITATION (found 2026-07-08): `ss -ltnp` only reports a PID for
    sockets owned by the SSHing user (or root) — an unprivileged `sgsilva`
    session cannot see the PID for another user's listening socket, and no
    `sudo`/`lsof` escalation is available on this cluster. Per
    [[feedback_no_silent_fail]], a failed lookup must return a distinct
    sentinel, not a plausible-looking wrong answer — so this returns
    "unknown" whenever the port-specific PID can't be resolved, rather
    than guessing from a node-wide process list (the old bug: it silently
    misattributed one user's port to whichever vllm process happened to
    appear first for the SSHing user).
    """
    cmd = (
        f"pid=$(ss -ltnp 2>/dev/null | grep ':{port} ' "
        f"| sed -n 's/.*pid=\\([0-9]*\\).*/\\1/p' | head -1); "
        f"u=$([ -n \"$pid\" ] && ps -o user= -p \"$pid\" 2>/dev/null | tr -d ' '); "
        f"if [ -n \"$u\" ]; then echo \"$u\"; else echo 'unknown'; fi"
    )
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=2", "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes", node, cmd],
            capture_output=True, text=True, timeout=5,
        )
        owner = result.stdout.strip()
        return owner if owner else "unknown"
    except Exception:
        return "unknown"


def probe_node(node: str, port: int = VLLM_PORT) -> Optional[Tuple[str, int, List[str], str]]:
    """Return (node, port, [model_ids], owner) if a vLLM server is live, else None."""
    url = f"http://{node}:{port}/v1/models"
    try:
        r = requests.get(url, timeout=SCAN_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            models = [m["id"] for m in data.get("data", [])]
            owner = get_vllm_owner(node, port)
            return node, port, models, owner
    except Exception:
        pass
    return None


def scan_cluster() -> Tuple[str, list, List[str]]:
    """
    Probe all worker nodes in parallel.
    Returns (scan_summary_text, results, choices).
    results: list of (node, port, [model_ids], owner)
    choices: "worker-N | owner | <model name>" — shown in dropdown
    """
    results = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(probe_node, node, port): (node, port)
                   for node in WORKER_NODES for port in VLLM_PORTS}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)

    if not results:
        return "No vLLM servers found on worker-0 … worker-31", [], []

    results.sort(key=lambda x: (int(x[0].split("-")[1]), x[1]))

    lines = []
    choices = []
    for node, port, models, owner in results:
        # only annotate the port when it's not the default, to keep labels clean
        port_tag = "" if port == VLLM_PORT else f":{port}"
        for mid in models:
            short = mid.split("/")[-1]  # short name for display
            label = f"{node}{port_tag} | {owner} | {short}"
            choices.append(label)
            lines.append(f"✓ {node}:{port}  [{owner}]  {short}")

    summary = f"Found {len(results)} live server(s):\n" + "\n".join(lines)
    return summary, results, choices


def apply_scan_selection(selected: str, scan_results: list) -> Tuple[str, str]:
    """Given a selection 'worker-N[:PORT] | owner | short_name', return (server_url, full_model_id)."""
    if not selected or not scan_results:
        return "", ""
    parts = [p.strip() for p in selected.split("|")]
    if len(parts) < 3:
        return "", ""
    node_part = parts[0]            # "worker-N" or "worker-N:PORT"
    short = parts[2]
    if ":" in node_part:
        sel_node, sel_port = node_part.split(":", 1)
        sel_port = int(sel_port)
    else:
        sel_node, sel_port = node_part, VLLM_PORT
    for node, port, models, owner in scan_results:
        if node == sel_node and port == sel_port:
            for mid in models:
                if mid.split("/")[-1] == short:
                    return f"http://{node}:{port}", mid
            return f"http://{node}:{port}", models[0] if models else ""
    return "", ""
