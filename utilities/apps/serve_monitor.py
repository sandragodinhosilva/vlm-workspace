#!/usr/bin/env python3
"""
serve_monitor.py — Live vLLM serving monitor + settings frontend.

Two purposes in one app (port 7875):

  TAB 1 "Monitor"  — watch what a served model is DOING right now: requests
    running/waiting/preempted, KV-cache usage, rolling prompt/generation tok/s
    (delta between refreshes — NOT the lifetime average, which hides the current
    state), avg TTFT (prefill) and avg time-per-output-token (decode). Auto
    refreshes every 5s (+ manual button). Plotly time-series so you can WATCH an
    experiment's effect on throughput/load.

  TAB 2 "Settings & Ask"  — the KEY prefill/decode/serving settings for the
    selected server, shown TWO WAYS side by side: (a) LIVE — what the running
    engine actually reports (/v1/models + on-disk config.json dtype/quant); and
    (b) SCRIPT — what start_vllm_server.sh WOULD pass for that model family, so
    you can spot drift between configured and running. Plus a chat box that
    queries the served model itself (e.g. "what is your context length?").

Reuses scripts/live_inference.py (scan_cluster / apply_scan_selection) for
server discovery — same picker as vibe-test. Metrics are read over plain HTTP
(compute nodes are reachable by hostname from the login node); only the on-disk
config.json dtype/quant lookup uses SSH.

Usage:
    python utilities/apps/serve_monitor.py
    # → http://localhost:7878/   (local: http://localhost:17878/ via tunnel)
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
import requests
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from live_inference import scan_cluster, apply_scan_selection  # noqa: E402

DEFAULT_SERVER = os.environ.get("MONITOR_DEFAULT_SERVER", "http://worker-20:8000")
REFRESH_SECS = 5
HISTORY_LEN = 120  # keep ~10 min of 5s samples
MY_USER = os.environ.get("USER", "sgsilva")  # scan shows only MY servers

# Path where start_vllm_server.sh lives, for the SCRIPT-declared settings column.
SERVE_SCRIPT = "/home/sgsilva/utilities/serve/start_vllm_server.sh"


# ── metrics parsing ─────────────────────────────────────────────────────────

def _parse_prom(text: str) -> dict:
    """Parse the vLLM Prometheus /metrics text into {metric_name: float}.

    Keeps only vllm:* lines and ignores labels (single-engine servers). For a
    metric that appears once (gauge) we take its value; for _sum/_count pairs we
    keep both so the caller can compute an average.
    """
    out = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = re.match(r"^(vllm:[a-zA-Z0-9_]+)\{[^}]*\}\s+([0-9.eE+-]+)$", line)
        if not m:
            m = re.match(r"^(vllm:[a-zA-Z0-9_]+)\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name, val = m.group(1), m.group(2)
        try:
            out[name] = float(val)
        except ValueError:
            continue
    return out


def fetch_metrics(server: str) -> dict:
    """Return parsed /metrics, or {'_error': msg} on failure (distinct sentinel —
    never a plausible empty-but-healthy-looking dict)."""
    if not server:
        return {"_error": "no server selected"}
    try:
        r = requests.get(f"{server}/metrics", timeout=6)
        if r.status_code != 200:
            return {"_error": f"/metrics HTTP {r.status_code}"}
        parsed = _parse_prom(r.text)
        parsed["_raw"] = r.text  # kept so bucket-based percentiles can be computed
        return parsed
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def _histogram_percentiles(raw: str, metric: str, ps=(50, 90, 99)) -> dict:
    """Parse a vLLM _bucket histogram from raw /metrics text and return the
    bucket upper-edge at which each percentile is reached, plus n. Bucket edges
    are coarse, so a value means '≤ this many tokens'."""
    edges = []
    for line in (raw or "").splitlines():
        if not line.startswith(f"vllm:{metric}_bucket"):
            continue
        m = re.search(r'le="([^"]+)"', line)
        if not m:
            continue
        le = float("inf") if "Inf" in m.group(1) else float(m.group(1))
        try:
            cum = float(line.split()[-1])
        except ValueError:
            continue
        edges.append((le, cum))
    if not edges:
        return {}
    edges.sort(key=lambda x: x[0])
    total = edges[-1][1]
    if total <= 0:
        return {"_n": 0}
    out = {"_n": int(total)}
    for p in ps:
        target = p / 100.0 * total
        hit = edges[-1][0]
        for le, cum in edges:
            if cum >= target:
                hit = le
                break
        out[p] = hit
    return out


def _fmt_tok(x):
    if x == float("inf"):
        return "∞"
    return f"{int(x):,}"


def fetch_served_model(server: str) -> str:
    try:
        r = requests.get(f"{server}/v1/models", timeout=6)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                return data[0]["id"]
    except Exception:  # noqa: BLE001
        pass
    return ""


def _avg(m: dict, sum_key: str, count_key: str):
    c = m.get(count_key, 0.0)
    if c and c > 0:
        return m.get(sum_key, 0.0) / c
    return None


# ── on-disk config (dtype / quant) via SSH to the node ──────────────────────

def _node_from_server(server: str):
    m = re.match(r"https?://([^:/]+)", server or "")
    return m.group(1) if m else None


def fetch_disk_config(server: str, model_id: str) -> dict:
    """SSH to the serving node and read dtype/quant from the model's config.json.
    Returns {} on any failure (caller renders 'unavailable')."""
    node = _node_from_server(server)
    if not node or not model_id or not model_id.startswith("/"):
        return {}
    cfg = str(Path(model_id) / "config.json")
    cmd = (
        f"python3 - <<'PY'\n"
        f"import json\n"
        f"try:\n"
        f"    d=json.load(open({cfg!r}))\n"
        f"    tc=d.get('text_config',{{}})\n"
        f"    dt=d.get('dtype') or d.get('torch_dtype') or tc.get('dtype') or tc.get('torch_dtype') or 'n/a'\n"
        f"    md=d.get('mamba_ssm_dtype') or tc.get('mamba_ssm_dtype') or 'n/a'\n"
        f"    qz='yes' if (d.get('quantization_config') or tc.get('quantization_config')) else 'none'\n"
        f"    print('dtype', dt)\n"
        f"    print('quant', qz)\n"
        f"    print('mamba_ssm_dtype', md)\n"
        f"except Exception as e:\n"
        f"    print('err', e)\n"
        f"PY"
    )
    try:
        res = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", node, cmd],
            capture_output=True, text=True, timeout=15,
        )
        out = {}
        for ln in res.stdout.splitlines():
            parts = ln.split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
        return out
    except Exception:  # noqa: BLE001
        return {}


# ── nvidia-smi on the serving node ──────────────────────────────────────────

def fetch_gpus(server: str) -> list:
    """SSH to the node and return a list of per-GPU dicts (index/util/mem/power/temp).
    Empty list on failure (caller renders 'unavailable')."""
    node = _node_from_server(server)
    if not node:
        return []
    q = ("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,"
         "power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null")
    try:
        res = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", node, q],
            capture_output=True, text=True, timeout=12,
        )
        gpus = []
        for ln in res.stdout.strip().splitlines():
            f = [x.strip() for x in ln.split(",")]
            if len(f) == 6:
                gpus.append({"idx": f[0], "util": f[1], "mem_used": f[2],
                             "mem_total": f[3], "power": f[4], "temp": f[5]})
        return gpus
    except Exception:  # noqa: BLE001
        return []


def render_gpus(gpus: list) -> str:
    if not gpus:
        return "<div style='color:#777'>GPU stats unavailable (ssh/nvidia-smi).</div>"
    rows = ""
    for g in gpus:
        try:
            util = int(g["util"]); mem_pct = 100 * int(g["mem_used"]) / max(1, int(g["mem_total"]))
        except ValueError:
            util, mem_pct = 0, 0
        bar = f"<div style='background:#e0e0e0;border-radius:4px;width:80px;height:10px;display:inline-block'><div style='background:{'#e53935' if util>90 else '#43a047'};width:{util}%;height:10px;border-radius:4px'></div></div>"
        rows += (f"<tr><td style='padding:4px 8px'>GPU {g['idx']}</td>"
                 f"<td style='padding:4px 8px'>{bar} {util}%</td>"
                 f"<td style='padding:4px 8px'>{int(int(g['mem_used'])/1024)}/{int(int(g['mem_total'])/1024)} GB ({mem_pct:.0f}%)</td>"
                 f"<td style='padding:4px 8px'>{g['power']} W</td>"
                 f"<td style='padding:4px 8px'>{g['temp']}°C</td></tr>")
    return (f"<table style='border-collapse:collapse;font-family:system-ui;font-size:13px'>"
            f"<tr style='background:#e8eaed'><th style='padding:4px 8px'>GPU</th>"
            f"<th style='padding:4px 8px'>Util</th><th style='padding:4px 8px'>Memory</th>"
            f"<th style='padding:4px 8px'>Power</th><th style='padding:4px 8px'>Temp</th></tr>{rows}</table>")


def render_token_stats(m: dict) -> str:
    """Avg + p50/p90/p99 input and output token counts per request, from the
    request_{prompt,generation}_tokens histograms. Answers 'how big are my
    requests' — and, next to --max-model-len, whether it's over-provisioned."""
    if "_error" in m:
        return ""
    raw = m.get("_raw", "")
    avg_in = _avg(m, "vllm:request_prompt_tokens_sum", "vllm:request_prompt_tokens_count")
    avg_out = _avg(m, "vllm:request_generation_tokens_sum", "vllm:request_generation_tokens_count")
    pin = _histogram_percentiles(raw, "request_prompt_tokens")
    pout = _histogram_percentiles(raw, "request_generation_tokens")
    n = pin.get("_n", 0)

    def cell(label, avg, pc):
        return (f"<div style='flex:1;min-width:170px;padding:12px;background:#eef2f7;border-radius:8px'>"
                f"<div style='font-size:12px;color:#555'>{label}</div>"
                f"<div style='font-size:24px;font-weight:700'>{_fmt_tok(avg) if avg else '—'}"
                f"<span style='font-size:12px;color:#777'> avg</span></div>"
                f"<div style='font-size:12px;color:#555'>p50 ≤{_fmt_tok(pc.get(50,0))} · "
                f"p90 ≤{_fmt_tok(pc.get(90,0))} · p99 ≤{_fmt_tok(pc.get(99,0))}</div></div>")

    note = ("<div style='font-size:12px;color:#777;margin-top:6px'>Percentiles are histogram "
            "upper-edges (coarse): “≤ this many tokens”. Compare p99 to --max-model-len — a p99 "
            "far below max-len means the context window is over-provisioned.</div>")
    return (f"<div style='font-family:system-ui'>"
            f"<div style='font-size:13px;font-weight:600;margin-bottom:4px'>Request size — tokens "
            f"in / out (n={n:,} completed)</div>"
            f"<div style='display:flex;gap:10px;flex-wrap:wrap'>"
            f"{cell('INPUT (prompt)', avg_in, pin)}{cell('OUTPUT (generation)', avg_out, pout)}</div>"
            f"{note}</div>")


def render_phase_breakdown(m: dict) -> str:
    """Average per-request time split: queue → prefill → decode, from the vLLM
    request-phase histograms. This is the 'where does time go' view."""
    if "_error" in m:
        return ""
    queue = _avg(m, "vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count")
    prefill = _avg(m, "vllm:request_prefill_time_seconds_sum", "vllm:request_prefill_time_seconds_count")
    decode = _avg(m, "vllm:request_decode_time_seconds_sum", "vllm:request_decode_time_seconds_count")
    infer = _avg(m, "vllm:request_inference_time_seconds_sum", "vllm:request_inference_time_seconds_count")
    e2e = _avg(m, "vllm:e2e_request_latency_seconds_sum", "vllm:e2e_request_latency_seconds_count")
    parts = [("queue", queue, "#9e9e9e"), ("prefill", prefill, "#1a73e8"), ("decode", decode, "#e37400")]
    total = sum(v for _, v, _ in parts if v) or 1.0
    bar = "<div style='display:flex;height:26px;border-radius:6px;overflow:hidden;margin:6px 0'>"
    legend = ""
    for name, val, col in parts:
        if not val:
            continue
        pct = 100 * val / total
        bar += f"<div style='background:{col};width:{pct:.1f}%' title='{name} {val*1000:.0f}ms'></div>"
        legend += (f"<span style='margin-right:14px'><span style='display:inline-block;width:10px;"
                   f"height:10px;background:{col};border-radius:2px'></span> "
                   f"{name}: <b>{val*1000:.0f} ms</b> ({pct:.0f}%)</span>")
    bar += "</div>"
    footer = (f"<div style='font-size:12px;color:#555'>avg inference (prefill+decode): "
              f"<b>{(infer or 0)*1000:.0f} ms</b> · avg e2e latency: <b>{(e2e or 0):.2f}s</b> "
              f"(e2e includes queue+detokenize)</div>")
    return (f"<div style='font-family:system-ui'><div style='font-size:13px;font-weight:600'>"
            f"Avg time per request — where it goes</div>{bar}{legend}{footer}</div>")


# ── serve-log parsing (the GROUND TRUTH for engine args) ────────────────────
# vLLM prints the fully-resolved config to its startup log. /metrics does NOT
# expose these, so the log is the only way to CONFIRM what a running server
# actually launched with (esp. after a GPU_MEM_UTIL / MAX_NUM_SEQS sweep).

def find_serve_logs(server: str, model_id: str = "", limit: int = 12) -> list:
    """Discover candidate serve logs, RANKED by relevance to THIS server, not just
    recency. The self-logging path (added to start_vllm_server.sh) is
    serve_<model>_<node>_p<port>_<pid>.out — so we score each candidate on:
      +100 filename contains this node       +100 filename contains this port
      +60  filename contains model short-name +40 file body mentions node/model
      + recency (tie-break)
    A server launched WITHOUT the script and WITHOUT any redirection leaves no file;
    the manual path box is the ultimate fallback."""
    import glob
    node = _node_from_server(server) or ""
    m = re.search(r":(\d+)", server or "")
    port = m.group(1) if m else ""
    short = (model_id or "").split("/")[-1].lower()
    pats = [
        "/mnt/data/sgsilva/logs/serve/**/*.log",
        "/mnt/data/sgsilva/logs/serve/**/*.out",
        "/mnt/data/sgsilva/logs/serve/*.out",
    ]
    cands = set()
    for p in pats:
        cands.update(glob.glob(p, recursive=True))

    def score(f):
        s = 0.0
        base = os.path.basename(f).lower()
        if node and node in base:
            s += 100
        if port and f"p{port}" in base:
            s += 100
        if short and short in base:
            s += 60
        try:
            s += os.path.getmtime(f) / 1e12  # recency tie-break (tiny weight)
        except OSError:
            pass
        # only pay for a body grep when the filename didn't already strongly match
        if s < 100 and (node or short):
            try:
                head = open(f, "r", errors="replace").read(4000).lower()
                if node and node in head:
                    s += 40
                if short and short and short in head:
                    s += 40
            except OSError:
                pass
        return s

    ranked = sorted(cands, key=score, reverse=True)
    return ranked[:limit]


def read_log_tail(path: str, n_lines: int = 200) -> str:
    """Return the last n_lines of a (possibly remote) log. Local read only —
    logs live on the shared /mnt mount visible from every node."""
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", errors="replace") as f:
            return "".join(f.readlines()[-n_lines:])
    except Exception:  # noqa: BLE001
        return ""


def parse_serve_log(path: str) -> dict:
    """Extract the resolved config from a vLLM serve log:
      - non_default_args: the '{...}' dict from the 'non-default args:' line
      - kv_cache_tokens, max_concurrency, per_request_tokens: the KV lines
    Uses the LAST occurrence (a log may hold several restarts). {} if not found."""
    if not path or not os.path.exists(path):
        return {"_error": f"log not found: {path}"}
    try:
        txt = open(path, "r", errors="replace").read()
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}
    out = {}
    nda = re.findall(r"non-default args:\s*(\{.*\})", txt)
    if nda:
        raw = nda[-1]
        try:
            import ast as _ast
            out["non_default_args"] = _ast.literal_eval(raw)
        except Exception:  # noqa: BLE001
            out["non_default_args_raw"] = raw
    kv = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", txt)
    if kv:
        out["kv_cache_tokens"] = kv[-1]
    mc = re.findall(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", txt)
    if mc:
        out["per_request_tokens"] = mc[-1][0]
        out["max_concurrency"] = mc[-1][1]
    return out


def render_serve_log(path: str) -> tuple:
    """Return (config_html, tail_text) for the serve-log tab."""
    parsed = parse_serve_log(path)
    tail = read_log_tail(path, 200)
    if "_error" in parsed:
        return f"<div style='color:#611a15'>{parsed['_error']}</div>", tail
    nda = parsed.get("non_default_args", {})
    rows = ""
    # surface the perf-relevant keys first, then the rest
    priority = ["model", "max_model_len", "gpu_memory_utilization", "max_num_seqs",
                "max_num_batched_tokens", "enable_chunked_prefill", "enable_prefix_caching",
                "kv_cache_dtype", "quantization", "reasoning_parser", "tensor_parallel_size"]
    seen = set()
    def add(k, v):
        return (f"<tr><td style='padding:5px 10px;font-weight:600'>{k}</td>"
                f"<td style='padding:5px 10px'><code>{v}</code></td></tr>")
    for k in priority:
        if k in nda:
            rows += add(k, nda[k]); seen.add(k)
    for k, v in nda.items():
        if k not in seen:
            rows += add(k, v)
    kv_block = ""
    if "kv_cache_tokens" in parsed:
        kv_block = (
            f"<div style='margin:8px 0;padding:10px;background:#e8f0fe;border-radius:8px;font-family:system-ui'>"
            f"<b>GPU KV cache:</b> {parsed['kv_cache_tokens']} tokens · "
            f"<b>Max concurrency:</b> {parsed.get('max_concurrency','?')}x "
            f"(assuming {parsed.get('per_request_tokens','?')} tokens/request = --max-model-len)"
            f"<div style='font-size:12px;color:#555;margin-top:4px'>This concurrency figure scales "
            f"INVERSELY with --max-model-len: halve max-len → ~2× this number. Compare to your real "
            f"p99 request size (Monitor tab) to judge if max-len is over-provisioned.</div></div>")
    if not rows and "non_default_args_raw" in parsed:
        rows = add("(unparsed)", parsed["non_default_args_raw"])
    table = (f"<div style='font-size:13px;font-weight:600;font-family:system-ui'>Resolved engine "
             f"config (from log — GROUND TRUTH)</div>"
             f"<table style='border-collapse:collapse;font-family:system-ui;font-size:13px'>{rows}</table>"
             if rows else "<div style='color:#777'>No 'non-default args' line found in this log yet "
             "(server may still be starting).</div>")
    return kv_block + table, tail


# ── SCRIPT-declared settings (parse start_vllm_server.sh's Qwen3.5 branch) ───

def script_declared_settings(model_id: str, thinking_hint: bool) -> dict:
    """The prefill/decode/serving args start_vllm_server.sh WOULD pass for this
    model family. Read from the script text so it tracks edits to the script,
    rather than hardcoding a snapshot here."""
    try:
        txt = Path(SERVE_SCRIPT).read_text()
    except Exception:  # noqa: BLE001
        txt = ""
    name = (model_id or "").lower()
    is_qwen35 = "qwen3.5" in name or "qwen35" in name
    branch = "Qwen3.5" if is_qwen35 else ("GLM-4.7" if "glm-4.7" in name else
             ("Kimi" if "kimi" in name else "standard"))

    def has(flag):
        return flag in txt

    # These are the fixed defaults in the Qwen3.5 / standard branches.
    gpu_util = "0.85" if is_qwen35 else "0.90"
    return {
        "branch": branch,
        "chunked_prefill": "ON (--enable-chunked-prefill)" if has("--enable-chunked-prefill") else "off",
        "max_num_batched_tokens": "8192 (scheduler default w/ chunked prefill)",
        "prefix_caching": "OFF (--no-enable-prefix-caching)" if has("--no-enable-prefix-caching") else "on",
        "gpu_memory_utilization": gpu_util,
        "mm_encoder_tp_mode": "data" if has("--mm-encoder-tp-mode data") else "n/a",
        "mm_processor_cache_type": "shm" if has("--mm-processor-cache-type shm") else "n/a",
        "video_num_frames": "2048" if "num_frames" in txt else "n/a",
        "reasoning_parser": "qwen3 (thinking ON)" if thinking_hint else "none (thinking OFF → enable_thinking:false)",
        "speculative_decoding": "none (not configured in script)",
        "cuda_graph_mode": "FULL_AND_PIECEWISE (V1 engine default, mixed prefill-decode)",
        "quantization": "none (no --quantization flag)",
    }


# ── rendering ────────────────────────────────────────────────────────────────

def _fmt_int(x):
    return f"{int(x):,}" if x is not None else "—"


def render_monitor(server: str, model_id: str, hist: dict):
    """Build the live monitor HTML + throughput figure + phase breakdown + token
    stats + GPU table. Mutates hist in place (rolling window) and returns
    (html, tokens_html, phase_html, figure, gpu_html, hist)."""
    m = fetch_metrics(server)
    now = time.time()
    if "_error" in m:
        html = (f"<div style='padding:12px;background:#fdecea;color:#611a15;"
                f"border-radius:8px'><b>⚠ metrics unavailable</b> for {server}<br>"
                f"{m['_error']}</div>")
        return html, "", "", go.Figure(), render_gpus(fetch_gpus(server)), hist

    running = m.get("vllm:num_requests_running", 0.0)
    waiting = m.get("vllm:num_requests_waiting", 0.0)
    preempt = m.get("vllm:num_preemptions_total", 0.0)
    kv = m.get("vllm:gpu_cache_usage_perc")
    if kv is None:
        kv = m.get("vllm:kv_cache_usage_perc")
    prompt_tot = m.get("vllm:prompt_tokens_total", 0.0)
    gen_tot = m.get("vllm:generation_tokens_total", 0.0)
    ttft = _avg(m, "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count")
    tpot = _avg(m, "vllm:request_time_per_output_token_seconds_sum",
                "vllm:request_time_per_output_token_seconds_count")

    # rolling deltas for CURRENT tok/s (lifetime totals hide the present)
    prev = hist.get("_last")
    p_rate = g_rate = None
    if prev and now > prev["t"]:
        dt = now - prev["t"]
        p_rate = max(0.0, (prompt_tot - prev["prompt"]) / dt)
        g_rate = max(0.0, (gen_tot - prev["gen"]) / dt)
    hist["_last"] = {"t": now, "prompt": prompt_tot, "gen": gen_tot}

    for key, val in (("t", now), ("running", running), ("gen_rate", g_rate or 0.0),
                     ("prompt_rate", p_rate or 0.0)):
        hist.setdefault(key, []).append(val)
        hist[key] = hist[key][-HISTORY_LEN:]

    decode_tps = (1.0 / tpot) if tpot else None
    load_color = "#1b5e20" if waiting == 0 else "#8a6d00"
    # first sample has no prior delta → rates are None; show 0 rather than crash
    g_rate_d = g_rate if g_rate is not None else 0.0
    p_rate_d = p_rate if p_rate is not None else 0.0

    html = f"""
    <div style='font-family:system-ui'>
      <div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px'>
        <div style='flex:1;min-width:150px;padding:12px;background:#e8f0fe;border-radius:8px'>
          <div style='font-size:12px;color:#555'>REQUESTS RUNNING</div>
          <div style='font-size:28px;font-weight:700;color:{load_color}'>{int(running)}</div>
          <div style='font-size:12px;color:#555'>waiting {int(waiting)} · preempted {int(preempt)}</div>
        </div>
        <div style='flex:1;min-width:150px;padding:12px;background:#e6f4ea;border-radius:8px'>
          <div style='font-size:12px;color:#555'>GENERATION tok/s (now)</div>
          <div style='font-size:28px;font-weight:700'>{g_rate_d:.0f}</div>
          <div style='font-size:12px;color:#555'>prompt {p_rate_d:.0f} tok/s (now)</div>
        </div>
        <div style='flex:1;min-width:150px;padding:12px;background:#fef7e0;border-radius:8px'>
          <div style='font-size:12px;color:#555'>KV-CACHE USAGE</div>
          <div style='font-size:28px;font-weight:700'>{(kv*100 if kv is not None else 0):.1f}%</div>
          <div style='font-size:12px;color:#555'>gpu prefix cache</div>
        </div>
      </div>
      <div style='display:flex;gap:10px;flex-wrap:wrap'>
        <div style='flex:1;min-width:150px;padding:12px;background:#f1f3f4;border-radius:8px'>
          <div style='font-size:12px;color:#555'>AVG TTFT (prefill)</div>
          <div style='font-size:22px;font-weight:700'>{ttft:.3f}s</div>
        </div>
        <div style='flex:1;min-width:150px;padding:12px;background:#f1f3f4;border-radius:8px'>
          <div style='font-size:12px;color:#555'>AVG DECODE (per-request)</div>
          <div style='font-size:22px;font-weight:700'>{(decode_tps if decode_tps else 0):.1f} tok/s</div>
          <div style='font-size:12px;color:#555'>{(tpot*1000 if tpot else 0):.1f} ms/tok</div>
        </div>
        <div style='flex:1;min-width:150px;padding:12px;background:#f1f3f4;border-radius:8px'>
          <div style='font-size:12px;color:#555'>LIFETIME TOKENS</div>
          <div style='font-size:16px;font-weight:700'>{_fmt_int(prompt_tot)} in</div>
          <div style='font-size:16px;font-weight:700'>{_fmt_int(gen_tot)} out</div>
        </div>
      </div>
      <div style='margin-top:8px;font-size:12px;color:#777'>
        server <b>{server}</b> · model <code>{model_id or '?'}</code> · updated {time.strftime('%H:%M:%S')}
      </div>
    </div>
    """

    fig = go.Figure()
    ts = hist.get("t", [])
    if ts:
        x = [i * REFRESH_SECS for i in range(-len(ts) + 1, 1)]  # seconds ago → 0
        fig.add_trace(go.Scatter(x=x, y=hist.get("gen_rate", []), name="gen tok/s",
                                 mode="lines", line=dict(color="#1a73e8")))
        fig.add_trace(go.Scatter(x=x, y=hist.get("running", []), name="running reqs",
                                 mode="lines", yaxis="y2", line=dict(color="#e37400")))
    fig.update_layout(
        height=280, margin=dict(l=40, r=40, t=30, b=30),
        xaxis_title="seconds ago", yaxis_title="gen tok/s",
        yaxis2=dict(title="running reqs", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15), template="plotly_white",
    )
    return (html, render_token_stats(m), render_phase_breakdown(m), fig,
            render_gpus(fetch_gpus(server)), hist)


def render_settings(server: str, model_id: str):
    thinking_hint = "thinking" not in (model_id or "").lower()  # weak default; refined below
    m = fetch_metrics(server)
    served = model_id or fetch_served_model(server)
    disk = fetch_disk_config(server, served)
    # Detect thinking from a chat probe would cost a call; infer from served metrics
    # presence of a reasoning parser isn't in /metrics, so we report the SCRIPT view
    # for reasoning and label it as such.
    decl = script_declared_settings(served, thinking_hint=True)  # script serves 397B thinking-ON

    def row(label, live, script):
        return (f"<tr><td style='padding:6px 10px;font-weight:600'>{label}</td>"
                f"<td style='padding:6px 10px'>{live}</td>"
                f"<td style='padding:6px 10px;color:#555'>{script}</td></tr>")

    live_dtype = disk.get("dtype", "unavailable (ssh)")
    live_quant = disk.get("quant", "unavailable")
    live_mamba = disk.get("mamba_ssm_dtype", "—")
    running = m.get("vllm:num_requests_running")
    live_running = "—" if "_error" in m or running is None else str(int(running))

    rows = "".join([
        row("served model id", f"<code>{served or '?'}</code>", f"branch: {decl['branch']}"),
        row("dtype (weights)", live_dtype, "auto → inherits config (no --dtype)"),
        row("mamba_ssm_dtype", live_mamba, "float32 (SSM state, model default)"),
        row("quantization", live_quant, decl["quantization"]),
        row("chunked prefill", "reported via engine (see log)", decl["chunked_prefill"]),
        row("max_num_batched_tokens", "—", decl["max_num_batched_tokens"]),
        row("prefix caching", "—", decl["prefix_caching"]),
        row("gpu_memory_utilization", "—", decl["gpu_memory_utilization"]),
        row("mm_encoder_tp_mode", "—", decl["mm_encoder_tp_mode"]),
        row("video num_frames", "—", decl["video_num_frames"]),
        row("reasoning parser", "—", decl["reasoning_parser"]),
        row("speculative decoding", "—", decl["speculative_decoding"]),
        row("cuda graph mode", "—", decl["cuda_graph_mode"]),
        row("requests running (live)", live_running, "—"),
    ])
    err = f"<div style='color:#611a15'>metrics: {m['_error']}</div>" if "_error" in m else ""
    return f"""
    {err}
    <table style='border-collapse:collapse;width:100%;font-family:system-ui;font-size:13px'>
      <tr style='background:#e8eaed'>
        <th style='padding:6px 10px;text-align:left'>Setting</th>
        <th style='padding:6px 10px;text-align:left'>LIVE (reported / on-disk)</th>
        <th style='padding:6px 10px;text-align:left'>SCRIPT (start_vllm_server.sh)</th>
      </tr>
      {rows}
    </table>
    <div style='font-size:12px;color:#777;margin-top:8px'>
      LIVE = read from the running server (/v1/models, /metrics) and the model's on-disk
      config.json (via SSH). SCRIPT = what start_vllm_server.sh would pass for this family.
      A “—” means the value isn't exposed by an HTTP endpoint — read it from the serve log,
      or trust the SCRIPT column if the server was launched via the script.
    </div>
    """


# ── chat: ask the served model ───────────────────────────────────────────────

def ask_model(server: str, model_id: str, question: str, history):
    """Append a user+assistant turn using Gradio's messages format (list of
    {'role','content'} dicts — required by gradio>=5 Chatbot(type='messages'))."""
    history = list(history or [])
    if not question.strip():
        return history, ""
    history.append({"role": "user", "content": question})
    served = model_id or fetch_served_model(server)
    if not served:
        history.append({"role": "assistant", "content": "⚠ no served model / server unreachable"})
        return history, ""
    try:
        client = OpenAI(base_url=f"{server}/v1", api_key="EMPTY")
        resp = client.chat.completions.create(
            model=served,
            messages=[{"role": "user", "content": question}],
            max_tokens=2048, temperature=0.3,  # 2048: thinking models need room (eval gotcha)
        )
        ans = resp.choices[0].message.content or ""
        think = getattr(resp.choices[0].message, "reasoning_content", None)
        if think:
            ans = f"<think>\n{think}\n</think>\n\n{ans}"
        if not ans:
            ans = "(empty response — try again; thinking models need adequate max_tokens)"
        history.append({"role": "assistant", "content": ans})
    except Exception as e:  # noqa: BLE001
        history.append({"role": "assistant", "content": f"⚠ {type(e).__name__}: {e}"})
    return history, ""


# ── UI ───────────────────────────────────────────────────────────────────────

def build():
    with gr.Blocks(theme=gr.themes.Soft(), title="serve-monitor") as demo:
        gr.Markdown("## 🔭 vLLM Serve Monitor — live load, throughput, and settings")

        server_state = gr.State(DEFAULT_SERVER)
        model_state = gr.State("")
        scan_state = gr.State([])
        hist_state = gr.State({})

        with gr.Row():
            server_box = gr.Textbox(DEFAULT_SERVER, label="Server URL", scale=3)
            scan_btn = gr.Button("🔍 Scan cluster", scale=1)
            model_box = gr.Textbox("", label="Served model id (auto-filled)", scale=4)
        with gr.Row():
            show_all = gr.Checkbox(False, label=f"Show ALL servers (default: only {MY_USER}'s)")
        scan_dd = gr.Dropdown([], label="Pick a discovered server", visible=False)
        scan_summary = gr.Markdown("")

        with gr.Tab("Monitor"):
            auto = gr.Checkbox(True, label=f"Auto-refresh every {REFRESH_SECS}s")
            refresh_btn = gr.Button("↻ Refresh now", variant="primary")
            mon_html = gr.HTML()
            with gr.Group():
                gr.Markdown("#### 📏 Request size — tokens in / out (avg + p50/p90/p99)")
                tokens_html = gr.HTML()
            with gr.Group():
                gr.Markdown("#### ⏱ Time per request — prefill vs decode vs queue")
                phase_html = gr.HTML()
            mon_plot = gr.Plot()
            with gr.Accordion("🖥 GPUs on this node (nvidia-smi)", open=True):
                gpu_html = gr.HTML()
            timer = gr.Timer(REFRESH_SECS, active=True)

        with gr.Tab("Settings & Ask"):
            set_btn = gr.Button("↻ Read settings (live + script)", variant="primary")
            set_html = gr.HTML()
            gr.Markdown("### Ask the served model")
            chat = gr.Chatbot(height=320)  # gradio 6: messages-format (dict) is the only format
            q = gr.Textbox("", label="Question", placeholder="e.g. what is your maximum context length?")
            ask_btn = gr.Button("Ask", variant="primary")

        with gr.Tab("Serve log"):
            gr.Markdown(
                "Paste the server's log path (or auto-find). The `non-default args` line is the "
                "**ground truth** for engine args (`gpu_memory_utilization`, `max_num_seqs`, "
                "`max_model_len`, …) that `/metrics` does NOT expose — this is how you CONFIRM a "
                "sweep took effect. Logs launched via `log_run.sh` live under "
                "`/mnt/data/sgsilva/logs/serve/`; a server started without it (e.g. the 397B) needs "
                "its path pasted manually.")
            with gr.Row():
                log_path = gr.Textbox("", label="Serve log path", scale=4)
                find_btn = gr.Button("🔎 Auto-find", scale=1)
                log_read_btn = gr.Button("↻ Parse + tail", variant="primary", scale=1)
            log_found = gr.Dropdown([], label="Discovered logs (newest first)", visible=False)
            log_config = gr.HTML()
            log_tail = gr.Code(label="Log tail (last 200 lines)", language=None, lines=20)

        # ── wiring ──
        def _sync_server(url):
            served = fetch_served_model(url)
            return url, served, served

        server_box.submit(_sync_server, server_box, [server_state, model_state, model_box])

        def _scan(show_all_servers):
            _summary, results, _choices = scan_cluster()
            # scan_cluster returns (node, port, [model_ids], owner) tuples.
            # Filter to MY servers unless "show all" is ticked. Owner attribution
            # can be "unknown" (an unprivileged session can't read another user's
            # socket PID — documented in live_inference.get_vllm_owner), so an
            # unknown-owner server is NOT mine and stays hidden by default.
            sel = results if show_all_servers else [r for r in results if r[3] == MY_USER]
            lines, choices = [], []
            for node, port, models, owner in sel:
                port_tag = "" if port == 8000 else f":{port}"
                # re-fetch the model id if the scan came back empty (happens when
                # /v1/models returned an empty data list at scan time)
                mids = [m for m in models if m] or [fetch_served_model(f"http://{node}:{port}")]
                for mid in mids:
                    short = (mid or "?").split("/")[-1]
                    choices.append(f"{node}{port_tag} | {owner} | {short}")
                    lines.append(f"✓ {node}:{port}  [{owner}]  {short}")
            scope = "ALL users" if show_all_servers else MY_USER
            summary = (f"Found {len(sel)} server(s) ({scope}):\n" + "\n".join(lines)
                       if sel else f"No servers found for scope: {scope}.")
            return summary, sel, gr.update(choices=choices, visible=bool(choices))
        scan_btn.click(_scan, show_all, [scan_summary, scan_state, scan_dd])

        def _pick(sel, results):
            url, mid = apply_scan_selection(sel, results)
            return url, mid, url, mid
        scan_dd.change(_pick, [scan_dd, scan_state], [server_state, model_state, server_box, model_box])

        def _mon(server, model, hist):
            html, tokens, phase, fig, gpus, hist = render_monitor(server, model, hist)
            return html, tokens, phase, fig, gpus, hist
        mon_outputs = [mon_html, tokens_html, phase_html, mon_plot, gpu_html, hist_state]
        refresh_btn.click(_mon, [server_state, model_state, hist_state], mon_outputs)

        def _tick(s, m, h, on):
            if not on:
                return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), h)
            return _mon(s, m, h)
        timer.tick(_tick, [server_state, model_state, hist_state, auto], mon_outputs)
        auto.change(lambda on: gr.update(active=on), auto, timer)

        set_btn.click(render_settings, [server_state, model_state], set_html)
        ask_btn.click(ask_model, [server_state, model_state, q, chat], [chat, q])
        q.submit(ask_model, [server_state, model_state, q, chat], [chat, q])

        def _find_logs(server, model):
            logs = find_serve_logs(server, model)
            return gr.update(choices=logs, visible=bool(logs), value=logs[0] if logs else None)
        find_btn.click(_find_logs, [server_state, model_state], log_found)
        log_found.change(lambda p: p, log_found, log_path)

        def _read_log(path):
            cfg, tail = render_serve_log(path)
            return cfg, tail
        log_read_btn.click(_read_log, log_path, [log_config, log_tail])

        demo.load(_sync_server, server_box, [server_state, model_state, model_box])
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("MONITOR_PORT", "7878"))
    build().launch(server_name="0.0.0.0", server_port=port, share=False)
