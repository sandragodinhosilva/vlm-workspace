#!/usr/bin/env python3
"""Gradio dashboard for monitoring VLM **SFT** training runs.

Gives the visibility that's missing while an SFT run is in flight: loss/grad-norm/
lr curves, val-loss at save points, step timing & throughput, GPU/host health, and
an config-intelligence panel that flags SWEPT-vs-FROZEN knob drift against the
canonical safe-default block.

Data source: each run writes tensorboard event files to
    <logs-dir>/<run>/exp_NNN/tensorboard/events.out.tfevents.*
(`logs-dir` defaults to /home/sgsilva/nemo-rl-vlm/logs — where training writes).
Resumed runs get a higher exp_NNN; on step-number conflicts the higher exp wins.

Config panel: prefers a `config_snapshot.yaml` written into the run's log dir at
launch (exact config-as-run); falls back to auto-matching the run name to
examples/configs/sft_vlm_<stem>_megatron.yaml.

Reading tfevents needs the `tensorboard` package — only the in-repo nemo .venv has
it, so launch with that interpreter:
    /home/sgsilva/nemo-rl-vlm/.venv/bin/python sft_dashboard.py [--port 7875]

Sibling of grpo_dashboard.py (port 7873). Lives in ~/utilities/apps.
"""

import argparse
import glob
import os
import re
from functools import lru_cache

import gradio as gr
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LOGS_DIR = "/home/sgsilva/nemo-rl-vlm/logs"
# Also scanned if it exists (mirrors the grpo_logs convention).
EXTRA_LOGS_DIRS = ["/mnt/data/sgsilva/logs/sft_logs"]
CONFIGS_DIR = "/home/sgsilva/nemo-rl-vlm/examples/configs"
# the launcher writes node-0 training logs here: <SLURM_LOGS>/<YYYYMMDD>/training_<stem>_node_0_*.log
SLURM_LOGS_DIR = "/home/sgsilva/nemo-rl-vlm/slurm_logs"

COLORS = [
    "#1976d2", "#d32f2f", "#388e3c", "#f57c00", "#7b1fa2",
    "#00838f", "#c2185b", "#455a64", "#6d4c41", "#afb42b",
]

# Core per-step train scalars to expose as curves.
TRAIN_CURVES = [
    ("train/loss", "Training Loss", "loss"),
    ("train/grad_norm", "Gradient Norm", "grad_norm"),
    ("train/lr", "Learning Rate", "lr"),
    ("train/num_unmasked_tokens", "Unmasked Tokens / step", "tokens"),
]
TIMING_CURVES = [
    ("timing/train/total_step_time", "Total Step Time (s)", "seconds"),
    ("timing/train/policy_training", "Policy Training Time (s)", "seconds"),
    ("timing/train/data_processing", "Data Processing Time (s)", "seconds"),
    ("timing/train/valid_tokens_per_sec_per_gpu", "Valid Tokens/sec/GPU", "tok/s/gpu"),
]
VAL_TAG = "validation/val_loss"

# --- Config intelligence (from sft_grpo_hyperparameters_LIVE.md) -------------
# FROZEN knobs: asserted constant. Any deviation is flagged.
FROZEN_KNOBS = {
    "policy.megatron_cfg.optimizer.lr": 5e-6,
    "policy.megatron_cfg.optimizer.min_lr": 5e-7,
    "policy.megatron_cfg.optimizer.weight_decay": 0.01,
    "policy.megatron_cfg.optimizer.adam_beta2": 0.98,
    "policy.megatron_cfg.scheduler.lr_decay_style": "cosine",
}
# SWEPT knobs: vary per run; shown but not flagged.
SWEPT_KNOBS = [
    "sft.max_num_epochs",
    "policy.train_global_batch_size",
    "policy.train_micro_batch_size",
    "policy.max_total_sequence_length",
    "policy.megatron_cfg.tensor_model_parallel_size",
    "policy.megatron_cfg.pipeline_model_parallel_size",
    "policy.megatron_cfg.context_parallel_size",
    "policy.megatron_cfg.attention_backend",
    "policy.megatron_cfg.activation_checkpointing",
    "policy.megatron_cfg.sequence_parallel",
]
# Identity knobs (shown for context).
IDENTITY_KNOBS = [
    "policy.model_name",
    "checkpointing.checkpoint_dir",
    "cluster.num_nodes",
    "cluster.gpus_per_node",
    "sft.save_period",
]
# Safe-default fit-in-memory guardrails (omitting any reintroduces a known crash).
GUARDRAILS = [
    ("policy.megatron_cfg.attention_backend", "flash",
     "unfused-attention silent fp32 OOM — set 'flash'"),
    ("policy.megatron_cfg.sequence_parallel", True,
     "needed with TP≥2 for long-video; whole seq on one GPU otherwise"),
    ("policy.megatron_cfg.activation_checkpointing", True,
     "memory headroom; off risks OOM on 27B"),
]
VIDEO_SEQLEN_MIN = 62000  # video configs need ≥62k or seqlen crash mid-run


# ---------------------------------------------------------------------------
# Tensorboard reading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _event_accumulator_cls():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    return EventAccumulator


def _read_event_file(path):
    """Return {tag: (steps_np, values_np)} for all scalars in one event file."""
    EA = _event_accumulator_cls()
    ea = EA(path, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        sc = ea.Scalars(tag)
        steps = np.array([s.step for s in sc], dtype=np.int64)
        vals = np.array([s.value for s in sc], dtype=np.float64)
        out[tag] = (steps, vals)
    return out


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def discover_runs(logs_dirs):
    """Return [(run_name, run_dir, [event_files_sorted_by_exp])] for SFT runs.

    A run with tfevents is a real training run (events non-empty). A run dir that has
    a `config_snapshot.yaml` (written at LAUNCH, before training) but NO tfevents yet is
    a STARTING run (queued / spinning up) — surfaced with an EMPTY event list so it shows
    in the selector (config tab works; curves are empty until step-1 tfevents appear).
    """
    runs = {}
    starting = {}
    for logs_dir in logs_dirs:
        if not os.path.isdir(logs_dir):
            continue
        for name in sorted(os.listdir(logs_dir)):
            run_dir = os.path.join(logs_dir, name)
            if not os.path.isdir(run_dir):
                continue
            # exp_NNN/tensorboard/events.out.tfevents.*  (sorted: later exp wins)
            evs = sorted(glob.glob(
                os.path.join(run_dir, "exp_*", "tensorboard", "events.out.tfevents.*")))
            if not evs:
                # also accept a flat tensorboard dir
                evs = sorted(glob.glob(
                    os.path.join(run_dir, "tensorboard", "events.out.tfevents.*")))
            if evs:
                runs.setdefault(name, (run_dir, evs))
            elif glob.glob(os.path.join(run_dir, "config_snapshot*.yaml")):
                # launched but not training yet — keep its snapshot mtime to sort/filter by
                starting.setdefault(name, (run_dir, []))
    # real runs first (newest by latest event mtime), then starting runs (newest snapshot first)
    items = sorted(runs.items(), key=lambda kv: _mtime(kv[1][1][-1]), reverse=True)
    out = [(name, rd, evs) for name, (rd, evs) in items]
    def _snap_mtime(rd):
        snaps = glob.glob(os.path.join(rd, "config_snapshot*.yaml"))
        return max((_mtime(s) for s in snaps), default=0.0)
    start_items = sorted(
        ((n, rd) for n, (rd, _) in starting.items() if n not in runs),
        key=lambda nr: _snap_mtime(nr[1]), reverse=True)
    out += [(name, rd, []) for name, rd in start_items]
    return out


def load_run_scalars(event_files):
    """Stitch a tag's curve across exp_* dirs by STEP, later-exp wins on collision.

    A resume writes a fresh exp_NNN. Two cases occur in the wild:
      (a) the resumed exp CONTINUES the global step counter (e.g. resume from
          step_220 → exp logs steps 221..N). Here the earlier exp holds 1..220
          and the new one holds 221..N with NO overlap — they must be
          CONCATENATED, or the chart starts at 220 and you lose the full run.
      (b) the resumed exp RESTARTS at step 1. Here both exps cover 1..N and a
          naive concat would double each step; the later exp must WIN per step.

    A per-step "later wins, then take the union" merge handles BOTH: overlapping
    steps take the newest exp's value (case b — the abandoned earlier segment is
    overwritten), and non-overlapping steps from the earlier exp are kept (case a
    — the pre-resume history is preserved). The old "latest exp whole" rule was
    correct only for (b) and silently truncated (a) — the vobs2906 resume bug.

    Returns {tag: (steps_np_sorted_unique, values_np)}.
    """
    # tag -> dict{step: value}; iterate exps ascending so a later exp overwrites
    # an earlier exp's value at the SAME step, while unique earlier steps survive.
    merged = {}
    for ev in event_files:  # ascending exp order
        try:
            tag_data = _read_event_file(ev)
        except Exception:
            continue
        for tag, (steps, vals) in tag_data.items():
            d = merged.setdefault(tag, {})
            for st, vl in zip(steps.tolist(), vals.tolist()):
                d[st] = vl  # later exp wins on step collision; new steps added
    out = {}
    for tag, d in merged.items():
        ks = np.array(sorted(d.keys()))
        out[tag] = (ks, np.array([d[k] for k in ks]))
    return out


# Cache keyed on (event paths, latest mtime) so live runs refresh but idle ones don't re-parse.
_SCALAR_CACHE = {}


def get_run_scalars(event_files):
    if not event_files:          # starting run (no tfevents yet)
        return {}
    key = (tuple(event_files), round(_mtime(event_files[-1]), 1))
    if key not in _SCALAR_CACHE:
        _SCALAR_CACHE[key] = load_run_scalars(event_files)
    return _SCALAR_CACHE[key]


# ---------------------------------------------------------------------------
# GPU / host health aggregation (ray/* tags)
# ---------------------------------------------------------------------------
GPU_WINDOW = 5  # samples to window util over, so a single transient/shutdown 0 doesn't dominate


def _windowed_util(vals):
    """Max util over the last GPU_WINDOW samples.

    ray/* telemetry keeps ticking through shutdown, so a finished run's final
    util sample reads 0; a single live sample can also momentarily dip to 0 on a
    healthy rank. Taking the windowed MAX answers 'was this GPU recently busy?'
    rather than 'is this exact instant 0?', which is the question that matters.
    """
    if len(vals) == 0:
        return float("nan")
    w = vals[-GPU_WINDOW:]
    w = w[~np.isnan(w)]
    return float(np.max(w)) if len(w) else float("nan")


def gpu_summary(scalars):
    """Aggregate recent ray/* GPU+host stats into a per-node table dict.

    GPU util is windowed (max over last GPU_WINDOW samples); GPU mem and host
    mem use the last value. Returns {"nodes": {ni: {"util": {gi: v}, "mem": {gi: v}}},
    "host": {ni: {mem_gb, mem_total_gb}}}.
    """
    nodes = {}
    host = {}
    for tag, (steps, vals) in scalars.items():
        if not tag.startswith("ray/node."):
            continue
        m = re.match(r"ray/node\.(\d+)\.gpu\.(\d+)\.(util|mem_gb)$", tag)
        if m:
            ni, gi, kind = int(m.group(1)), int(m.group(2)), m.group(3)
            nd = nodes.setdefault(ni, {"util": {}, "mem": {}})
            if kind == "util":
                nd["util"][gi] = _windowed_util(vals)
            else:
                nd["mem"][gi] = float(vals[-1]) if len(vals) else float("nan")
            continue
        m2 = re.match(r"ray/node\.(\d+)\.(mem_gb|mem_total_gb)$", tag)
        if m2:
            ni, kind = int(m2.group(1)), m2.group(2)
            host.setdefault(ni, {})[kind] = float(vals[-1]) if len(vals) else float("nan")
    return {"nodes": nodes, "host": host}


def gpu_health_markdown(scalars, is_running=True):
    """Render the per-node GPU/host table.

    ⚠️ low-util warnings are only meaningful for an ACTIVE run — a finished run's
    telemetry trails off to 0, so the warning is suppressed when is_running=False.
    """
    gs = gpu_summary(scalars)
    if not gs["nodes"]:
        return "_No ray/* GPU telemetry in this run's logs._"
    header = ("latest sample, util windowed over last "
              f"{GPU_WINDOW}" + (" — run IDLE/finished, util warnings suppressed"
                                 if not is_running else ""))
    lines = [f"### GPU / host health ({header})", ""]
    lines.append("| node | mean GPU util % | min util | GPU mem (max GB) | host mem GB |")
    lines.append("|---|---|---|---|---|")
    for ni in sorted(gs["nodes"]):
        nd = gs["nodes"][ni]
        utils = [v for v in nd["util"].values() if not np.isnan(v)]
        mems = [v for v in nd["mem"].values() if not np.isnan(v)]
        hm = gs["host"].get(ni, {})
        host_mem = hm.get("mem_gb", float("nan"))
        host_tot = hm.get("mem_total_gb", float("nan"))
        mean_u = np.mean(utils) if utils else float("nan")
        min_u = np.min(utils) if utils else float("nan")
        max_m = np.max(mems) if mems else float("nan")
        host_str = (f"{host_mem:.0f} / {host_tot:.0f}"
                    if not np.isnan(host_mem) else "—")
        warn = " ⚠️" if (is_running and utils and min_u < 50) else ""
        lines.append(f"| {ni} | {mean_u:.0f}{warn} | {min_u:.0f} | {max_m:.0f} | {host_str} |")
    lines.append("")
    if is_running:
        lines.append(f"_⚠️ on a node = a GPU stayed under 50% util across the last "
                     f"{GPU_WINDOW} samples (straggler / imbalance / idle rank)._")
    else:
        lines.append("_Run is idle/finished — GPU util naturally trails to 0 at shutdown; "
                     "util warnings are suppressed. GPU/host mem shown for reference._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config intelligence
# ---------------------------------------------------------------------------
def _deep_get(d, dotted):
    cur = d
    for k in dotted.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def _resolve_config(cfg_path):
    """Load a config, merging its `defaults` base (shallow per top-key) underneath."""
    if yaml is None:
        return None
    try:
        d = yaml.safe_load(open(cfg_path))
    except Exception:
        return None
    base_name = d.get("defaults")
    if isinstance(base_name, str):
        bp = base_name if base_name.endswith((".yaml", ".yml")) else base_name + ".yaml"
        bp = os.path.join(os.path.dirname(cfg_path), bp)
        if os.path.exists(bp):
            try:
                base = yaml.safe_load(open(bp)) or {}
            except Exception:
                base = {}
            # merge base under d (d overrides), one level deep on dict top-keys
            for k, bv in base.items():
                if k not in d:
                    d[k] = bv
                elif isinstance(d[k], dict) and isinstance(bv, dict):
                    d[k] = {**bv, **d[k]}
    return d


def find_config(run_name, run_dir):
    """Locate the config for a run. Returns (config_dict, source_label) or (None, msg).

    Priority: config_snapshot.yaml in the run dir (exact-as-run) →
    examples/configs/sft_vlm_<stem>_megatron.yaml matched by run-name stem.
    """
    # 1) snapshot written at launch
    snap = os.path.join(run_dir, "config_snapshot.yaml")
    if os.path.exists(snap):
        cfg = _resolve_config(snap)
        if cfg is not None:
            return cfg, "config_snapshot.yaml (exact-as-run)"
    # 2) match run name -> config file
    stem = run_name
    if stem.startswith("sft_"):
        stem = stem[len("sft_"):]
    # candidate filenames, most specific first
    candidates = [
        f"sft_vlm_{stem}_megatron.yaml",
        f"sft_vlm_{stem}.yaml",
        f"{stem}_megatron.yaml",
        f"{stem}.yaml",
    ]
    for c in candidates:
        p = os.path.join(CONFIGS_DIR, c)
        if os.path.exists(p):
            cfg = _resolve_config(p)
            if cfg is not None:
                return cfg, f"matched {c}"
    return None, f"no config found for '{run_name}' (no snapshot, no name match)"


def _fmt_val(v):
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.2e}"
        return f"{v:g}"
    return str(v)


def config_markdown(run_name, run_dir):
    cfg, src = find_config(run_name, run_dir)
    if cfg is None:
        return f"### Config\n\n_{src}_"
    lines = [f"### Config — {run_name}", f"<sub>source: {src}</sub>", ""]

    # Identity
    lines.append("#### Identity")
    lines.append("| knob | value |\n|---|---|")
    for k in IDENTITY_KNOBS:
        v = _deep_get(cfg, k)
        lines.append(f"| `{k.split('.')[-1]}` | {_fmt_val(v) if v is not None else '—'} |")

    # Swept
    lines.append("\n#### Swept knobs (this run's choices)")
    lines.append("| knob | value |\n|---|---|")
    for k in SWEPT_KNOBS:
        v = _deep_get(cfg, k)
        lines.append(f"| `{k.split('.')[-1]}` | {_fmt_val(v) if v is not None else '—'} |")

    # Frozen + drift check
    lines.append("\n#### Frozen knobs (drift-checked)")
    lines.append("| knob | value | canonical | status |\n|---|---|---|---|")
    for k, canon in FROZEN_KNOBS.items():
        v = _deep_get(cfg, k)
        if v is None:
            status = "—"
        elif isinstance(canon, float):
            # YAML may load scientific notation (5e-6) as a string — coerce.
            try:
                status = "✅" if abs(float(v) - canon) < 1e-12 else "⚠️ DRIFT"
            except (TypeError, ValueError):
                status = "⚠️ DRIFT"
        else:
            status = "✅" if str(v) == str(canon) else "⚠️ DRIFT"
        lines.append(f"| `{k.split('.')[-1]}` | {_fmt_val(v) if v is not None else '—'} "
                     f"| {_fmt_val(canon)} | {status} |")

    # Guardrail audit
    lines.append("\n#### Fit-in-memory guardrails")
    warnings = []
    for k, want, why in GUARDRAILS:
        v = _deep_get(cfg, k)
        ok = (str(v) == str(want))
        if not ok:
            warnings.append(f"- ⚠️ `{k.split('.')[-1]}` = `{_fmt_val(v)}` (want `{_fmt_val(want)}`) — {why}")
    seqlen = _deep_get(cfg, "policy.max_total_sequence_length")
    if isinstance(seqlen, (int, float)) and seqlen < VIDEO_SEQLEN_MIN:
        warnings.append(f"- ⚠️ `max_total_sequence_length` = `{int(seqlen)}` < {VIDEO_SEQLEN_MIN} — "
                        f"OK for non-video; a video config will crash mid-run.")
    if warnings:
        lines.extend(warnings)
    else:
        lines.append("- ✅ all guardrails satisfied")
    return "\n".join(lines)


def raw_config_text(run_name, run_dir):
    """The exact YAML the run launched with — the config_snapshot if present (best:
    captures the resolved-as-run config), else the name-matched config file. Returns
    (yaml_text, source_label)."""
    snaps = sorted(glob.glob(os.path.join(run_dir, "config_snapshot*.yaml")))
    # prefer the plain (latest-launch) snapshot; fall back to the newest timestamped one
    plain = os.path.join(run_dir, "config_snapshot.yaml")
    path = plain if os.path.exists(plain) else (snaps[-1] if snaps else None)
    if path:
        try:
            return open(path).read(), f"{os.path.basename(path)} (exact-as-run)"
        except OSError as e:
            return f"# could not read {path}: {e}", "read error"
    # fall back to the name-matched config in examples/configs
    stem = run_name[4:] if run_name.startswith("sft_") else run_name
    for c in (f"sft_vlm_{stem}_megatron.yaml", f"sft_vlm_{stem}.yaml",
              f"{stem}_megatron.yaml", f"{stem}.yaml"):
        p = os.path.join(CONFIGS_DIR, c)
        if os.path.exists(p):
            try:
                return open(p).read(), f"{c} (name-matched, NOT a launch snapshot)"
            except OSError:
                pass
    return f"# no config found for '{run_name}'", "none"


def find_training_log(run_name):
    """Locate the node-0 training log for a run. The launcher writes
    <SLURM_LOGS>/<YYYYMMDD>/training_<stem>_node_0_*.log. The <stem> is the launcher's
    LOG_FILE stem (e.g. 'qwen35_4b_vobs2906'), not the full run name, so match by the run
    name's distinctive middle token(s). Returns the newest matching path or None."""
    stem = run_name[4:] if run_name.startswith("sft_") else run_name  # e.g. qwen35_4b_vobs2906_<variant>
    # progressively shorter prefixes so 'sft_qwen35_4b_vobs2906_<variant>' matches a
    # 'training_qwen35_4b_vobs2906_node_0_*.log' (launcher LOG_FILE often drops the variant)
    toks = stem.split("_")
    cands = []
    for k in range(len(toks), 1, -1):
        cands.append("_".join(toks[:k]))
    for dated in sorted(glob.glob(os.path.join(SLURM_LOGS_DIR, "*")), reverse=True):
        if not os.path.isdir(dated):
            continue
        for pref in cands:
            hits = sorted(glob.glob(os.path.join(dated, f"training_{pref}*node_0*.log")),
                          key=_mtime, reverse=True)
            if hits:
                return hits[0]
    return None


def training_log_tail(run_name, n_lines=400):
    """Tail the run's node-0 training log. Returns (text, source_label)."""
    path = find_training_log(run_name)
    if not path:
        return ("(no training log yet — the run hasn't produced a node-0 log. PENDING jobs\n"
                "have none until they land on a node and start.)", "none")
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        tail = "".join(lines[-n_lines:])
        head = f"# {path}\n# (last {min(n_lines, len(lines))} of {len(lines)} lines)\n\n"
        return head + tail, os.path.basename(path)
    except OSError as e:
        return (f"# could not read {path}: {e}", "read error")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def smooth(arr, w):
    if w <= 1 or len(arr) < w:
        return arr
    valid = ~np.isnan(arr)
    arr_clean = np.where(valid, arr, 0.0)
    cumsum = np.cumsum(arr_clean)
    cumcount = np.cumsum(valid.astype(np.float64))
    sums = cumsum[w - 1:] - np.concatenate([[0.0], cumsum[:-w]])
    counts = cumcount[w - 1:] - np.concatenate([[0.0], cumcount[:-w]])
    return np.where(counts > 0, sums / counts, np.nan)


def smooth_x(x, w):
    if w <= 1 or len(x) < w:
        return x
    return x[w // 2: w // 2 + (len(x) - w + 1)]


def _overlay(run_scalars, tag, title, ylabel, smoothing, logy=False):
    """Overlay one tag across selected runs. run_scalars = [(label, scalars_dict)]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.max_open_warning"] = 0  # we hand figs to Gradio; suppress the pile-up warning

    fig, ax = plt.subplots(figsize=(11, 4.5))
    any_data = False
    for i, (label, scalars) in enumerate(run_scalars):
        if tag not in scalars:
            continue
        steps, vals = scalars[tag]
        if len(steps) == 0:
            continue
        any_data = True
        c = COLORS[i % len(COLORS)]
        # raw faint + smoothed bold
        if smoothing > 1 and len(vals) >= smoothing:
            ax.plot(steps, vals, color=c, alpha=0.15, linewidth=0.8)
            sx, sy = smooth_x(steps, smoothing), smooth(vals, smoothing)
            ax.plot(sx, sy, color=c, linewidth=1.8, label=label)
        else:
            ax.plot(steps, vals, color=c, linewidth=1.6, label=label,
                    marker="o" if len(steps) <= 12 else None, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if logy and any_data:
        ax.set_yscale("log")
    if any_data:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f"no '{tag}' in selected runs", ha="center", va="center",
                transform=ax.transAxes, color="#888")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def summary_markdown(run_scalars):
    if not run_scalars:
        return "No runs selected."
    lines = ["### Run summary", ""]
    lines.append("| run | steps | loss (first→last) | min loss | val_loss (last) | "
                 "grad_norm (last) | mean step (s) | tok/s (recent) | checkpoints |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for label, sc in run_scalars:
        loss = sc.get("train/loss", (np.array([]), np.array([])))[1]
        steps = sc.get("train/loss", (np.array([]), np.array([])))[0]
        gn = sc.get("train/grad_norm", (None, np.array([])))[1]
        val = sc.get(VAL_TAG, (None, np.array([])))[1]
        step_t = sc.get("timing/train/total_step_time", (None, np.array([])))[1]
        gtoks = sc.get("train/global_valid_toks", (None, np.array([])))[1]
        ckpt = sc.get("timing/train/checkpointing", (np.array([]), None))[0]
        n = len(steps)
        loss_str = (f"{loss[0]:.3f}→{loss[-1]:.3f}" if len(loss) else "—")
        min_loss = f"{np.nanmin(loss):.3f}" if len(loss) else "—"
        val_str = f"{val[-1]:.4f}" if len(val) else "—"
        gn_str = f"{gn[-1]:.2f}" if len(gn) else "—"
        step_str = f"{np.nanmean(step_t):.0f}" if len(step_t) else "—"
        # recent throughput: mean(global_valid_toks) / mean(step_time) over last 50 steps
        if len(gtoks) and len(step_t):
            k = min(50, len(gtoks), len(step_t))
            mt = np.nanmean(step_t[-k:])
            tps = (np.nanmean(gtoks[-k:]) / mt) if mt > 0 else float("nan")
            tok_str = f"{tps:,.0f}" if np.isfinite(tps) else "—"
        else:
            tok_str = "—"
        ckpt_str = f"{len(ckpt)}" if ckpt is not None else "0"
        lines.append(f"| {label} | {n} | {loss_str} | {min_loss} | {val_str} | "
                     f"{gn_str} | {step_str} | {tok_str} | {ckpt_str} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self, logs_dirs):
        self.logs_dirs = logs_dirs
        self.refresh()

    def refresh(self):
        self.runs = discover_runs(self.logs_dirs)  # [(name, dir, events)]
        self.by_name = {n: (d, e) for n, d, e in self.runs}

    def is_running(self, name):
        """True if the run's latest event file was touched in the last 10 min."""
        import time
        ent = self.by_name.get(name)
        if not ent or not ent[1]:
            return False
        return _mtime(ent[1][-1]) > (time.time() - 600)

    def is_starting(self, name):
        """Launched (config snapshot present) but not emitting tfevents yet."""
        ent = self.by_name.get(name)
        return bool(ent) and not ent[1]

    def _latest_mtime(self, name):
        """Latest activity mtime: last tfevents if any, else the config snapshot."""
        ent = self.by_name.get(name)
        if not ent:
            return 0.0
        run_dir, evs = ent
        if evs:
            return _mtime(evs[-1])
        snaps = glob.glob(os.path.join(run_dir, "config_snapshot*.yaml"))
        return max((_mtime(s) for s in snaps), default=_mtime(run_dir))

    def is_today(self, name):
        """True if the run's latest activity (tfevents OR launch snapshot) is today."""
        import datetime
        ts = self._latest_mtime(name)
        if not ts:
            return False
        return (datetime.date.fromtimestamp(ts) == datetime.date.today())

    def status_prefix(self, name):
        """Display-only status marker (NOT baked into the dropdown value, so choices stay
        stable across refreshes and Gradio never rejects a stale selection)."""
        if self.is_running(name):
            return "🟢 "
        if self.is_starting(name):
            return "⏳ "
        return ""

    def labels(self, today_only=False):
        """Dropdown choices = PLAIN run names (stable values). Status shown elsewhere."""
        return [name for name, _, _ in self.runs
                if (not today_only or self.is_today(name))]

    def resolve(self, label):
        # values ARE plain names now; tolerate a legacy-prefixed value just in case
        return label.replace("🟢 ", "").replace("⏳ ", "").strip()


def build_ui(state):
    with gr.Blocks(title="SFT Training Dashboard") as demo:
        gr.Markdown("# 🏋️ SFT Training Dashboard\n"
                    "Live visibility into VLM SFT runs — loss / grad-norm / lr, val-loss, "
                    "step timing, GPU health, and config drift. 🟢 = active in the last 10 min.")

        with gr.Row():
            run_filter = gr.Radio(
                ["All", "Today"], value="All", label="Show runs",
                info="Today = latest activity (training OR launch) is today; ⏳ = launched, not training yet",
                scale=0)
            run_sel = gr.Dropdown(
                choices=state.labels(), label="Runs (multi-select to compare)",
                multiselect=True, value=state.labels()[:1] if state.labels() else [],
                # don't let a stale client-side selection (a value no longer in choices after
                # a Today-filter / refresh) trip Gradio's preprocess membership check — the
                # server-side handlers filter to valid runs anyway.
                allow_custom_value=True)
            smoothing = gr.Slider(1, 50, value=5, step=1, label="Smoothing window")
            loss_log = gr.Checkbox(
                value=False, label="Loss log-scale",
                info="log y-axis — resolves small late-SFT changes")
            refresh_btn = gr.Button("🔄 Refresh", scale=0)

        summary = gr.Markdown()

        with gr.Tab("📉 Loss & optimization"):
            loss_plot = gr.Plot(label="Training Loss")
            with gr.Row():
                gn_plot = gr.Plot(label="Gradient Norm")
                lr_plot = gr.Plot(label="Learning Rate")
            val_plot = gr.Plot(label="Validation Loss (at save points)")

        with gr.Tab("⏱️ Throughput & timing"):
            with gr.Row():
                steptime_plot = gr.Plot(label="Total Step Time")
                tok_plot = gr.Plot(label="Valid Tokens/sec/GPU")
            with gr.Row():
                policy_plot = gr.Plot(label="Policy Training Time")
                dataproc_plot = gr.Plot(label="Data Processing Time")

        # Per-run subtabs: the selection is dynamic but Gradio's component tree is fixed at build
        # time, so pre-build a POOL of MAX_RUNS subtabs inside each tab; on update we fill the first
        # len(rs) (set label + content + visible) and hide the rest. Lets you flip between runs
        # within GPU / Config / Log instead of one giant concatenated scroll.
        MAX_RUNS = 6
        gpu_subtabs, gpu_mds = [], []
        cfgint_subtabs, config_mds = [], []
        cfgview_subtabs, config_srcs, config_raws = [], [], []
        logview_subtabs, log_srcs, log_raws = [], [], []

        with gr.Tab("🖥️ GPU / host health"):
            with gr.Tabs():
                for i in range(MAX_RUNS):
                    with gr.Tab(f"run {i+1}", visible=(i == 0)) as t:
                        gpu_subtabs.append(t)
                        gpu_mds.append(gr.Markdown("_Select run(s) above._"))

        with gr.Tab("⚙️ Config intelligence"):
            with gr.Tabs():
                for i in range(MAX_RUNS):
                    with gr.Tab(f"run {i+1}", visible=(i == 0)) as t:
                        cfgint_subtabs.append(t)
                        config_mds.append(gr.Markdown("_Select run(s) above to inspect config._"))

        with gr.Tab("📋 Config viewer"):
            with gr.Tabs():
                for i in range(MAX_RUNS):
                    with gr.Tab(f"run {i+1}", visible=(i == 0)) as t:
                        cfgview_subtabs.append(t)
                        config_srcs.append(gr.Markdown("_Select run(s) to view config._"))
                        config_raws.append(gr.Code(label="config (YAML)", language="yaml"))

        with gr.Tab("📜 Log viewer"):
            with gr.Row():
                log_refresh_btn = gr.Button("🔄 Reload logs", scale=0)
            with gr.Tabs():
                for i in range(MAX_RUNS):
                    with gr.Tab(f"run {i+1}", visible=(i == 0)) as t:
                        logview_subtabs.append(t)
                        log_srcs.append(gr.Markdown("_Select run(s) to tail log._"))
                        log_raws.append(gr.Code(label="training log (node-0, tail)"))

        # ----- callbacks -----
        def _selected_scalars(labels):
            out = []
            for lab in labels or []:
                name = state.resolve(lab)
                if name not in state.by_name:
                    continue
                _, events = state.by_name[name]
                # starting runs have no tfevents yet → empty scalars (no curves, but the
                # run still shows + its config/log viewers work)
                out.append((name, get_run_scalars(events) if events else {}))
            return out

        def update(labels, sm, loss_logy=False):
            rs = _selected_scalars(labels)
            # status marker (🟢 active / ⏳ starting) shown in the SUMMARY, not the dropdown
            rs_disp = [(state.status_prefix(n) + n, sc) for n, sc in rs]
            outputs = {
                summary: summary_markdown(rs_disp),
                loss_plot: _overlay(rs, "train/loss", "Training Loss", "loss", sm, logy=loss_logy),
                gn_plot: _overlay(rs, "train/grad_norm", "Gradient Norm", "grad_norm", sm, logy=True),
                lr_plot: _overlay(rs, "train/lr", "Learning Rate", "lr", 1),
                val_plot: _overlay(rs, VAL_TAG, "Validation Loss", "val_loss", 1),
                steptime_plot: _overlay(rs, "timing/train/total_step_time", "Total Step Time", "s", sm),
                tok_plot: _overlay(rs, "timing/train/valid_tokens_per_sec_per_gpu",
                                   "Valid Tokens/sec/GPU", "tok/s/gpu", sm),
                policy_plot: _overlay(rs, "timing/train/policy_training", "Policy Training Time", "s", sm),
                dataproc_plot: _overlay(rs, "timing/train/data_processing", "Data Processing Time", "s", sm),
            }
            # Fill the per-run subtab POOL: one subtab per selected run (up to MAX_RUNS), the rest
            # hidden. Each subtab is labelled with its run name so you flip between all N runs.
            for i in range(MAX_RUNS):
                if i < len(rs):
                    name, sc = rs[i]
                    disp = state.status_prefix(name) + name
                    run_dir = state.by_name[name][0]
                    raw, csrc = raw_config_text(name, run_dir)
                    ltext, lsrc = training_log_tail(name)
                    outputs[gpu_subtabs[i]] = gr.update(visible=True, label=disp)
                    outputs[gpu_mds[i]] = gpu_health_markdown(
                        sc, is_running=state.is_running(name))
                    outputs[cfgint_subtabs[i]] = gr.update(visible=True, label=disp)
                    outputs[config_mds[i]] = config_markdown(name, run_dir)
                    outputs[cfgview_subtabs[i]] = gr.update(visible=True, label=disp)
                    outputs[config_srcs[i]] = f"**{name}** — config source: `{csrc}`"
                    outputs[config_raws[i]] = raw
                    outputs[logview_subtabs[i]] = gr.update(visible=True, label=disp)
                    outputs[log_srcs[i]] = f"**{name}** — log: `{lsrc}`"
                    outputs[log_raws[i]] = ltext
                else:
                    # hide the subtab AND clear its content — every pooled component in all_outputs
                    # MUST get a key each update, or the `d[o]` gather raises KeyError.
                    outputs[gpu_subtabs[i]] = gr.update(visible=False)
                    outputs[gpu_mds[i]] = ""
                    outputs[cfgint_subtabs[i]] = gr.update(visible=False)
                    outputs[config_mds[i]] = ""
                    outputs[cfgview_subtabs[i]] = gr.update(visible=False)
                    outputs[config_srcs[i]] = ""
                    outputs[config_raws[i]] = ""
                    outputs[logview_subtabs[i]] = gr.update(visible=False)
                    outputs[log_srcs[i]] = ""
                    outputs[log_raws[i]] = ""
            return outputs

        # plots + summary, then every pooled per-run subtab component (tabs + their content)
        all_outputs = ([summary, loss_plot, gn_plot, lr_plot, val_plot,
                        steptime_plot, tok_plot, policy_plot, dataproc_plot]
                       + gpu_subtabs + gpu_mds
                       + cfgint_subtabs + config_mds
                       + cfgview_subtabs + config_srcs + config_raws
                       + logview_subtabs + log_srcs + log_raws)

        def update_list(labels, sm, loss_logy):
            d = update(labels, sm, loss_logy)
            return [d[o] for o in all_outputs]

        def do_refresh(labels, sm, flt, loss_logy):
            state.refresh()
            new_choices = state.labels(today_only=(flt == "Today"))
            kept = [l for l in (labels or []) if state.resolve(l) in state.by_name
                    and l in new_choices]
            d = update(kept, sm, loss_logy)
            return [gr.update(choices=new_choices, value=kept)] + [d[o] for o in all_outputs]

        def do_filter(flt, labels, sm, loss_logy):
            new_choices = state.labels(today_only=(flt == "Today"))
            kept = [l for l in (labels or []) if l in new_choices]
            d = update(kept, sm, loss_logy)
            return [gr.update(choices=new_choices, value=kept)] + [d[o] for o in all_outputs]

        def reload_log(labels):
            """Re-tail every selected run's log into its subtab (log_srcs[i], log_raws[i])."""
            names = [state.resolve(l) for l in (labels or [])]
            names = [n for n in names if n in state.by_name][:MAX_RUNS]
            src_out, raw_out = [], []
            for i in range(MAX_RUNS):
                if i < len(names):
                    ltext, lsrc = training_log_tail(names[i])
                    src_out.append(f"**{names[i]}** — log: `{lsrc}`")
                    raw_out.append(ltext)
                else:
                    src_out.append("_Select run(s) to tail log._")
                    raw_out.append("")
            return src_out + raw_out

        run_sel.change(update_list, [run_sel, smoothing, loss_log], all_outputs)
        smoothing.change(update_list, [run_sel, smoothing, loss_log], all_outputs)
        loss_log.change(update_list, [run_sel, smoothing, loss_log], all_outputs)
        run_filter.change(do_filter, [run_filter, run_sel, smoothing, loss_log], [run_sel] + all_outputs)
        refresh_btn.click(do_refresh, [run_sel, smoothing, run_filter, loss_log], [run_sel] + all_outputs)
        log_refresh_btn.click(reload_log, [run_sel], log_srcs + log_raws)
        demo.load(update_list, [run_sel, smoothing, loss_log], all_outputs)

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR)
    ap.add_argument("--port", type=int, default=7875)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true",
                     help="expose a public gradio-share link (expires in 1 week)")
    args = ap.parse_args()

    logs_dirs = [args.logs_dir] + [d for d in EXTRA_LOGS_DIRS if os.path.isdir(d)]
    state = AppState(logs_dirs)
    print(f"[sft-dashboard] {len(state.runs)} runs from {logs_dirs}")
    demo = build_ui(state)
    try:
        demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                    theme=gr.themes.Soft())
    except TypeError:
        # Older Gradio (<6): theme belongs on Blocks, not launch — fall back plain.
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
