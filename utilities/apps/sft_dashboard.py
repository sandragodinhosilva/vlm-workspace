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
    """Return [(run_name, run_dir, [event_files_sorted_by_exp])] for valid SFT runs."""
    runs = {}
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
    # newest first by latest event mtime
    items = sorted(runs.items(), key=lambda kv: _mtime(kv[1][1][-1]), reverse=True)
    return [(name, rd, evs) for name, (rd, evs) in items]


def load_run_scalars(event_files):
    """Pick, per tag, the LATEST exp_* that contains it — taken WHOLE.

    SFT resumes write a fresh exp_NNN whose step counter RESTARTS from 1 (unlike
    GRPO, which continues). So a per-step "later wins" merge would splice two
    different training runs at an invisible seam (e.g. take steps 1..649 from the
    resumed exp, then 650..675 from the abandoned earlier exp) — a curve that
    belongs to neither run. Instead we treat each launch's exp as authoritative
    for its own tags and show only the latest exp that logged a given tag. This
    means the displayed curve is always exactly one contiguous training segment.
    Earlier (abandoned/crashed) exps are dropped for that tag, not blended.

    Returns {tag: (steps_np_sorted, values_np)}.
    """
    out = {}  # tag -> (steps, vals) from the latest exp that had this tag
    for ev in event_files:  # ascending exp order; later overwrites wholesale
        try:
            tag_data = _read_event_file(ev)
        except Exception:
            continue
        for tag, (steps, vals) in tag_data.items():
            order = np.argsort(steps, kind="stable")
            out[tag] = (steps[order], vals[order])  # last exp with this tag wins, whole
    return out


# Cache keyed on (event paths, latest mtime) so live runs refresh but idle ones don't re-parse.
_SCALAR_CACHE = {}


def get_run_scalars(event_files):
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
        if not ent:
            return False
        return _mtime(ent[1][-1]) > (time.time() - 600)

    def labels(self):
        return [("🟢 " if self.is_running(name) else "") + name
                for name, _, _ in self.runs]

    def resolve(self, label):
        return label.replace("🟢 ", "").strip()


def build_ui(state):
    with gr.Blocks(title="SFT Training Dashboard") as demo:
        gr.Markdown("# 🏋️ SFT Training Dashboard\n"
                    "Live visibility into VLM SFT runs — loss / grad-norm / lr, val-loss, "
                    "step timing, GPU health, and config drift. 🟢 = active in the last 10 min.")

        with gr.Row():
            run_sel = gr.Dropdown(
                choices=state.labels(), label="Runs (multi-select to compare)",
                multiselect=True, value=state.labels()[:1] if state.labels() else [])
            smoothing = gr.Slider(1, 50, value=5, step=1, label="Smoothing window")
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

        with gr.Tab("🖥️ GPU / host health"):
            gpu_md = gr.Markdown()

        with gr.Tab("⚙️ Config intelligence"):
            config_md = gr.Markdown(
                "_Select a single run above to inspect its config._")

        # ----- callbacks -----
        def _selected_scalars(labels):
            out = []
            for lab in labels or []:
                name = state.resolve(lab)
                if name not in state.by_name:
                    continue
                _, events = state.by_name[name]
                out.append((name, get_run_scalars(events)))
            return out

        def update(labels, sm):
            rs = _selected_scalars(labels)
            outputs = {
                summary: summary_markdown(rs),
                loss_plot: _overlay(rs, "train/loss", "Training Loss", "loss", sm),
                gn_plot: _overlay(rs, "train/grad_norm", "Gradient Norm", "grad_norm", sm, logy=True),
                lr_plot: _overlay(rs, "train/lr", "Learning Rate", "lr", 1),
                val_plot: _overlay(rs, VAL_TAG, "Validation Loss", "val_loss", 1),
                steptime_plot: _overlay(rs, "timing/train/total_step_time", "Total Step Time", "s", sm),
                tok_plot: _overlay(rs, "timing/train/valid_tokens_per_sec_per_gpu",
                                   "Valid Tokens/sec/GPU", "tok/s/gpu", sm),
                policy_plot: _overlay(rs, "timing/train/policy_training", "Policy Training Time", "s", sm),
                dataproc_plot: _overlay(rs, "timing/train/data_processing", "Data Processing Time", "s", sm),
            }
            # GPU + config from the FIRST selected run
            if rs:
                first_name = rs[0][0]
                outputs[gpu_md] = gpu_health_markdown(
                    rs[0][1], is_running=state.is_running(first_name))
                run_dir = state.by_name[first_name][0]
                outputs[config_md] = config_markdown(first_name, run_dir)
            else:
                outputs[gpu_md] = "_Select a run._"
                outputs[config_md] = "_Select a single run above to inspect its config._"
            return outputs

        all_outputs = [summary, loss_plot, gn_plot, lr_plot, val_plot,
                       steptime_plot, tok_plot, policy_plot, dataproc_plot,
                       gpu_md, config_md]

        def update_list(labels, sm):
            d = update(labels, sm)
            return [d[o] for o in all_outputs]

        def do_refresh(labels, sm):
            state.refresh()
            new_choices = state.labels()
            kept = [l for l in (labels or []) if state.resolve(l) in state.by_name]
            d = update(kept, sm)
            return [gr.update(choices=new_choices, value=kept)] + [d[o] for o in all_outputs]

        run_sel.change(update_list, [run_sel, smoothing], all_outputs)
        smoothing.change(update_list, [run_sel, smoothing], all_outputs)
        refresh_btn.click(do_refresh, [run_sel, smoothing], [run_sel] + all_outputs)
        demo.load(update_list, [run_sel, smoothing], all_outputs)

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR)
    ap.add_argument("--port", type=int, default=7875)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    logs_dirs = [args.logs_dir] + [d for d in EXTRA_LOGS_DIRS if os.path.isdir(d)]
    state = AppState(logs_dirs)
    print(f"[sft-dashboard] {len(state.runs)} runs from {logs_dirs}")
    demo = build_ui(state)
    try:
        demo.launch(server_name=args.host, server_port=args.port, share=False,
                    theme=gr.themes.Soft())
    except TypeError:
        # Older Gradio (<6): theme belongs on Blocks, not launch — fall back plain.
        demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
