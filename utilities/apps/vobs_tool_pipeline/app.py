#!/usr/bin/env python3
"""
pipeline-inspector — VObs-tool-SFT pipeline quality inspector (port 7880).

Row-by-row inspection of the run_tool_sft_4k.py output (gen → rewrite → judge →
regen trail, the rep's own video, the per-step `step_metrics` block) + a
run-level overview ("how good is the pipeline?") per flavor.

CORE PRINCIPLE (Sandra, 2026-07-15): this app ONLY MIRRORS the produced JSONL.
The row is self-describing (video_frames / images_path / fps / need_to_flip /
step_metrics — producer fields added 2026-07-15). If a row lacks a field the
app needs, the app shows a LOUD distinct gap state and the fix happens at the
source (run_tool_sft_4k.py), never in here.

FUTURE-PROOFING CONTRACT (the pipeline keeps changing — flavors/steps/fields):
  * `step_metrics` is rendered by ITERATING the dict: dict-valued keys = stages
    (rows), union of their sub-keys = columns. A new stage or field in
    step_metrics.py just appears — no code change.
  * The aggregate is IMPORTED from the producer's own step_metrics.py
    (`summarize_step_metrics`) — the schema lives in ONE place. If that module
    changes, this app follows automatically. sys.path points at
    /home/sgsilva/vlm-post-training/visual_obs (the coupling).
  * Flavors / prompt_origin / judge_verdict_kind filter choices are derived
    from the loaded rows, never a hardcoded list.
  * Pipeline stages in the trail are DISCOVERED from field presence
    (all_attempts / rewrite_prompt / judge_attempts[]), not a fixed branch.
  * Every top-level row key not explicitly rendered is dumped in the
    "All row fields" panel, so a future field is never invisible.
  * Missing fields render as a distinct "not present" state — never a crash,
    never a silent blank ([[feedback_no_silent_fail]]).

Reuses: scripts/nav_widgets.py (house nav + counter), scripts/row_video.py
(fps-correct, mirror-correct frames→MP4, lifted from video_sft), the producer's
step_metrics.summarize_step_metrics.

Launch (registry): ~/utilities/apps/launch_app.sh pipeline-inspector
Env: DEFAULT_JSONL=<kept-rows jsonl> (its sibling .dropped.jsonl auto-loads)
"""

import argparse
import html
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "/home/sgsilva/utilities/apps/scripts")
import nav_widgets  # noqa: E402
from row_video import build_row_video  # noqa: E402

# The producer's own metrics module — the single schema source of truth.
_VISUAL_OBS_DIR = "/home/sgsilva/vlm-post-training/visual_obs"
sys.path.insert(0, _VISUAL_OBS_DIR)
try:
    from step_metrics import summarize_step_metrics  # noqa: E402
    _SUMMARIZE_IMPORT_ERROR = None
except Exception as _e:  # loud, not fatal — per-row rendering still works
    summarize_step_metrics = None
    _SUMMARIZE_IMPORT_ERROR = repr(_e)

import gradio as gr  # noqa: E402

DATASET_ROOT = os.environ.get(
    "DATASET_ROOT", "/mnt/data/sgsilva/datasets/1806/vobs_tool_sft_4k")
DEFAULT_JSONL = os.environ.get(
    "DEFAULT_JSONL",
    "/mnt/data/sgsilva/datasets/1806/vobs_tool_sft_4k/smoke_selfdesc_0715/smoke.jsonl")
VIDEO_CACHE_DIR = os.environ.get(
    "VIDEO_CACHE_DIR", "/mnt/data/sgsilva/tmp/vobs_tool_pipeline_videos")

ALL = "(all)"

# ---------------------------------------------------------------------------
# Data loading — kept rows + sibling .dropped.jsonl, tagged with _disposition
# ---------------------------------------------------------------------------

STATE: Dict = {"rows": [], "path": None}


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # a mid-write torn line (live run) — skip, don't crash
                    continue
    return rows


def load_run(jsonl_path: str) -> str:
    """Load kept rows + the sibling .dropped.jsonl. Returns a status line."""
    p = Path(jsonl_path).expanduser()
    kept = _read_jsonl(p)
    for r in kept:
        r["_disposition"] = "kept"
    dropped_path = Path(str(p) + ".dropped.jsonl")
    dropped = _read_jsonl(dropped_path)
    for r in dropped:
        r["_disposition"] = "dropped"
    STATE["rows"] = kept + dropped
    STATE["path"] = str(p)
    if not p.exists():
        return f"🔴 file not found: `{p}`"
    return (f"Loaded **{len(kept)} kept** (`{p.name}`) + **{len(dropped)} dropped** "
            f"(`{dropped_path.name}`) from `{p.parent}`")


def discover_runs() -> List[str]:
    """Every kept-rows JSONL under DATASET_ROOT (newest first). ckpt/dropped
    sidecars excluded. Free-text path box covers anything outside the root."""
    root = Path(DATASET_ROOT)
    if not root.is_dir():
        return []
    hits = [q for q in root.glob("*/*.jsonl")
            if not q.name.endswith((".ckpt.jsonl", ".dropped.jsonl"))]
    hits.sort(key=lambda q: q.stat().st_mtime, reverse=True)
    return [str(q) for q in hits]


# ---------------------------------------------------------------------------
# Filters — choices derived from the data, never hardcoded
# ---------------------------------------------------------------------------

def _choices(field: str) -> List[str]:
    vals = sorted({str(r.get(field)) for r in STATE["rows"] if r.get(field) is not None})
    return [ALL] + vals


def disposition_choices() -> List[str]:
    # judge-excluded = dropped specifically by the Gate-4 judge (drop_reason
    # sentinel `judge:<verdict_kind>`) — a subset of dropped worth first-class access.
    return [ALL, "kept", "dropped", "judge-excluded"]


def _matches(row: Dict, disposition: str, flavor: str, origin: str, verdict: str) -> bool:
    if disposition == "kept" and row["_disposition"] != "kept":
        return False
    if disposition == "dropped" and row["_disposition"] != "dropped":
        return False
    if disposition == "judge-excluded" and not str(row.get("drop_reason") or "").startswith("judge:"):
        return False
    if flavor != ALL and str(row.get("flavor")) != flavor:
        return False
    if origin != ALL and str(row.get("prompt_origin")) != origin:
        return False
    if verdict != ALL and str(row.get("judge_verdict_kind")) != verdict:
        return False
    return True


def filtered(disposition: str, flavor: str, origin: str, verdict: str) -> List[int]:
    rows = STATE["rows"]
    return nav_widgets.filtered_indices(
        len(rows), lambda i: _matches(rows[i], disposition, flavor, origin, verdict))


def _scope_label(disposition: str, flavor: str, origin: str, verdict: str) -> Optional[str]:
    parts = [v for v in (disposition, flavor, origin, verdict) if v != ALL]
    return " · ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# HTML helpers — verbatim <pre> blocks inside <details> (stage count is data-
# driven, so the trail is ONE html component, not fixed widgets per stage)
# ---------------------------------------------------------------------------

def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _pre(text, empty="<not present on row>") -> str:
    body = _esc(text) if (text is not None and text != "") else _esc(empty)
    return (f"<pre style='white-space:pre-wrap;word-break:break-word;"
            f"background:var(--background-fill-secondary);padding:10px;"
            f"border-radius:6px;max-height:520px;overflow:auto;font-size:12px'>"
            f"{body}</pre>")


def _details(summary: str, body: str, open_: bool = False) -> str:
    return (f"<details{' open' if open_ else ''} style='margin:6px 0;border:1px solid "
            f"var(--border-color-primary);border-radius:8px;padding:6px 10px'>"
            f"<summary style='cursor:pointer;font-weight:600'>{summary}</summary>"
            f"{body}</details>")


def _chip(label: str, value, color: str = "var(--background-fill-secondary)") -> str:
    return (f"<span style='display:inline-block;background:{color};border-radius:12px;"
            f"padding:2px 10px;margin:2px 4px 2px 0;font-size:12px'>"
            f"<b>{_esc(label)}</b>: {_esc(value)}</span>")


def _text_of(content) -> str:
    """messages[].content may be a str or a multimodal list — render its text."""
    if isinstance(content, list):
        return " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    return "" if content is None else str(content)


# ---------------------------------------------------------------------------
# Per-row renderers
# ---------------------------------------------------------------------------

def render_header(r: Dict) -> str:
    disp = r.get("_disposition")
    disp_color = "#1a7f37" if disp == "kept" else "#b91c1c"
    bits = [
        f"<span style='background:{disp_color};color:white;border-radius:12px;"
        f"padding:2px 12px;font-weight:700'>{_esc(disp).upper()}</span>",
        _chip("flavor", r.get("flavor")),
        _chip("session|rep", f"{r.get('session_id')}|{r.get('rep_index')}"),
        _chip("exercise", r.get("exercise_id")),
        _chip("prompt_origin", r.get("prompt_origin")),
        _chip("status", r.get("status")),
    ]
    if r.get("drop_reason"):
        bits.append(_chip("drop_reason", r.get("drop_reason"), "#fde8e8"))
    if r.get("partial_row"):
        bits.append(_chip("partial_row", True, "#fde8e8"))
    # judge outcome line — headline fields the handoff names explicitly
    jv = r.get("judge_verdict_kind")
    if jv is not None:
        jcolor = "#dcfce7" if jv == "pass" else "#fde8e8"
        bits.append(_chip("judge", jv, jcolor))
        if r.get("judge_tags"):
            bits.append(_chip("judge_tags", ", ".join(map(str, r["judge_tags"])), "#fef9c3"))
        bits.append(_chip("accepted_after_regen", r.get("judge_accepted_after_regen")))
    gt = (f"GT severity_scores: <code>{_esc(r.get('severity_scores'))}</code> · "
          f"effectiveness={_esc(r.get('effectiveness'))} · "
          f"injury_risk={_esc(r.get('injury_risk'))}")
    return ("<div>" + "".join(bits) + f"<div style='margin-top:6px;font-size:13px'>{gt}</div></div>")


def _fmt_cell(v) -> str:
    """One step_metrics leaf → display. Score dicts get the is_perfect-first
    treatment: is_perfect is THE signal; f1 is secondary (f1=0.0 WITH
    is_perfect=true is a zero-GT-error rep, NOT a failure — _score_row's F1
    formula returns 0.0 when there is nothing to detect)."""
    if isinstance(v, dict) and "is_perfect" in v:
        ok = v.get("is_perfect")
        f1 = v.get("f1")
        f1s = "—" if f1 is None else f"{f1:.3f}"
        if ok:
            note = " <small>(no GT errors)</small>" if f1 == 0.0 else ""
            return f"✅ perfect <small>f1={f1s}</small>{note}"
        return f"<span style='color:#b91c1c;font-weight:700'>✗ not perfect</span> <small>f1={f1s}</small>"
    if isinstance(v, dict):
        return _esc(json.dumps(v, default=str))
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.3f}"
    if v is None:
        return "<span style='opacity:.45'>—</span>"
    return _esc(v)


def render_step_metrics(r: Dict) -> str:
    sm = r.get("step_metrics")
    if not sm:
        return ("<div style='color:#b91c1c;font-weight:700'>🔴 step_metrics not present "
                "on this row — pre-2026-07-15 run or a partial (timeout/exception) stub. "
                "Regenerate at the producer if this is unexpected.</div>")
    # Schema-driven split: dict-valued keys = stages (table rows, insertion
    # order); everything else = row-level scalars shown as chips.
    stages = {k: v for k, v in sm.items() if isinstance(v, dict)}
    scalars = {k: v for k, v in sm.items() if not isinstance(v, dict)}

    chips = "".join(_chip(k, "—" if v is None else v,
                          "#fde8e8" if (k == "ae_contradiction" and v is True) else
                          "var(--background-fill-secondary)")
                    for k, v in scalars.items())

    # Column order = first-seen across stages, so new fields just appear.
    cols: List[str] = []
    for st in stages.values():
        for k in st:
            if k not in cols:
                cols.append(k)
    # Change-ratio fields are the headline "how much repair" signal — flag them.
    hot = {"changed_ratio", "regen_changed_ratio", "delta_chars", "regen_delta_chars"}
    head = "".join(
        f"<th style='padding:4px 8px;text-align:left;"
        f"{'background:#fff7ed' if c in hot else ''}'>{_esc(c)}</th>" for c in cols)
    body_rows = []
    for name, st in stages.items():
        tds = "".join(
            f"<td style='padding:4px 8px;{'background:#fff7ed' if c in hot else ''}'>"
            f"{_fmt_cell(st.get(c)) if c in st else '<span style=\"opacity:.45\">·</span>'}</td>"
            for c in cols)
        body_rows.append(f"<tr><td style='padding:4px 8px;font-weight:700'>{_esc(name)}</td>{tds}</tr>")
    if stages.get("judge") is None and "judge" in sm:
        body_rows.append("<tr><td style='padding:4px 8px;font-weight:700'>judge</td>"
                         f"<td colspan='{len(cols)}' style='padding:4px 8px;opacity:.6'>"
                         "not run (— null block)</td></tr>")
    table = (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;"
             f"font-size:12.5px'><tr><th style='padding:4px 8px'></th>{head}</tr>"
             + "".join(body_rows) + "</table></div>")
    return chips + table


def _attempt_shape(a: Dict) -> str:
    """Human tool-shape label; delegates to the producer's flavor_classifier
    (single source of truth, same as preview_tool_sft_pipeline._shape)."""
    try:
        from flavor_classifier import classify_counts
        n_calls = a.get("n_tool_calls", 0) or 0
        n_q = len(a.get("tool_calls", []) or [])
        kind = classify_counts(n_calls, n_q)
        return f"{kind}(zero-call)" if n_calls == 0 else f"{kind}({n_calls} call/{n_q} q)"
    except Exception:
        return f"calls={a.get('n_tool_calls')}"


def render_trail(r: Dict) -> str:
    """The gen → rewrite → judge(+regen) trail. Stages are DISCOVERED from
    field presence, prompts shown VERBATIM (the strings the driver persisted at
    send time — never reconstructed). Mirrors preview_tool_sft_pipeline.py."""
    out = []

    # ① GENERATION — best-of-K attempts share one prompt (shown once when identical)
    attempts = r.get("all_attempts") or []
    gen_prompt = r.get("generation_prompt") or ""
    if attempts or gen_prompt or r.get("raw_model_output"):
        parts = []
        prompts = {a.get("generation_prompt", "") for a in attempts} or {gen_prompt}
        if len(prompts) == 1:
            parts.append(_details(
                f"① GENERATION PROMPT — verbatim, shared by all {max(len(attempts),1)} "
                "best-of-K attempts (they differ only by sampling seed)",
                _pre(gen_prompt or next(iter(prompts)))))
        sel_raw = r.get("raw_model_output")
        if attempts:
            for a in attempts:
                idx = a.get("attempt_idx")
                selected = sel_raw is not None and a.get("raw_model_output") == sel_raw
                met = (f"f1={a.get('f1')} · is_perfect={a.get('is_perfect')} · "
                       f"status={_esc(a.get('status'))} · shape={_esc(_attempt_shape(a))}")
                label = (f"① attempt {idx} — {met}"
                         + ("  ⭐ SELECTED (best-of-K winner)" if selected else ""))
                body = ""
                if len(prompts) > 1:
                    body += "<div><b>prompt (differs from other attempts):</b></div>" \
                            + _pre(a.get("generation_prompt"))
                body += _pre(a.get("raw_model_output"), empty="<no raw output>")
                parts.append(_details(label, body, open_=False))
        else:
            parts.append(_details("① RAW MODEL OUTPUT (no per-attempt trail on row)",
                                  _pre(r.get("raw_model_output"))))
        out.append("<h4>① Generation — teacher best-of-K</h4>" + "".join(parts))

    # ② REWRITE (stage-2 GT-align / pass-2 override; multimodal in the real call)
    if r.get("rewrite_prompt"):
        meta = (f"kind={_esc(r.get('rewrite_kind'))} · applied={r.get('rewrite_applied')} · "
                f"failed_reason={_esc(r.get('rewrite_failed_reason'))}")
        out.append(
            "<h4>② Rewrite — GT-align pass (the rep VIDEO is attached in the real call)</h4>"
            f"<div style='font-size:13px;margin-bottom:4px'>{meta}</div>"
            + _details("② REWRITE PROMPT (verbatim, incl. embedded transcript)",
                       _pre(r.get("rewrite_prompt")))
            + _details("② REWRITE RAW RESPONSE", _pre(r.get("rewrite_raw_response"))))
    elif "rewrite_applied" in r:
        out.append("<h4>② Rewrite</h4><div style='opacity:.7'>no rewrite — natural "
                   f"generation already on GT (rewrite_applied={r.get('rewrite_applied')})</div>")

    # ③ JUDGE (+ regen) — one block per judge_attempts entry, depth is data-driven
    jatt = r.get("judge_attempts") or []
    if jatt:
        parts = [f"<div style='font-size:13px;margin-bottom:4px'>final verdict: "
                 f"<b>{_esc(r.get('judge_verdict_kind'))}</b> · tags={_esc(r.get('judge_tags'))} · "
                 f"accepted_after_regen={r.get('judge_accepted_after_regen')} · "
                 f"notes={_esc(r.get('judge_notes'))}</div>"]
        for a in jatt:
            n = a.get("attempt")
            if a.get("regen_prompt"):
                parts.append(_details(
                    f"③ attempt {n} — REGEN PROMPT (rewrite re-run with the judge's "
                    "correction hint)", _pre(a["regen_prompt"])))
                parts.append(_details(f"③ attempt {n} — REGEN RAW RESPONSE",
                                      _pre(a.get("regen_raw_response"))))
            v = a.get("judge_verdict") or {}
            vline = (f" → pass={v.get('pass')} tags={v.get('tags')}"
                     if v else " → <no parsed verdict>")
            parts.append(_details(f"③ attempt {n} — JUDGE PROMPT (verbatim){_esc(vline)}",
                                  _pre(a.get("judge_prompt"))))
            parts.append(_details(f"③ attempt {n} — JUDGE RAW RESPONSE",
                                  _pre(a.get("judge_raw_response"))))
        out.append("<h4>③ Gate-4 judge (+ regen)</h4>" + "".join(parts))
    else:
        out.append("<h4>③ Gate-4 judge</h4><div style='opacity:.7'>not run for this row "
                   "(--no-judge run, or dropped before the judge)</div>")

    if not out:
        return "<div style='opacity:.7'>no pipeline-trail fields on this row</div>"
    return "".join(out)


def render_final_messages(r: Dict) -> str:
    msgs = r.get("messages") or []
    if not msgs:
        return ("<div style='color:#b91c1c;font-weight:700'>no `messages` on this row"
                " (dropped before packing, or a partial stub)</div>")
    role_colors = defaultdict(lambda: "var(--background-fill-secondary)",
                              {"assistant": "#eef6ff", "tool": "#f3ffe6",
                               "system": "#fdf2f8"})
    parts = []
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        parts.append(_details(
            f"«{role.upper()}» — turn {i}",
            f"<div style='background:{role_colors[role]};border-radius:6px'>"
            + _pre(_text_of(m.get("content"))) + "</div>",
            open_=(role in ("assistant", "tool"))))
    return (f"<div style='font-size:13px;margin-bottom:4px'>{len(msgs)} turns — this is "
            "what SFT trains on (loss masks all non-assistant roles)</div>" + "".join(parts))


# Keys already rendered somewhere above — everything else lands in the dump panel.
_RENDERED_KEYS = {
    "_disposition", "flavor", "session_id", "rep_index", "exercise_id",
    "prompt_origin", "status", "drop_reason", "severity_scores", "effectiveness",
    "injury_risk", "judge_verdict_kind", "judge_tags", "judge_accepted_after_regen",
    "judge_notes", "judge_attempts", "step_metrics", "messages", "all_attempts",
    "generation_prompt", "raw_model_output", "rewrite_prompt", "rewrite_raw_response",
    "rewrite_applied", "rewrite_kind", "rewrite_failed_reason",
    "video_frames", "images_path", "fps", "video_fps", "need_to_flip",
    "num_frames", "num_frames_attached", "partial_row",
}


def render_other_fields(r: Dict) -> str:
    rest = {k: v for k, v in r.items() if k not in _RENDERED_KEYS}
    return _pre(json.dumps(rest, indent=2, default=str, ensure_ascii=False),
                empty="<no additional fields>")


# ---------------------------------------------------------------------------
# Row display driver
# ---------------------------------------------------------------------------

def show_row(abs_idx: int, disposition: str, flavor: str, origin: str, verdict: str):
    rows = STATE["rows"]
    sel = filtered(disposition, flavor, origin, verdict)
    if not rows:
        empty = "<div style='opacity:.6'>no rows loaded</div>"
        return (None, "load a run first", empty, empty, empty, empty, empty,
                "No samples loaded", abs_idx)
    if abs_idx not in sel and sel:
        abs_idx = sel[0]
    abs_idx = max(0, min(abs_idx, len(rows) - 1))
    r = rows[abs_idx]

    try:
        video, vstatus = build_row_video(r, VIDEO_CACHE_DIR)
    except Exception as e:  # encoding failure is loud, never a blank
        video, vstatus = None, f"🔴 video encode failed: {_esc(e)}"

    pos = (sel.index(abs_idx) + 1) if abs_idx in sel else 0
    counter = nav_widgets.format_scoped_counter(
        abs_idx, len(rows), pos, len(sel),
        scope=_scope_label(disposition, flavor, origin, verdict))
    return (video, vstatus, render_header(r), render_step_metrics(r),
            render_trail(r), render_final_messages(r), render_other_fields(r),
            counter, abs_idx)


def nav(delta: Optional[int], abs_idx: int, disposition, flavor, origin, verdict):
    sel = filtered(disposition, flavor, origin, verdict)
    if delta is None:
        new = nav_widgets.random_filtered(sel, len(STATE["rows"]))
    else:
        new = nav_widgets.step_filtered(abs_idx, delta, sel)
    return show_row(new, disposition, flavor, origin, verdict)


# ---------------------------------------------------------------------------
# Run-level overview — REUSES the producer's summarize_step_metrics()
# ---------------------------------------------------------------------------

def overview():
    path = STATE["path"]
    if not path:
        return "<div>load a run first</div>", None
    if summarize_step_metrics is None:
        return (f"<div style='color:#b91c1c'>🔴 could not import the producer's "
                f"step_metrics.py: <code>{_esc(_SUMMARIZE_IMPORT_ERROR)}</code></div>", None)

    paths = [path, path + ".dropped.jsonl"]
    lines: List[str] = []
    agg = summarize_step_metrics(paths, print_fn=lines.append)

    # Schema-driven aggregate table: rows=flavors, cols=union of agg keys.
    htm = []
    if agg:
        cols: List[str] = []
        for d in agg.values():
            for k in d:
                if k not in cols:
                    cols.append(k)
        head = "".join(f"<th style='padding:4px 8px;text-align:left'>{_esc(c)}</th>"
                       for c in cols)
        rows_h = []
        for fl in sorted(agg):
            tds = "".join(f"<td style='padding:4px 8px'>{_fmt_cell(agg[fl].get(c))}</td>"
                          for c in cols)
            rows_h.append(f"<tr><td style='padding:4px 8px;font-weight:700'>{_esc(fl)}</td>{tds}</tr>")
        htm.append("<h4>step_metrics aggregate per flavor — "
                   "computed by the producer's own <code>summarize_step_metrics()</code> "
                   "(kept + dropped rows; rows without step_metrics skipped)</h4>"
                   f"<div style='overflow-x:auto'><table style='border-collapse:collapse;"
                   f"font-size:12.5px'><tr><th style='padding:4px 8px'>flavor</th>{head}</tr>"
                   + "".join(rows_h) + "</table></div>")
        htm.append("<div style='font-size:12px;opacity:.75'>final_not_perfect_count "
                   "should be ~0 — the 100% invariant (every step after raw generation "
                   "is_perfect). f1=0.0 with is_perfect=true = zero-GT-error rep, "
                   "not a failure.</div>")
    else:
        htm.append("<div style='color:#b91c1c;font-weight:700'>🔴 no rows with "
                   "step_metrics in this run (pre-2026-07-15 output) — the aggregate "
                   "needs a regenerated run.</div>")

    # Generic disposition tallies per flavor (from the loaded rows — works even
    # without step_metrics): kept / judge verdicts / drop reasons.
    per_flavor: Dict[str, Counter] = defaultdict(Counter)
    for r in STATE["rows"]:
        fl = str(r.get("flavor"))
        per_flavor[fl][r["_disposition"]] += 1
        if r.get("judge_verdict_kind"):
            per_flavor[fl][f"judge:{r['judge_verdict_kind']}"] += 1
        if r.get("drop_reason"):
            per_flavor[fl][f"drop:{r['drop_reason']}"] += 1
    keys: List[str] = []
    for c in per_flavor.values():
        for k in c:
            if k not in keys:
                keys.append(k)
    head = "".join(f"<th style='padding:4px 8px;text-align:left'>{_esc(k)}</th>" for k in keys)
    rows_h = []
    for fl in sorted(per_flavor):
        tds = "".join(f"<td style='padding:4px 8px'>{per_flavor[fl].get(k) or '—'}</td>"
                      for k in keys)
        rows_h.append(f"<tr><td style='padding:4px 8px;font-weight:700'>{_esc(fl)}</td>{tds}</tr>")
    htm.append("<h4>Disposition / judge / drop tallies per flavor (loaded rows)</h4>"
               f"<div style='overflow-x:auto'><table style='border-collapse:collapse;"
               f"font-size:12.5px'><tr><th style='padding:4px 8px'>flavor</th>{head}</tr>"
               + "".join(rows_h) + "</table></div>")

    htm.append(_details("raw aggregate table (as printed at run end)",
                        _pre("\n".join(lines))))

    # Change-ratio distributions (rewrite + judge-regen) per flavor.
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        series: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in STATE["rows"]:
            sm = r.get("step_metrics") or {}
            fl = str(r.get("flavor"))
            rw = (sm.get("rewrite") or {}).get("changed_ratio")
            if rw is not None:
                series["rewrite changed_ratio"][fl].append(rw)
            jd = sm.get("judge") or {}
            if jd.get("n_regen_calls"):
                rg = jd.get("regen_changed_ratio")
                if rg is not None:
                    series["judge regen_changed_ratio (regen fired)"][fl].append(rg)
        panels = [(t, d) for t, d in series.items() if d]
        if panels:
            fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 3.2))
            if len(panels) == 1:
                axes = [axes]
            for ax, (title, d) in zip(axes, panels):
                for fl in sorted(d):
                    ax.hist(d[fl], bins=20, range=(0, 1), alpha=0.55, label=f"{fl} (n={len(d[fl])})")
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("changed_ratio (0=untouched, 1=fully replaced)", fontsize=8)
                ax.legend(fontsize=8)
            fig.tight_layout()
    except Exception as e:
        htm.append(f"<div style='opacity:.7'>histogram unavailable: {_esc(e)}</div>")

    return "".join(htm), fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="pipeline-inspector — VObs-tool-SFT",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🔬 pipeline-inspector — VObs-tool-SFT pipeline quality "
                    "(gen → rewrite → judge → regen · row video · step_metrics)")

        with gr.Row():
            run_dd = gr.Dropdown(choices=discover_runs(), value=None,
                                 label=f"Runs under {DATASET_ROOT} (newest first)",
                                 scale=3, allow_custom_value=True)
            path_tb = gr.Textbox(value=DEFAULT_JSONL, label="…or kept-rows JSONL path "
                                 "(sibling .dropped.jsonl auto-loads)", scale=3)
            load_btn = gr.Button("Load / Reload", variant="primary", scale=1)
        load_status = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Row inspector"):
                with gr.Row():
                    disp_dd = gr.Dropdown(choices=disposition_choices(), value=ALL,
                                          label="kept / dropped / judge-excluded")
                    flavor_dd = gr.Dropdown(choices=[ALL], value=ALL, label="flavor")
                    origin_dd = gr.Dropdown(choices=[ALL], value=ALL, label="prompt_origin")
                    verdict_dd = gr.Dropdown(choices=[ALL], value=ALL,
                                             label="judge_verdict_kind")
                prev_btn, next_btn, random_btn, refresh_btn, counter_md = \
                    nav_widgets.make_nav_row()
                jump_input, jump_btn = nav_widgets.make_jump_row("Jump to row index (0-based)")

                header_html = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=2):
                        video = gr.Video(label="the rep video, from the row's OWN "
                                         "video_frames (fps + mirror per row)", height=420)
                        video_status = gr.Markdown()
                    with gr.Column(scale=3):
                        gr.Markdown("**step_metrics** — schema-driven (stages/fields "
                                    "iterated from the row; source of truth = producer's "
                                    "`step_metrics.py`)")
                        metrics_html = gr.HTML()
                trail_html = gr.HTML(label="pipeline trail")
                gr.Markdown("**FINAL shipped `messages`** — what SFT actually trains on")
                final_html = gr.HTML()
                with gr.Accordion("All row fields not rendered above (future-proof dump)",
                                  open=False):
                    other_html = gr.HTML()

            with gr.Tab("Run overview — how good is the pipeline?"):
                overview_btn = gr.Button("Compute overview (kept + dropped)",
                                         variant="primary")
                overview_html = gr.HTML()
                overview_plot = gr.Plot(label="change-ratio distributions per flavor")

        idx_state = gr.State(0)

        row_outputs = [video, video_status, header_html, metrics_html, trail_html,
                       final_html, other_html, counter_md, idx_state]
        filter_inputs = [disp_dd, flavor_dd, origin_dd, verdict_dd]

        def do_load(dd_path, tb_path):
            path = dd_path or tb_path
            status = load_run(path)
            upd = [gr.update(choices=_choices(f), value=ALL)
                   for f in ("flavor", "prompt_origin", "judge_verdict_kind")]
            first = show_row(0, ALL, ALL, ALL, ALL)
            return [status, gr.update(value=ALL)] + upd + list(first)

        load_btn.click(do_load, [run_dd, path_tb],
                       [load_status, disp_dd, flavor_dd, origin_dd, verdict_dd]
                       + row_outputs)

        for dd in filter_inputs:
            dd.change(lambda i, d, f, o, v: show_row(i, d, f, o, v),
                      [idx_state] + filter_inputs, row_outputs)
        prev_btn.click(lambda i, d, f, o, v: nav(-1, i, d, f, o, v),
                       [idx_state] + filter_inputs, row_outputs)
        next_btn.click(lambda i, d, f, o, v: nav(+1, i, d, f, o, v),
                       [idx_state] + filter_inputs, row_outputs)
        random_btn.click(lambda i, d, f, o, v: nav(None, i, d, f, o, v),
                         [idx_state] + filter_inputs, row_outputs)
        refresh_btn.click(lambda i, d, f, o, v: show_row(i, d, f, o, v),
                          [idx_state] + filter_inputs, row_outputs)
        jump_btn.click(lambda j, d, f, o, v: show_row(int(j), d, f, o, v),
                       [jump_input] + filter_inputs, row_outputs)

        overview_btn.click(overview, [], [overview_html, overview_plot])

        demo.load(do_load, [run_dd, path_tb],
                  [load_status, disp_dd, flavor_dd, origin_dd, verdict_dd]
                  + row_outputs)
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=7880)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                allowed_paths=[VIDEO_CACHE_DIR])


if __name__ == "__main__":
    main()
