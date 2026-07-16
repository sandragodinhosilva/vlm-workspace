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

# The CLI previewer's per-sample text renderer — REUSED for the app's "download
# this sample as .txt" so the download is byte-for-byte the /preview-output view
# (single source of truth; the reasoning fields we added live in that module).
try:
    from preview_tool_sft_pipeline import render_sample as _render_sample_txt  # noqa: E402
    _PREVIEW_IMPORT_ERROR = None
except Exception as _e:  # loud, not fatal — download just degrades to a JSON dump
    _render_sample_txt = None
    _PREVIEW_IMPORT_ERROR = repr(_e)

import gradio as gr  # noqa: E402

DATASET_ROOT = os.environ.get(
    "DATASET_ROOT", "/mnt/data/sgsilva/datasets/1806/vobs_tool_sft_4k")
DEFAULT_JSONL = os.environ.get(
    "DEFAULT_JSONL",
    # smoke_final_0715 = the canonical inspection run: all 5 flavours + every
    # 2026-07-15 fix (inline judge, step_metrics, self-describing frames, regen
    # change-ratio, E multi-call prompt). The ↻ Runs button + dropdown still let
    # you pick any other run under DATASET_ROOT.
    "/mnt/data/sgsilva/datasets/1806/vobs_tool_sft_4k/smoke_final_0715/smoke.jsonl")
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
# App Guidance — plain-language glossary of the workflow's vocabulary + the
# live workflow diagram, so someone who doesn't know the pipeline can read the
# app. The diagram is the CANONICAL .mmd from the repo (single source of truth —
# it updates when the pipeline does), rendered client-side by mermaid.js.
# ---------------------------------------------------------------------------
_WORKFLOW_MMD = os.path.join(_VISUAL_OBS_DIR, "workflow_tool_use.mmd")


def _read_workflow_mmd() -> str:
    try:
        with open(_WORKFLOW_MMD) as fh:
            return fh.read()
    except Exception as e:  # loud, not fatal
        return f"%% could not read {_WORKFLOW_MMD}: {e!r}"


def guidance_html() -> str:
    """The App Guidance tab: what every idea in this app means, then the live
    workflow diagram. Read straight from the .mmd so it can't drift."""
    import html as _html
    mmd = _read_workflow_mmd()
    mmd_esc = _html.escape(mmd)

    # Render the diagram inside an <iframe srcdoc> (2026-07-15 fix): Gradio's
    # gr.HTML sanitizes component HTML and STRIPS <script> tags, so an inline
    # mermaid.js loader never runs → the diagram didn't render. An iframe's srcdoc
    # is an isolated document Gradio does NOT sanitize, so the mermaid <script>
    # executes inside it. The mermaid source goes in a <pre class="mermaid"> and we
    # escape it for HTML; the whole srcdoc is then attribute-escaped (&quot; etc.).
    # Zoom controls live INSIDE the iframe (the sandbox isolates it, so parent
    # JS can't reach in): a sticky toolbar of +/−/reset buttons scales the
    # rendered SVG via CSS transform. The scroll container keeps the enlarged
    # diagram pannable instead of clipping it.
    inner_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{margin:0;background:#fff;font-family:Helvetica,Arial,sans-serif}"
        "#bar{position:sticky;top:0;z-index:10;background:#f8fafc;"
        "border-bottom:1px solid #e2e8f0;padding:6px 10px;display:flex;"
        "gap:6px;align-items:center}"
        "#bar button{font-size:15px;font-weight:600;cursor:pointer;"
        "border:1px solid #cbd5e1;border-radius:6px;background:#fff;"
        "padding:2px 10px;min-width:34px}"
        "#bar span{color:#64748b;font-size:12px}"
        # grab/grabbing cursor + no text-selection while dragging to pan
        "#scroll{overflow:auto;padding:8px;cursor:grab;user-select:none}"
        "#scroll.dragging{cursor:grabbing}"
        "#zoom{transform-origin:top left;transition:transform .08s}"
        ".mermaid{padding:8px}"
        # only stop foreignObject from CLIPPING; do NOT force-wrap the label text
        # (word-break/overflow-wrap:anywhere with wrap:true collapsed the label to
        # ~1 char wide → per-character vertical text, 2026-07-16 regression). The
        # .mmd already uses explicit <br/> breaks, so leave label flow alone.
        ".mermaid foreignObject{overflow:visible}"
        ".mermaid .nodeLabel{white-space:nowrap}"
        "</style></head><body>"
        "<div id='bar'><button id='zout'>−</button>"
        "<button id='zin'>+</button><button id='zrst'>reset</button>"
        "<span id='zlbl'>100%</span></div>"
        "<div id='scroll'><div id='zoom'>"
        f"<pre class='mermaid'>{mmd_esc}</pre>"
        "</div></div>"
        "<script type='module'>"
        "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';"
        # htmlLabels renders each label as HTML (honours the .mmd's <br/> breaks) and
        # sizes the box to the text — no clipping. Do NOT set wrap:true here: it makes
        # mermaid recompute label width and, with the label CSS, collapsed nodes to
        # 1-char-wide vertical text (2026-07-16 regression). padding gives edge room.
        "mermaid.initialize({startOnLoad:true,securityLevel:'loose',"
        "flowchart:{htmlLabels:true,padding:12}});"
        "let z=1;const zt=document.getElementById('zoom'),"
        "lbl=document.getElementById('zlbl');"
        "function apply(){zt.style.transform='scale('+z+')';"
        "lbl.textContent=Math.round(z*100)+'%';}"
        "document.getElementById('zin').onclick=()=>{z=Math.min(4,z+0.25);apply();};"
        "document.getElementById('zout').onclick=()=>{z=Math.max(0.5,z-0.25);apply();};"
        "document.getElementById('zrst').onclick=()=>{z=1;apply();};"
        # click-and-drag to PAN the (zoomed) diagram: track pointer delta and
        # scroll the container by it (Sandra 2026-07-16). scrollLeft/Top pan the
        # overflow:auto container, so this works at any zoom level.
        "const sc=document.getElementById('scroll');"
        "let drag=false,px=0,py=0,sl=0,st=0;"
        "sc.addEventListener('pointerdown',e=>{drag=true;px=e.clientX;py=e.clientY;"
        "sl=sc.scrollLeft;st=sc.scrollTop;sc.classList.add('dragging');"
        "sc.setPointerCapture(e.pointerId);});"
        "sc.addEventListener('pointermove',e=>{if(!drag)return;"
        "sc.scrollLeft=sl-(e.clientX-px);sc.scrollTop=st-(e.clientY-py);});"
        "const end=e=>{drag=false;sc.classList.remove('dragging');};"
        "sc.addEventListener('pointerup',end);"
        "sc.addEventListener('pointercancel',end);"
        # ctrl/cmd + wheel zooms toward the cursor; plain wheel scrolls as usual
        "sc.addEventListener('wheel',e=>{if(!(e.ctrlKey||e.metaKey))return;"
        "e.preventDefault();z=Math.min(4,Math.max(0.5,z+(e.deltaY<0?0.15:-0.15)));apply();},"
        "{passive:false});"
        "mermaid.run().then(()=>{const s=document.querySelector('svg');"
        "if(s){s.style.maxWidth='none';}apply();});"
        "</script></body></html>"
    )
    srcdoc = _html.escape(inner_doc, quote=True)
    iframe = (f"<iframe srcdoc=\"{srcdoc}\" "
              "style='width:100%;height:1600px;border:1px solid #e2e8f0;"
              "border-radius:8px;background:#fff' "
              "sandbox='allow-scripts'></iframe>")

    def row(term, meaning):
        return (f"<tr><td style='padding:6px 12px;vertical-align:top;white-space:nowrap;"
                f"font-weight:600;color:#1e1b4b'>{term}</td>"
                f"<td style='padding:6px 12px;color:#334155'>{meaning}</td></tr>")

    def bullets(intro, items):
        """A cell rendered as a short intro + a bulleted list, so dense
        multi-item fields (drop reasons, judge tags) read line-by-line instead
        of as a run-on paragraph (Sandra 2026-07-16)."""
        lis = "".join(f"<li style='margin:2px 0'>{it}</li>" for it in items)
        head = f"{intro}<br>" if intro else ""
        return (f"{head}<ul style='margin:4px 0 0;padding-left:18px;"
                f"list-style:disc'>{lis}</ul>")

    flavors = "".join(row(t, m) for t, m in [
        ("A · zero-call", "The model grades from the video ALONE and never calls the tool. Harvested from a free-choice pool — kept only when the teacher <i>naturally</i> chose not to call. The most common 'normal' behaviour."),
        ("B · one call, many Q", "One tool call that batches several questions at once. The everyday tool-use shape."),
        ("C · spot wrong answer", "The tool is deliberately fed a plausible-but-WRONG answer; a good C trace NOTICES it, distrusts it, and grades correctly anyway. Teaches skepticism of the tool."),
        ("D · one call, one Q", "A single call asking the single most useful question. The minimal tool use."),
        ("E · several calls", "Ask, read the answer, then ask again in light of it — genuine iterative querying. <b>RARE by design (~5% of the mix)</b>: multi-call is a situational, 'the model is genuinely confused' behaviour, not a habit. If the final model never multi-calls, that's fine."),
    ])

    stages = "".join(row(t, m) for t, m in [
        ("① Generation", "The 397B teacher writes the reasoning trace, best-of-K tries (K=16), stopping early when it exactly matches the correct grade (severity-exact, not just presence)."),
        ("② Rewrite — stage-2 GT-align", "If the trace's grade isn't already correct, re-reason it onto the correct grade (with the rep's video attached). A clip can OPT OUT instead of laundering — A emits <code>[CANNOT_GROUND_GT]</code>, B/C/D/E emit <code>[CANNOT_RECONCILE_GT]</code> → dropped with a distinct sentinel rather than fabricating cues. Also condenses a rambling &lt;think&gt;. (Distinct from the STAGE-4 repair that runs after the judge — see regen.)"),
        ("③ Judging (inline) — 3-judge cascade", bullets(
            "The Gate-3 judge is a CASCADE of three specialists (2026-07-16), each its own axis:", [
            "<b>J2 format/coherence</b> runs first (cheapest) — a malformed clip is dropped before the rest.",
            "<b>J1 grounding/laundering</b> — is every grade EARNED from a named video cue, or reverse-engineered from the target?",
            "<b>J3 flavour-purpose</b> — does the tool-use match this flavour?",
            "A clip is kept only if ALL THREE pass. On any fail → one shared regen with the failing tags' hints → the whole cascade RE-runs; still failing → EXCLUDED. All in the SAME run.",
            "(Legacy single 15-tag judge still available via <code>--judge-mode single</code>.)",
        ])),
        ("Up to ~7+ calls", "One clip can cost several teacher calls: generate → rewrite → cascade (J2+J1+J3) → on fail regen → re-run cascade, up to the regen budget (<code>--max-regen</code>)."),
    ])

    fields = "".join(row(t, m) for t, m in [
        ("flavor", "Which of A–E behaviours this clip teaches."),
        ("prompt_origin", "<code>forced</code> = generated on this flavour's own prompt. <code>free_choice</code> = an A-pool clip that DID call the tool, re-routed to its observed flavour (B/D/E) — the behaviour is the signal, so the compute isn't wasted."),
        ("drop_reason", bullets(
            "Why a clip was set aside (never silently thrown away — kept for inspection):", [
            "<b>Judge drops:</b> <code>judge:regen_still_failing</code> (a cascade pass kept failing) · <code>judge:regen_error</code> (the rewrite's own post-check rejected the repair — e.g. <code>tool_parts_changed</code>, <code>final_neq_gt</code>) · <code>judge:parse_failed</code> / <code>judge_error</code>.",
            "<b>Opt-out drops</b> (teacher declined to launder): <code>sample_excluded_gt_ungroundable</code> (A) · <code>sample_excluded_gt_unreconcilable</code> (B/C/D/E).",
            "<b>Other:</b> <code>C_no_corrupted_served</code>.",
        ])),
        ("judge_failed_pass", "Which specialist errored / parse-failed (<code>format</code> | <code>grounding</code> | <code>flavor_purpose</code>) when the whole cascade bailed. (<code>judge_mode</code> is always <code>complementary</code> now — the legacy single judge was removed.)"),
        ("judge cascade tags — ALL possible flags (why a clip failed)", bullets(
            "Each specialist has its OWN closed tag set; every flag it can raise:", [
            "<b>J2 format</b> — <code>structured_answer_leak</code> full report block inside &lt;think&gt; · <code>fabricated_tool_exchange</code> narrated a tool call that never happened · <code>too_long</code> waffle-loop think · <code>too_short</code> barely reasons · <code>incoherent</code> final doesn't follow the reasoning · <code>malformed_final</code> wrong section format.",
            "<b>J1 grounding</b> — <code>ungrounded_conclusion</code> score not established from the video · <code>fabricated_detail</code> invented cue · <code>unearned_reversal</code> score flips with no new observation · <code>override_without_cue</code> overrode a tool answer with no named cue · <code>target_restated</code> cue just paraphrases the target · <code>source_leak</code> implies it was handed the answer.",
            "<b>J3 flavour-purpose</b> (per flavour):",
            "&nbsp;&nbsp;• <b>A</b> confident non-use — <code>unexpected_tool_call</code> called the tool (A is zero-call) · <code>fabricated_tool_narration</code> narrated a phantom consult · <code>should_have_asked</code> scored through admitted uncertainty · <code>rule_narration</code> cited the rule as its motive.",
            "&nbsp;&nbsp;• <b>B</b> batching — <code>over_batched</code> queried unneeded points · <code>premature_drafting_turn1</code> drafted the answer in turn-1 · <code>tool_answer_ignored</code> got an answer then scored against it · <code>rule_narration</code> cited the rule.",
            "&nbsp;&nbsp;• <b>C</b> notice+override — <code>silent_endorsement</code> accepted a corrupted answer · <code>unaddressed_corruption</code> never engaged a corrupted cell · <code>false_alarm</code> distrusted a correct answer · <code>blanket_distrust</code> dismissed the whole tool · <code>rule_narration</code> cited the rule.",
            "&nbsp;&nbsp;• <b>D</b> single-Q — <code>filler_single_q</code> asked a non-uncertain question · <code>premature_drafting_turn1</code> drafted in turn-1 · <code>tool_oracle_override</code> reversed a confident read to obey the tool · <code>multi_question_creep</code> asked &gt;1 question · <code>rule_narration</code> cited the rule.",
            "&nbsp;&nbsp;• <b>E</b> iteration — <code>preplanned_split</code> front-planned round-2 in turn-1 · <code>non_reactive_followup</code> follow-up ignores round-1 answers · <code>manufactured_followup</code> fired a needless 2nd round · <code>report_drafting_turn1</code> drafted the assessment in turn-1 · <code>rule_narration</code> cited the rule.",
        ])),
        ("is_perfect vs severity_exact vs f1", bullets(
            "Three signals, increasing strictness:", [
            "<b>is_perfect</b> = the right errors are PRESENT/ABSENT (severity binarized at &gt;1) — a grade of 5 where GT is 2 still counts 'perfect' here.",
            "<b>severity_exact</b> = every severity MAGNITUDE matches GT — the real 'is this grade correct?' signal (2026-07 audit).",
            "<b>full_exact</b> = also matches Effectiveness+Injury (tri-state: True / False / None-when-unknown, never a silent True).",
            "<b>f1</b> can read 0.0 on a clip with NO graded errors while is_perfect is still true — NOT a failure.",
            "➜ Trust <b>severity_exact</b> for correctness; is_perfect overstates it (~1.5× on the 0715 smoke).",
        ])),
        ("natural_severity_exact", "Whether the BEST natural (pre-rewrite) attempt already matched GT magnitudes — the RFT-vs-repair signal. High = the flavour earns GT without rewriting (RFT-lean); low = it needs the repair pass."),
        ("step_metrics", "Per-stage panel on every clip: prompt/output size (chars + tokens), the grade-score AT each stage (should stay perfect after generation), how much the REWRITE and the JUDGE-regen changed the answer (changed_ratio 0→1), wall-time, and #model-calls."),
        ("changed_ratio", "How much a step rewrote the answer: 0.0 = identical, 1.0 = fully replaced. High rewrite changed_ratio = the teacher's first draft needed heavy repair."),
        ("re-route", "An A-pool clip that called the tool becomes a B/D/E clip (not dropped) — see prompt_origin=free_choice."),
        ("regen — stage-4 repair", "The judge-triggered rewrite: when ANY cascade pass fails, a STAGE-4 repair runs — keyed to the failing pass's class (<code>format</code> keeps grade+tool turns identical · <code>grounding</code> re-derives the flagged score from a named cue · <code>workflow</code> restructures the tool rounds), then the WHOLE cascade re-checks it. Distinct from the stage-2 GT-align rewrite (②). Bounded by <code>--max-regen</code>."),
    ])

    tbl = ("style='border-collapse:collapse;width:100%;font-size:14px;"
           "border:1px solid #e2e8f0;border-radius:8px;overflow:hidden'")
    section = lambda title, body: (
        f"<h3 style='color:#1e1b4b;margin:18px 0 8px'>{title}</h3>"
        f"<table {tbl}>{body}</table>")

    return f"""
<div style="max-width:1100px;line-height:1.5">
  <p style="color:#475569;font-size:15px">
    This app inspects the <b>VObs-tool-SFT pipeline</b> — how a VLM is taught
    <b>when to consult a visual-observation tool</b> while grading physiotherapy videos.
    Below: what every term in this app means, then the live workflow diagram.
  </p>
  {section("The five flavours (tool-use behaviours)", flavors)}
  {section("The three stages (one run, up to 5 teacher calls)", stages)}
  {section("Fields &amp; metrics you'll see on each clip", fields)}

  <h3 style="color:#1e1b4b;margin:22px 0 8px">The full workflow</h3>
  <p style="color:#64748b;font-size:13px;margin:0 0 8px">
    Source of truth: <code>visual_obs/workflow_tool_use.mmd</code> — rendered live, so it
    stays current as the pipeline changes.</p>
  {iframe}
  <details style="margin-top:8px">
    <summary style="cursor:pointer;color:#64748b;font-size:13px">diagram source (mermaid)</summary>
    <pre style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;
                font-size:12px;overflow:auto">{mmd_esc}</pre>
  </details>
</div>
"""


# ---------------------------------------------------------------------------
# Filters — choices derived from the data, never hardcoded
# ---------------------------------------------------------------------------

def _choices(field: str) -> List[str]:
    vals = sorted({str(r.get(field)) for r in STATE["rows"] if r.get(field) is not None})
    return [ALL] + vals


def disposition_choices() -> List[str]:
    # judge-excluded = dropped specifically by the Gate-3 judge (drop_reason
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
    # font-size is driven by the --pre-fs CSS var (set live by the "Prompt text
    # size" slider); falls back to 12px if the var is absent.
    return (f"<pre style='white-space:pre-wrap;word-break:break-word;"
            f"background:var(--background-fill-secondary);padding:10px;"
            f"border-radius:6px;max-height:520px;overflow:auto;"
            f"font-size:var(--pre-fs,12px)'>"
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
        # severity_exact is the magnitude-truth (2026-07 audit); show it when the
        # score dict carries it — a presence-perfect but severity-wrong grade is
        # the exact case the old is_perfect-only display hid.
        sev = v.get("is_severity_exact")
        if ok:
            note = " <small>(no GT errors)</small>" if f1 == 0.0 else ""
            if sev is False:
                return (f"⚠️ <b>presence-ok, severity WRONG</b> "
                        f"<small>f1={f1s} (magnitude ≠ GT)</small>{note}")
            sev_tag = " <small>· severity_exact ✓</small>" if sev else ""
            return f"✅ perfect <small>f1={f1s}</small>{sev_tag}{note}"
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
        # The teacher's own <think> for the rewrite — 397B is in thinking mode, so
        # this is the natural reasoning BEFORE the [REWRITTEN REASONING] output.
        # Captured separately by the producer (rewrite_reasoning); shown open so
        # it's visible (it's the thing that was previously invisible).
        rr = r.get("rewrite_reasoning")
        think_block = _details(
            "② REWRITE — teacher &lt;think&gt; (natural reasoning before the rewrite)",
            _pre(rr, empty="<no separate reasoning captured — server ran non-thinking, "
                 "or think was inline in the raw response below>"),
            open_=True) if "rewrite_reasoning" in r else ""
        out.append(
            "<h4>② Rewrite — GT-align pass (the rep VIDEO is attached in the real call)</h4>"
            f"<div style='font-size:13px;margin-bottom:4px'>{meta}</div>"
            + think_block
            + _details("② REWRITE PROMPT (verbatim, incl. embedded transcript)",
                       _pre(r.get("rewrite_prompt")))
            + _details("② REWRITE RAW RESPONSE (parsed content — post-think)",
                       _pre(r.get("rewrite_raw_response"))))
    elif "rewrite_applied" in r:
        out.append("<h4>② Rewrite</h4><div style='opacity:.7'>no rewrite — natural "
                   f"generation already on GT (rewrite_applied={r.get('rewrite_applied')})</div>")

    # ③ JUDGE (+ regen) — one block per judge_attempts entry, depth is data-driven.
    # TWO attempt shapes (--judge-mode, 2026-07-16):
    #   single       — each attempt carries a FLAT judge_prompt/judge_raw_response/
    #                   judge_verdict (the legacy 15-tag flavour judge).
    #   complementary — a CASCADE attempt carries `cascade: [{judge_key, pass, tags,
    #                   notes, judge_prompt, judge_raw_response, judge_reasoning}, ...]`
    #                   (one entry per specialist J2 format / J1 grounding / J3
    #                   flavour-purpose); a REGEN attempt carries regen_prompt/
    #                   regen_union_tags. Render EACH of the (up to 3) passes so the
    #                   app shows all three judges (Sandra 2026-07-16), and don't fall
    #                   back to the flat fields (which are absent → "not present").
    jatt = r.get("judge_attempts") or []
    _JKEY_LABEL = {"format": "J2 FORMAT/COHERENCE", "grounding": "J1 GROUNDING/LAUNDERING",
                   "flavor_purpose": "J3 FLAVOUR-PURPOSE"}
    if jatt:
        _mode = r.get("judge_mode", "single")
        _fp = r.get("judge_failed_pass")
        header = (f"<div style='font-size:13px;margin-bottom:4px'>mode: "
                  f"<b>{_esc(_mode)}</b> · final verdict: "
                  f"<b>{_esc(r.get('judge_verdict_kind'))}</b> · tags={_esc(r.get('judge_tags'))} · "
                  f"accepted_after_regen={r.get('judge_accepted_after_regen')}"
                  + (f" · failed_pass=<b>{_esc(_fp)}</b>" if _fp else "")
                  + f" · notes={_esc(r.get('judge_notes'))}</div>")
        parts = [header]
        for a in jatt:
            n = a.get("attempt")
            # --- REGEN entry (shared between modes) ---
            if a.get("regen_prompt") or a.get("regen_union_tags") is not None:
                ut = a.get("regen_union_tags")
                rr = a.get("regen_reason")
                tag = (f" → union_tags={ut}" if ut is not None else "")
                tag += (f" → <b>{_esc(rr)}</b>" if rr else "")
                parts.append(_details(
                    f"③ attempt {n} — REGEN PROMPT (rewrite re-run with the judge's "
                    f"correction hint){_esc(tag)}",
                    _pre(a.get("regen_prompt"), empty="<regen failed before a prompt was built>")))
                if "regen_reasoning" in a:
                    parts.append(_details(
                        f"③ attempt {n} — REGEN teacher &lt;think&gt; (natural reasoning)",
                        _pre(a.get("regen_reasoning"),
                             empty="<no separate reasoning captured>"), open_=True))
                parts.append(_details(f"③ attempt {n} — REGEN RAW RESPONSE (post-think)",
                                      _pre(a.get("regen_raw_response"))))
                continue
            # --- CASCADE entry (complementary mode): render each specialist pass ---
            if "cascade" in a:
                passes = a.get("cascade") or []
                shown = ", ".join(p.get("judge_key", "?") for p in passes)
                parts.append(f"<div style='font-size:12px;opacity:.75;margin:8px 0 4px'>"
                             f"③ attempt {n} — cascade ran: <b>{_esc(shown)}</b> "
                             f"(J2 first; a J2 fail short-circuits J1/J3)</div>")
                # per-judge SUB-PANEL: each specialist gets its own colored box so
                # J2 / J1 / J3 read as distinct sub-groups, not one flat list
                # (Sandra 2026-07-16). bg tints + accent border per judge_key.
                _JCOLOR = {  # (accent border, bg tint, chip text)
                    "format":        ("#d97706", "#fffbeb", "#92400e"),  # amber  J2
                    "grounding":     ("#2563eb", "#eff6ff", "#1e3a8a"),  # blue   J1
                    "flavor_purpose":("#7c3aed", "#f5f3ff", "#5b21b6"),  # purple J3
                }
                for p in passes:
                    key = p.get("judge_key")
                    lbl = _JKEY_LABEL.get(key, key or "?")
                    acc, bg, chip = _JCOLOR.get(key, ("#94a3b8", "#f8fafc", "#334155"))
                    kind = p.get("kind")
                    if kind == "ok":
                        ok = p.get("pass")
                        badge = ("✓ PASS" if ok else "✗ FAIL")
                        badge_bg = "#dcfce7" if ok else "#fee2e2"
                        badge_fg = "#166534" if ok else "#991b1b"
                        tagline = f" tags={p.get('tags')}" if p.get("tags") else ""
                    else:  # judge_error / parse_failed on this pass
                        badge = f"⚠ {kind}"; badge_bg = "#fef3c7"; badge_fg = "#92400e"
                        tagline = f" {p.get('error') or p.get('reason')}"
                    inner = []
                    inner.append(_details(
                        f"{lbl} PROMPT (verbatim)", _pre(p.get("judge_prompt"))))
                    if "judge_reasoning" in p:
                        inner.append(_details(
                            f"{lbl} teacher &lt;think&gt;",
                            _pre(p.get("judge_reasoning"),
                                 empty="<no separate reasoning captured>"), open_=True))
                    inner.append(_details(
                        f"{lbl} RAW RESPONSE (verdict — post-think)",
                        _pre(p.get("judge_raw_response"))))
                    if p.get("notes"):
                        inner.append(f"<div style='font-size:12px;opacity:.85;"
                                     f"margin:4px 0 2px'>notes: {_esc(p.get('notes'))}</div>")
                    header = (f"<div style='font-weight:700;font-size:13px;color:{chip};"
                              f"margin:0 0 4px;display:flex;align-items:center;gap:8px'>"
                              f"<span>{_esc(lbl)}</span>"
                              f"<span style='background:{badge_bg};color:{badge_fg};"
                              f"border-radius:10px;padding:1px 8px;font-size:11px'>"
                              f"{badge}</span>"
                              f"<span style='font-weight:400;opacity:.7;font-size:11px'>"
                              f"{_esc(tagline)}</span></div>")
                    parts.append(
                        f"<div style='border:1px solid {acc}33;border-left:3px solid {acc};"
                        f"background:{bg};border-radius:8px;padding:8px 10px;margin:0 0 8px'>"
                        f"{header}{''.join(inner)}</div>")
                continue
            # --- FLAT single-judge entry (legacy mode) ---
            v = a.get("judge_verdict") or {}
            vline = (f" → pass={v.get('pass')} tags={v.get('tags')}"
                     if v else " → <no parsed verdict>")
            parts.append(_details(f"③ attempt {n} — JUDGE PROMPT (verbatim){_esc(vline)}",
                                  _pre(a.get("judge_prompt"))))
            if "judge_reasoning" in a:
                parts.append(_details(
                    f"③ attempt {n} — JUDGE teacher &lt;think&gt; (natural reasoning)",
                    _pre(a.get("judge_reasoning"),
                         empty="<no separate reasoning captured>"), open_=True))
            parts.append(_details(f"③ attempt {n} — JUDGE RAW RESPONSE (verdict — post-think)",
                                  _pre(a.get("judge_raw_response"))))
        _title = "③ Gate-3 judge — 3-specialist cascade (+ stage-4 repair)"
        out.append(f"<h4>{_title}</h4>" + "".join(parts))
    else:
        out.append("<h4>③ Gate-3 judge</h4><div style='opacity:.7'>not run for this row "
                   "(--no-judge run, or dropped before the judge)</div>")

    if not out:
        return "<div style='opacity:.7'>no pipeline-trail fields on this row</div>"
    # Wrap each stage (each `out` entry begins with its own <h4>) in a spaced,
    # bordered card so the three stages read as distinct blocks instead of running
    # together (Sandra 2026-07-16). A left accent bar colour-codes the stage.
    _ACCENT = {"①": "#16a34a", "②": "#2563eb", "③": "#ea580c"}
    cards = []
    for blk in out:
        mark = next((m for m in _ACCENT if m in blk[:8]), None)
        bar = _ACCENT.get(mark, "#94a3b8")
        cards.append(
            f"<section style='margin:0 0 18px;padding:10px 14px;"
            f"border:1px solid #e2e8f0;border-left:4px solid {bar};"
            f"border-radius:8px;background:#fff'>{blk}</section>")
    return "".join(cards)


def render_final_messages(r: Dict) -> str:
    # A DROPPED row ships NOTHING — SFT trains on zero rows for it, so the
    # "final shipped messages" panel must NOT present its (failed) last trace as
    # if it trains (Sandra 2026-07-16). The `messages` field still carries the
    # last attempt for INSPECTION on the trail above; here we show the shipped
    # state, which for a drop is empty. `drop_reason` set == not shipped.
    dropped = r.get("drop_reason") not in (None, "")
    if dropped:
        return ("<div style='color:#b91c1c;font-weight:700;font-size:14px'>"
                "⛔ NOT shipped — this clip was DROPPED "
                f"(<code>{_esc(r.get('drop_reason'))}</code>), so SFT trains on "
                "NOTHING from it.</div>"
                "<div style='opacity:.75;font-size:13px;margin-top:4px'>The failed "
                "trace is still visible in the pipeline trail above (kept for "
                "inspection — paid compute is never discarded), but it is NOT part "
                "of the training set.</div>")
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
    "rewrite_reasoning", "rewrite_applied", "rewrite_kind", "rewrite_failed_reason",
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


def download_sample_txt(abs_idx: int, disposition, flavor, origin, verdict):
    """Write the CURRENTLY-DISPLAYED row to a well-formatted .txt (same layout as
    /preview-output) and return the path for gr.File to serve. Reuses the CLI
    previewer's render_sample — the single source of truth — so the download and
    the terminal preview never drift. Falls back to a pretty JSON dump if that
    module couldn't import."""
    rows = STATE["rows"]
    if not rows:
        return None
    sel = filtered(disposition, flavor, origin, verdict)
    if abs_idx not in sel and sel:
        abs_idx = sel[0]
    abs_idx = max(0, min(abs_idx, len(rows) - 1))
    r = rows[abs_idx]

    lines: List[str] = []
    w = lines.append
    if _render_sample_txt is not None:
        try:
            _render_sample_txt(w, r)
        except Exception as e:  # never a blank download — surface + dump
            w(f"[render_sample failed: {e!r} — full JSON below]")
            w(json.dumps(r, indent=2, default=str, ensure_ascii=False))
    else:
        w(f"[preview_tool_sft_pipeline import failed: {_PREVIEW_IMPORT_ERROR} — "
          "raw JSON dump]")
        w(json.dumps(r, indent=2, default=str, ensure_ascii=False))
    # +disposition/drop info at the very top so a dropped sample is self-labelling.
    disp = r.get("_disposition")
    head = (f"# disposition={disp}  drop_reason={r.get('drop_reason')}  "
            f"judge_verdict_kind={r.get('judge_verdict_kind')}\n")

    os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)
    stem = f"{r.get('flavor')}_{r.get('session_id')}_{r.get('rep_index')}_{disp}"
    stem = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(stem))
    out = Path(VIDEO_CACHE_DIR) / f"sample_{stem}.txt"
    out.write_text(head + "\n".join(lines), encoding="utf-8")
    return str(out)


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
            # Re-scan the dataset root for NEW runs produced after the app started
            # (e.g. a fresh smoke) — without this the dropdown is frozen at launch
            # time, so a just-produced run wouldn't be selectable until restart.
            rescan_btn = gr.Button("↻ Runs", scale=1)
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

                # Prompt text size — sets the --pre-fs CSS var live (client-side,
                # no server round-trip) so every prompt/output <pre> scales.
                with gr.Row():
                    pre_fs = gr.Slider(8, 28, value=12, step=1,
                                       label="Prompt text size (px)", scale=3)
                    dl_btn = gr.Button("⬇ Download this sample (.txt)", scale=1)
                pre_fs.change(
                    None, pre_fs, None,
                    js="(v)=>{document.documentElement.style.setProperty("
                       "'--pre-fs', v+'px'); return [];}")
                # Full formatted dump of the displayed sample (same layout as
                # /preview-output). Appears when the button is clicked.
                dl_file = gr.File(label="sample .txt (all data for this one sample)",
                                  visible=True)

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
                with gr.Accordion("All row fields not rendered above",
                                  open=False):
                    other_html = gr.HTML()

            with gr.Tab("Run overview — how good is the pipeline?"):
                overview_btn = gr.Button("Compute overview (kept + dropped)",
                                         variant="primary")
                overview_html = gr.HTML()
                overview_plot = gr.Plot(label="change-ratio distributions per flavor")

            with gr.Tab("App Guidance"):
                # Plain-language glossary of every idea in this app + the live
                # workflow diagram (read from the canonical .mmd, so it can't drift).
                gr.HTML(guidance_html())

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

        # Re-discover runs (newest first) and repopulate the dropdown, selecting the
        # newest so a fresh smoke is one click away. Does NOT load — user hits Load.
        def do_rescan():
            runs = discover_runs()
            return gr.update(choices=runs, value=(runs[0] if runs else None))
        rescan_btn.click(do_rescan, [], [run_dd])

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

        dl_btn.click(download_sample_txt,
                     [idx_state] + filter_inputs, [dl_file])

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
