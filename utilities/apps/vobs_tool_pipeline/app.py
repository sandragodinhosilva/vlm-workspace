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
    # smoke_stage4_0716c = the latest fixed-code cascade-judge run (3-specialist
    # J2/J1/J3 + stage-4 repair). This run name moves as new smokes supersede it
    # (0716b was renamed _pre_efix mid-session) — the ↻ Runs button + dropdown
    # always let you pick the actual newest run under DATASET_ROOT if this
    # default has gone stale.
    "/mnt/data/sgsilva/datasets/1806/vobs_tool_sft_4k/smoke_stage4_0716c/smoke.jsonl")
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


# Substrings that mark a run dir as SUPERSEDED / pre-fix / contaminated — hidden
# from the dropdown so a team-shared app only ever offers clean, current runs
# (Sandra 2026-07-17). The free-text path box still loads ANY path for debugging.
_SUPERSEDED_MARKERS = ("_pre_", "_pre.", "_prefix", "preaudit", "contaminated",
                       "superseded", "_buggy", "_archive", "_old")
# How many clean runs the dropdown offers (newest first).
_MAX_VISIBLE_RUNS = 3


def _is_superseded_run(path: Path) -> bool:
    """True if the run dir name carries a pre-fix / contaminated / archived marker."""
    name = path.parent.name.lower()
    return any(m in name for m in _SUPERSEDED_MARKERS)


def discover_runs() -> List[str]:
    """The newest CLEAN kept-rows JSONLs under DATASET_ROOT (newest first, capped at
    _MAX_VISIBLE_RUNS). ckpt/dropped sidecars AND superseded/pre-fix runs are excluded
    so a team-shared dropdown never offers a stale run. The free-text path box still
    loads anything outside this list for debugging."""
    root = Path(DATASET_ROOT)
    if not root.is_dir():
        return []
    hits = [q for q in root.glob("*/*.jsonl")
            if not q.name.endswith((".ckpt.jsonl", ".dropped.jsonl"))
            and not _is_superseded_run(q)]
    hits.sort(key=lambda q: q.stat().st_mtime, reverse=True)
    return [str(q) for q in hits[:_MAX_VISIBLE_RUNS]]


# ---------------------------------------------------------------------------
# Cross-tab GLOSSARY LINKS (Sandra 2026-07-17): a term shown in a panel can link
# to its definition in the App Guidance tab. `_gloss_slug` maps a term to a stable
# anchor id; guidance rows carry `id="gloss-<slug>"`; `_gloss(label, term)` wraps a
# term in a panel as a clickable link. A tiny JS handler (installed via the Blocks
# `js=` hook, which survives Gradio's <script> stripping) intercepts the click,
# switches to the guidance tab, scrolls to the anchor, and flashes it.
_GLOSS_SLUGS = {
    # canonical term -> slug (used for BOTH the guidance anchor and panel links)
    "all errors identified": "all-errors-identified",
    "severity-exact": "all-errors-identified",   # same guidance row
    "fully-correct": "all-errors-identified",
    "judge-excluded": "drop_reason",
    "workflow": "drop_reason",
    "drop_reason": "drop_reason",
    "step_metrics": "step_metrics",
    "changed_ratio": "changed_ratio",
    "regen": "regen",
    "re-route": "re-route",
    "prompt_origin": "prompt_origin",
    "flavor": "flavor",
    "natural_severity_exact": "natural_severity_exact",
    "judge_failed_pass": "judge_failed_pass",
    "best-of-k": "how-best-of-k-picks-the-winner",
    # full guidance-row titles → short slugs, so both the row anchor and the panel
    # links resolve to the same id.
    "all errors identified vs severity-exact vs f1": "all-errors-identified",
    "regen — stage-4 repair": "regen",
}


def _gloss_slug(term: str) -> str:
    """Stable anchor slug for a guidance term (lowercased, non-alnum → hyphen). The
    result is ALWAYS hyphen-only (no underscores) so both the anchor id and the link
    target agree — a mapped alias is itself re-normalized."""
    import re as _re
    key = term.strip().lower()
    raw = _GLOSS_SLUGS.get(key, key)
    return _re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def _gloss(label: str, term: str = None) -> str:
    """Wrap `label` as a clickable link to its App-Guidance definition. `term` (if
    given) resolves the slug; else `label` does. Renders a plain <a> with a data
    attribute — the Blocks js handler does the cross-tab navigation on click."""
    slug = _gloss_slug(term or label)
    return (f"<a href='#gloss-{slug}' class='gloss' data-gloss='{slug}' "
            f"style='color:#7c3aed;text-decoration:none;border-bottom:1px dotted #7c3aed;"
            f"cursor:pointer' title='See the definition in App Guidance'>{label}</a>")


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
        # TEXT-OVERFLOW + READABILITY FIX (2026-07-16, definitive). We use htmlLabels
        # (so the .mmd's <b>/<i>/&lt; markup renders — SVG-text mode showed them as
        # LITERAL '<i>' / '&lt;think&gt;', the unreadable state in Sandra's 2nd
        # screenshot). The overflow the FIRST screenshot showed is cured by pinning the
        # label to a FIXED width and letting it WRAP: mermaid then measures the box at
        # that same width, so text and box agree and nothing spills. The historical
        # 1-char-vertical collapse came from adding word-break/overflow-wrap:break-word
        # on TOP of wrap — so we set ONLY white-space:normal + a width, never word-break.
        ".mermaid .nodeLabel{white-space:normal !important;"
        "display:inline-block;max-width:260px;line-height:1.35}"
        ".mermaid foreignObject{overflow:visible}"
        "</style></head><body>"
        "<div id='bar'><button id='zout'>−</button>"
        "<button id='zin'>+</button><button id='zrst'>reset</button>"
        "<span id='zlbl'>100%</span></div>"
        "<div id='scroll'><div id='zoom'>"
        f"<pre class='mermaid'>{mmd_esc}</pre>"
        "</div></div>"
        "<script type='module'>"
        "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';"
        # htmlLabels:true + wrap:true (2026-07-16): htmlLabels renders the .mmd's
        # <b>/<i>/&lt; markup properly (SVG-text mode showed them literally — unreadable),
        # and wrap:true + the FIXED .nodeLabel max-width (CSS above) makes mermaid measure
        # each box at the wrapped width, so text can't overflow. This does NOT trigger the
        # old 1-char collapse — that needed word-break:break-word too, which we do NOT set.
        "mermaid.initialize({startOnLoad:true,securityLevel:'loose',"
        "flowchart:{htmlLabels:true,wrap:true,padding:12}});"
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
        # PLAIN mouse-wheel zooms (Sandra 2026-07-16) — scroll up = zoom in, down =
        # zoom out. Drag-to-pan (above) covers navigation, so the wheel is free to
        # zoom without a modifier. ctrl/cmd+wheel still zooms (same path). preventDefault
        # stops the iframe from also page-scrolling.
        "sc.addEventListener('wheel',e=>{"
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
        # Anchor id so a panel link (_gloss) can scroll here. Slug from the term text
        # with HTML tags + leading markers (e.g. 'A · ', '① ') stripped.
        import re as _re
        plain = _re.sub(r"<[^>]+>", "", term)
        plain = _re.sub(r"^[A-E]\s*·\s*|^[①②③④⑤]\s*", "", plain)
        slug = _gloss_slug(plain)
        return (f"<tr id='gloss-{slug}' style='scroll-margin-top:60px'>"
                f"<td style='padding:6px 12px;width:150px;max-width:150px;"
                f"vertical-align:top;font-weight:600;color:#1e1b4b'>{term}</td>"
                f"<td style='padding:6px 12px;color:#334155'>{meaning}</td></tr>")

    def bullets(intro, items):
        """A cell rendered as a short intro + a bulleted list, so dense
        multi-item fields (drop reasons, judge tags) read line-by-line instead
        of as a run-on paragraph (Sandra 2026-07-16). Each item may be a plain
        string (level-1 bullet) or a (label, sub_items) tuple — sub_items then
        render as a nested level-2 <ul>, and any of ITS entries may in turn be
        a (label, sub_sub_items) tuple for a level-3 <ul> (2026-07-16, up to 3
        levels for e.g. judge specialist -> flavor -> tag list)."""
        def render_items(items, level):
            lis = []
            for it in items:
                if isinstance(it, tuple):
                    label, sub = it
                    lis.append(f"<li style='margin:{3 if level==1 else 2}px 0'>{label}"
                               + render_items(sub, level + 1) + "</li>")
                else:
                    lis.append(f"<li style='margin:{3 if level==1 else 2}px 0'>{it}</li>")
            style = ("disc" if level == 1 else "circle" if level == 2 else "square")
            return (f"<ul style='margin:{4 if level==1 else 2}px 0 0;"
                    f"padding-left:{18 if level==1 else 16}px;list-style:{style}'>"
                    + "".join(lis) + "</ul>")
        head = f"{intro}" if intro else ""
        return head + render_items(items, 1)

    flavors = "".join(row(t, m) for t, m in [
        ("A · zero-call", "The model grades from the video ALONE and never calls the tool. Harvested from a free-choice pool — kept only when the teacher <i>naturally</i> chose not to call. The most common 'normal' behaviour."),
        ("B · one call, many Q", "One tool call that batches several questions at once. The everyday tool-use shape."),
        ("C · spot wrong answer", "The tool is deliberately fed a plausible-but-WRONG answer; a good C trace NOTICES it, distrusts it, and grades correctly anyway. Teaches skepticism of the tool."),
        ("D · one call, one Q", "A single call asking the single most useful question. The minimal tool use."),
        ("E · several calls", "Ask, read the answer, then ask again in light of it — genuine iterative querying. <b>RARE by design (~5% of the mix)</b>: multi-call is a situational, 'the model is genuinely confused' behaviour, not a habit. If the final model never multi-calls, that's fine."),
    ])

    stages = "".join(row(t, m) for t, m in [
        ("① Generation", "The 397B teacher writes the reasoning trace, best-of-K tries (K=16), stopping early when it exactly matches the correct grade (severity-exact, not just presence)."),
        ("② Rewrite — stage-2 GT-align", "If the trace's grade isn't already correct, the teacher <b>EDITS the existing &lt;think&gt; reasoning</b> (minimally) so it honestly leads to the correct grade — with the rep's video attached. <b>Reasoning-only contract (2026-07-16):</b> the teacher NEVER writes the final answer; the pipeline <b>composes the GT-correct final</b> (errors + scores from GT, the model's own movement-analysis &amp; feedback prose kept) and <b>appends it verbatim</b>, so the grade CANNOT drift. A clip can OPT OUT instead of laundering — A emits <code>[CANNOT_GROUND_GT]</code>, B/C/D/E emit <code>[CANNOT_RECONCILE_GT]</code> → dropped with a distinct sentinel rather than fabricating cues. Also condenses a rambling &lt;think&gt; (final kept byte-identical by construction). (Distinct from the STAGE-4 repair that runs after the judge — see regen.)"),
        ("③ Judging (inline) — 3-judge cascade", bullets(
            "The Gate-3 judge is a CASCADE of three specialists (2026-07-16), each its own axis:", [
            "<b>J2 format/coherence</b> runs first (cheapest) — a malformed clip is dropped before the rest.",
            "<b>J1 grounding/laundering</b> — is every grade EARNED from a named video cue, or reverse-engineered from the target?",
            "<b>J3 flavour-purpose</b> — does the tool-use match this flavour?",
            "A clip is kept only if ALL THREE pass. On any fail → one STAGE-4 repair keyed to the failing class → the whole cascade RE-runs; still failing → EXCLUDED. All in the SAME run.",
            "(The cascade is the ONLY judge topology since 2026-07-16 — the legacy single 15-tag judge was removed.)",
        ])),
        ("④ Stage-4 repair — keyed to the failing judge CLASS", bullets(
            "On a cascade fail, ONE repair runs, routed to the highest-priority failing class (grounding &gt; workflow &gt; format), then the cascade re-runs. <b>Reasoning-only (2026-07-16):</b> every class EDITS the &lt;think&gt; only — the final answer is GIVEN and appended verbatim, so the grade is untouched by construction (no class can move it):", [
            "<b>format</b> (J2) — edit the reasoning's presentation only; grade + tool turns byte-IDENTICAL.",
            "<b>grounding</b> (J1) — edit the reasoning so the flagged score is earned from a NAMED video cue; the grade can't move (final is given); opt-out if it can't be reconciled with the video.",
            "<b>workflow</b> (J3) — edit the reasoning + restructure the tool ROUNDS (the ONLY repair allowed to; sees the FULL per-round reasoning; tool ANSWERS stay a subset — no fabrication). The grade is untouched (final is given &amp; appended).",
        ])),
        ("Up to ~7+ calls", "One clip can cost several teacher calls: generate → rewrite → cascade (J2+J1+J3) → on fail stage-4 repair → re-run cascade, up to the regen budget (<code>--max-regen</code>)."),
    ])

    fields = "".join(row(t, m) for t, m in [
        ("flavor", "Which of A–E behaviours this clip teaches."),
        ("prompt_origin", "<code>forced</code> = generated on this flavour's own prompt. <code>free_choice</code> = an A-pool clip that DID call the tool, re-routed to its observed flavour (B/D/E) — the behaviour is the signal, so the compute isn't wasted."),
        ("drop_reason", bullets(
            "Why a clip was set aside (never silently thrown away — kept for inspection). "
            "These fall in two mutually-exclusive buckets the tab-1 filter + tab-2 table use:", [
            ("<b>Judge-excluded</b> (<code>judge:*</code>) — built + repaired, but the re-judge "
             "after the stage-4 rewrite failed AGAIN", [
                "<code>judge:regen_still_failing</code> — a cascade pass kept failing after the repair",
                "<code>judge:regen_error</code> — the rewrite's own STRUCTURAL post-check rejected "
                "the repair (2026-07-16 reasoning-only: the grade can't drift, so there is no "
                "<code>final_neq_gt</code> gate anymore — only structural sentinels remain, e.g. "
                "<code>tool_parts_changed</code> / <code>fabricated_tool_answer</code>, "
                "<code>earlier_turn_stale_answer</code>, <code>rewrite_malformed_output</code> "
                "(trivially-short reasoning), <code>given_final_malformed</code>)",
                "<code>judge:parse_failed</code> / <code>judge_error</code>",
            ]),
            ("<b>Workflow drops</b> (non-judge) — never reached a clean verdict", [
                ("<b>Opt-out</b> (teacher declined to launder)", [
                    "<code>sample_excluded_gt_ungroundable</code> (A)",
                    "<code>sample_excluded_gt_unreconcilable</code> (B/C/D/E)",
                ]),
                "<code>C_no_corrupted_served</code> — no corrupted answer was served for a C rep",
            ]),
        ])),
        ("judge_failed_pass", "Which specialist errored / parse-failed (<code>format</code> | <code>grounding</code> | <code>flavor_purpose</code>) when the whole cascade bailed. The judge is ALWAYS the 3-specialist complementary cascade now — the legacy single 15-tag judge was removed."),
        ("judge cascade tags — ALL possible flags (why a clip failed)", bullets(
            "Each specialist has its OWN closed tag set; every flag it can raise:", [
            ("<b>J2 format</b>", [
                "<code>structured_answer_leak</code> — full report block inside &lt;think&gt;",
                "<code>fabricated_tool_exchange</code> — narrated a tool call that never happened",
                "<code>too_long</code> — waffle-loop think",
                "<code>too_short</code> — barely reasons",
                "<code>incoherent</code> — final doesn't follow the reasoning",
                "<code>malformed_final</code> — wrong section format",
            ]),
            ("<b>J1 grounding</b>", [
                "<code>ungrounded_conclusion</code> — score not established from the video",
                "<code>fabricated_detail</code> — invented cue",
                "<code>unearned_reversal</code> — score flips with no new observation",
                "<code>override_without_cue</code> — overrode a tool answer with no named cue",
                "<code>target_restated</code> — cue just paraphrases the target",
                "<code>source_leak</code> — implies it was handed the answer",
            ]),
            ("<b>J3 flavour-purpose</b> (per flavour)", [
                ("<b>A</b> confident non-use", [
                    "<code>unexpected_tool_call</code> — called the tool (A is zero-call)",
                    "<code>fabricated_tool_narration</code> — narrated a phantom consult",
                    "<code>should_have_asked</code> — scored through admitted uncertainty",
                    "<code>rule_narration</code> — cited the rule as its motive",
                ]),
                ("<b>B</b> batching", [
                    "<code>over_batched</code> — queried unneeded points",
                    "<code>premature_drafting_turn1</code> — drafted the answer in turn-1",
                    "<code>tool_answer_ignored</code> — got an answer then scored against it",
                    "<code>rule_narration</code> — cited the rule",
                ]),
                ("<b>C</b> notice+override", [
                    "<code>silent_endorsement</code> — accepted a corrupted answer",
                    "<code>unaddressed_corruption</code> — never engaged a corrupted cell",
                    "<code>false_alarm</code> — distrusted a correct answer",
                    "<code>blanket_distrust</code> — dismissed the whole tool",
                    "<code>rule_narration</code> — cited the rule",
                ]),
                ("<b>D</b> single-Q", [
                    "<code>filler_single_q</code> — asked a non-uncertain question",
                    "<code>premature_drafting_turn1</code> — drafted in turn-1",
                    "<code>tool_oracle_override</code> — reversed a confident read to obey the tool",
                    "<code>multi_question_creep</code> — asked &gt;1 question",
                    "<code>rule_narration</code> — cited the rule",
                ]),
                ("<b>E</b> iteration", [
                    "<code>preplanned_split</code> — front-planned round-2 in turn-1",
                    "<code>non_reactive_followup</code> — follow-up ignores round-1 answers",
                    "<code>manufactured_followup</code> — fired a needless 2nd round",
                    "<code>report_drafting_turn1</code> — drafted the assessment in turn-1",
                    "<code>rule_narration</code> — cited the rule",
                ]),
            ]),
        ])),
        ("How best-of-K picks the winner", bullets(
            "The attempt label shows the REAL selection key (not just all-errors-identified, "
            "which ties on every clean rep). The driver keeps the attempt that is MAX on this "
            "cascade — <b>strictest tier dominates</b>. (The label lists them loosest→strictest "
            "for reading; the RANKING below is strictest-first.)", [
            "1. <b>Fully-correct</b> — error set + every severity magnitude + Effectiveness + Injury all == GT. The truly-perfect grade; a K-set with one of these skips the rewrite entirely.",
            "2. <b>Severity-exact</b> — every severity MAGNITUDE matches (eff/injury may be unchecked).",
            "3. <b>All errors identified</b> — the right errors flagged present/absent (the <code>is_perfect</code> field). Ties constantly — this is why all-errors-identified alone can't explain a pick.",
            "4. <b>F1</b> → then the tiebreaks: lower severity-L1 distance, lower eff/injury error, fewer false-positives, fewer tool rounds.",
            "➜ The ⭐ SELECTED line names the FIRST tier where the winner beat the field, so 'why attempt 2 not 4?' is answerable.",
        ])),
        ("All errors identified vs severity-exact vs F1", bullets(
            "Three signals, increasing strictness:", [
            "<b>All errors identified</b> (the <code>is_perfect</code> field) = the right errors are PRESENT/ABSENT (severity binarized at &gt;1) — a grade of 5 where GT is 2 still counts here.",
            "<b>Severity-exact</b> = every severity MAGNITUDE matches GT — the real 'is this grade correct?' signal (2026-07 audit).",
            "<b>Full-exact</b> = also matches Effectiveness+Injury (tri-state: True / False / None-when-unknown, never a silent True).",
            "<b>F1</b> can read 0.0 on a clip with NO graded errors while all-errors-identified is still true — NOT a failure.",
            "➜ Trust <b>severity-exact</b> for correctness; all-errors-identified overstates it (~1.5× on the 0715 smoke).",
        ])),
        ("natural_severity_exact", "Whether the BEST natural (pre-rewrite) attempt already matched GT magnitudes — the RFT-vs-repair signal. High = the flavour earns GT without rewriting (RFT-lean); low = it needs the repair pass."),
        ("step_metrics", "Per-stage panel on every clip: prompt/output size (chars + tokens), the grade-score AT each stage (should stay perfect after generation), how much the REWRITE and the JUDGE-regen changed the answer (changed_ratio 0→1), wall-time, and #model-calls."),
        ("changed_ratio", "How much a step rewrote the answer: 0.0 = identical, 1.0 = fully replaced. High rewrite changed_ratio = the teacher's first draft needed heavy repair."),
        ("re-route", "An A-pool clip that called the tool becomes a B/D/E clip (not dropped) — see prompt_origin=free_choice."),
        ("regen — stage-4 repair", bullets(
            "The judge-triggered rewrite: when ANY cascade pass fails, a STAGE-4 repair runs — "
            "keyed to the failing pass's class — then the WHOLE cascade re-checks it. "
            "<b>Reasoning-only (2026-07-16):</b> every class edits the &lt;think&gt; only — the "
            "final answer is GIVEN and appended verbatim, so the grade never moves. Distinct from "
            "the stage-2 GT-align rewrite (②). Bounded by <code>--max-regen</code>. The three "
            "repair classes:", [
            "<b>format</b> (J2) — edit the reasoning's presentation only; grade + tool turns byte-IDENTICAL.",
            "<b>grounding</b> (J1) — edit the reasoning so the flagged score is earned from a NAMED video cue; the grade can't move (final is given); opt-out if it can't be reconciled with the video.",
            "<b>workflow</b> (J3) — edit the reasoning + restructure the tool ROUNDS (the ONLY repair allowed to; sees the FULL per-round reasoning; tool ANSWERS stay a subset — no fabrication).",
        ])),
    ])

    tbl = ("style='border-collapse:collapse;width:100%;font-size:14px'")
    # Each guidance section is its own CARD (Sandra 2026-07-17): a bordered block
    # with a left accent bar + a tinted header band, so the sections read as
    # distinct areas instead of running together — the same visual separation the
    # stage cards give tab 1. `intro` is an optional one-line subtitle under the
    # header; `body` is the section's <tr> rows (rendered inside the card's table).
    def section(title, body, intro=""):
        sub = (f"<div style='color:#64748b;font-size:12.5px;margin-top:3px'>{intro}</div>"
               if intro else "")
        return (
            "<section style='border:1px solid #e2e8f0;border-left:4px solid #6366f1;"
            "border-radius:10px;overflow:hidden;margin:0 0 18px;"
            "box-shadow:0 1px 2px rgba(0,0,0,.04)'>"
            "<div style='background:#eef2ff;padding:10px 14px;border-bottom:1px solid #e2e8f0'>"
            f"<div style='color:#1e1b4b;font-weight:700;font-size:15px'>{title}</div>{sub}</div>"
            f"<div style='padding:4px 6px'><table {tbl}>{body}</table></div>"
            "</section>")

    return f"""
<div style="max-width:1100px;line-height:1.5">
  <p style="color:#475569;font-size:15px">
    This app inspects the <b>VObs-tool-SFT pipeline</b> — how a VLM is taught
    <b>when to consult a visual-observation tool</b> while grading physiotherapy videos.
    Below: what every term in this app means, then the live workflow diagram.
  </p>
  {section("The five flavours (tool-use behaviours)", flavors,
           "A–E: the five tool-use behaviours a clip can teach.")}
  {section("The three stages (one run, up to 5 teacher calls)", stages,
           "How one clip flows: generate → rewrite → judge cascade → stage-4 repair.")}
  {section("Fields &amp; metrics you'll see on each clip", fields,
           "Every column, flag, and score the row inspector shows.")}
  {section("The full workflow",
           "<tr><td style='padding:6px 12px'>"
           "<p style='color:#64748b;font-size:13px;margin:0 0 8px'>"
           "Source of truth: <code>visual_obs/workflow_tool_use.mmd</code> — rendered live, "
           "so it stays current as the pipeline changes.</p>"
           f"{iframe}"
           "<details style='margin-top:8px'>"
           "<summary style='cursor:pointer;color:#64748b;font-size:13px'>Diagram source (mermaid)</summary>"
           "<pre style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;"
           f"font-size:12px;overflow:auto'>{mmd_esc}</pre></details>"
           "</td></tr>",
           "The live pipeline diagram, straight from the .mmd.")}
</div>
"""


# ---------------------------------------------------------------------------
# Filters — choices derived from the data, never hardcoded
# ---------------------------------------------------------------------------

def _choices(field: str) -> List[str]:
    vals = sorted({str(r.get(field)) for r in STATE["rows"] if r.get(field) is not None})
    return [ALL] + vals


def disposition_choices() -> List[str]:
    # MUTUALLY EXCLUSIVE dispositions (Sandra 2026-07-17): 'dropped' = dropped for a
    # NON-judge reason (timeout, opt-out, exception, rewrite failure); 'judge-excluded'
    # = dropped by the Gate-3 judge (drop_reason `judge:<verdict_kind>`). They partition
    # the dropped set, so 'dropped' + 'judge-excluded' never double-count a row.
    return [ALL, "kept", "dropped", "judge-excluded"]


YES_NO = [ALL, "yes", "no"]


def _specialist_ever_failed(row: Dict, judge_key: str) -> bool:
    """True if judge_key (format/grounding/flavor_purpose) returned pass=False,
    judge_error, or parse_failed on ANY attempt of this row — even if a later
    regen fixed it and the row shipped (Sandra 2026-07-16: filter should surface
    every row a specialist ever caught something on, not just unrecovered drops)."""
    for a in (row.get("judge_attempts") or []):
        for p in (a.get("cascade") or []):
            if p.get("judge_key") != judge_key:
                continue
            if p.get("kind") != "ok" or p.get("pass") is False:
                return True
    return False


def _stage4_regen_fired(row: Dict) -> bool:
    """True if the JUDGE-triggered stage-4 repair ran at least once on this row.
    DISTINCT from rewrite_applied (the pre-judge stage-2 GT-align rewrite) —
    empirically independent (Sandra 2026-07-16: a row can have rewrite_applied
    =False (natural gen already on GT) but still get a stage-4 regen because the
    judge caught something, or rewrite_applied=True with a clean judge pass and
    no stage-4 regen at all). Detected via judge_attempts[*].regen_prompt /
    regen_union_tags — the same fields REGEN entries carry regardless of
    judge_mode (single vs complementary cascade)."""
    for a in (row.get("judge_attempts") or []):
        if a.get("regen_prompt") or a.get("regen_union_tags") is not None:
            return True
    return False


def _matches(row: Dict, disposition: str, flavor: str, origin: str, verdict: str,
            rewrite: str, stage4: str, j1: str, j2: str, j3: str) -> bool:
    _is_judge_excluded = str(row.get("drop_reason") or "").startswith("judge:")
    if disposition == "kept" and row["_disposition"] != "kept":
        return False
    # 'dropped' and 'judge-excluded' are MUTUALLY EXCLUSIVE (Sandra 2026-07-17):
    # 'dropped' now means dropped for a NON-judge reason (timeout, opt-out, exception,
    # rewrite failure), 'judge-excluded' means dropped by the Gate-3 judge (drop_reason
    # starts 'judge:'). Together they partition the dropped set — no row matches both.
    if disposition == "dropped" and (row["_disposition"] != "dropped" or _is_judge_excluded):
        return False
    if disposition == "judge-excluded" and not _is_judge_excluded:
        return False
    if flavor != ALL and str(row.get("flavor")) != flavor:
        return False
    if origin != ALL and str(row.get("prompt_origin")) != origin:
        return False
    if verdict != ALL and str(row.get("judge_verdict_kind")) != verdict:
        return False
    if rewrite != ALL:
        want = (rewrite == "yes")
        if bool(row.get("rewrite_applied")) != want:
            return False
    if stage4 != ALL:
        want = (stage4 == "yes")
        if _stage4_regen_fired(row) != want:
            return False
    for filt, jkey in ((j1, "grounding"), (j2, "format"), (j3, "flavor_purpose")):
        if filt == ALL:
            continue
        want = (filt == "yes")
        if _specialist_ever_failed(row, jkey) != want:
            return False
    return True


def filtered(disposition: str, flavor: str, origin: str, verdict: str,
            rewrite: str, stage4: str, j1: str, j2: str, j3: str) -> List[int]:
    rows = STATE["rows"]
    return nav_widgets.filtered_indices(
        len(rows), lambda i: _matches(rows[i], disposition, flavor, origin, verdict,
                                      rewrite, stage4, j1, j2, j3))


def _scope_label(disposition: str, flavor: str, origin: str, verdict: str,
                 rewrite: str, stage4: str, j1: str, j2: str, j3: str) -> Optional[str]:
    parts = [v for v in (disposition, flavor, origin, verdict) if v != ALL]
    if rewrite != ALL:
        parts.append(f"stage2_rewrite={rewrite}")
    if stage4 != ALL:
        parts.append(f"stage4_regen={stage4}")
    for lbl, v in (("J1", j1), ("J2", j2), ("J3", j3)):
        if v != ALL:
            parts.append(f"{lbl} fail={v}")
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


def _chip(label: str, value, color: str = "var(--background-fill-secondary)",
          gloss: str = None) -> str:
    # Link the LABEL to its App-Guidance definition when it names a KNOWN glossary
    # term (Sandra 2026-07-17). Auto-links only labels present in _GLOSS_SLUGS so
    # chips like 'session|rep' don't get dead links; pass gloss="term" to force one.
    term = gloss if gloss is not None else label
    lbl = (_gloss(f"<b>{_esc(label)}</b>", term)
           if term and term.strip().lower() in _GLOSS_SLUGS
           else f"<b>{_esc(label)}</b>")
    return (f"<span style='display:inline-block;background:{color};border-radius:12px;"
            f"padding:2px 10px;margin:2px 4px 2px 0;font-size:12px'>"
            f"{lbl}: {_esc(value)}</span>")


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
    # Distinguish JUDGE-EXCLUDED (repaired but the re-judge failed again, drop_reason
    # `judge:*`) from a plain workflow DROP — consistent with the tab-1 filter + tab-2
    # split (Sandra 2026-07-17). Kept = green, judge-excluded = red, workflow = amber.
    _judge_excl = str(r.get("drop_reason") or "").startswith("judge:")
    if disp == "kept":
        disp_label, disp_color = "KEPT", "#1a7f37"
    elif _judge_excl:
        disp_label, disp_color = "JUDGE-EXCLUDED", "#b91c1c"
    else:
        disp_label, disp_color = "DROPPED (workflow)", "#d97706"
    bits = [
        f"<span style='background:{disp_color};color:white;border-radius:12px;"
        f"padding:2px 12px;font-weight:700'>{disp_label}</span>",
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
        return ("<div style='color:#b91c1c;font-weight:700'>🔴 " + _gloss("step_metrics")
                + " not present on this row — most often a <b>workflow drop</b> "
                "(a partial timeout/exception stub that never finished the pipeline), or a "
                "pre-2026-07-15 run. Expected for a workflow-dropped row; regenerate at the "
                "producer only if you hit it on a kept or judge-excluded row.</div>")
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


# The REAL best-of-K selection key, mirrored from run_tool_sft_4k._select_best
# (2026-07-16). max() over: (fully_correct, severity_exact, presence_perfect, f1,
# -sev_l1, -eff_risk_err, -fp, -rounds). is_perfect ALONE is NOT the selector — it
# is only the 3rd tier and ties constantly (every zero-error rep is is_perfect), so
# showing it as THE signal is misleading (two is_perfect=True attempts, only one
# wins). This surfaces the cascade the driver actually ranks on.
def _fully_correct(a: Dict) -> Optional[bool]:
    """Mirror _is_fully_correct: is_full_exact, with the None sentinel (severity-
    exact but eff/injury GT unknown) counting as fully-correct. Returns None only
    when neither is_full_exact nor is_severity_exact is present on the attempt."""
    fe = a.get("is_full_exact")
    if fe is None:
        se = a.get("is_severity_exact")
        return bool(se) if se is not None else None
    return bool(fe)


def _attempt_selection_metrics(a: Dict) -> str:
    """The scoring line for ONE best-of-K attempt, in SELECTION-KEY ORDER so the
    reader sees exactly what the driver ranks on — not just f1/is_perfect. Grade
    correctness first (the real bar), presence/f1 second, error-count detail last."""
    def _tick(v):
        return "✓" if v is True else ("✗" if v is False else "?")
    fc = _fully_correct(a)
    sev = a.get("is_severity_exact")
    pres = a.get("is_perfect")            # presence (errors present/absent binarized)
    f1 = a.get("f1")
    tp, fp, fn = a.get("tp"), a.get("fp"), a.get("fn")
    nmm = a.get("n_sev_mismatch")
    bits = [
        # DISPLAY order = loosest → strictest (Sandra 2026-07-17): most-probable
        # first (all errors identified), tightening to the truly-perfect grade. This
        # is presentation only — the SELECTION cascade in _select_best still ranks
        # fully-correct HIGHEST (see _win_reason for what actually decided the pick).
        f"All errors identified {_tick(pres)}",  # right present/absent set (was 'is_perfect')
        f"Severity-exact {_tick(sev)}",     # + every magnitude matches
        f"Fully-correct {_tick(fc)}",       # + eff/injury match → the truly-perfect grade
        f"F1={f1}",                          # presence F1
    ]
    tail = []
    if fp is not None:
        tail.append(f"FP={fp}")             # tiebreak — fewer false-positives wins
    if nmm not in (None, 0):
        tail.append(f"Sev-miss={nmm}")      # how many severities are off (magnitude)
    if a.get("n_tool_calls") is not None:
        tail.append(f"Rounds={a.get('n_tool_calls')}")  # tiebreak — leaner wins
    line = " · ".join(bits)
    if tail:
        line += "  |  " + " · ".join(tail)
    return line


def _win_reason(sel: Dict, others: List[Dict]) -> str:
    """One clause on WHY the selected attempt won, keyed on the FIRST tier where it
    strictly beat the field — the same cascade _select_best walks. Makes 'why 2 not
    4?' answerable instead of 'both say is_perfect=True'."""
    def better_any(pred):
        return any(pred(sel) and not pred(o) for o in others)
    if better_any(lambda x: _fully_correct(x) is True):
        return "Won on: only fully-correct attempt (grade exactly matches GT)"
    if better_any(lambda x: x.get("is_severity_exact") is True):
        return "Won on: severity-exact (all magnitudes match) where others weren't"
    if better_any(lambda x: x.get("is_perfect") is True):
        return "Won on: all errors identified (correct present/absent set) where others missed one"
    # among grade-ties, F1 then the distance/leanness tiebreaks decided it
    sf1 = sel.get("f1") or 0.0
    if any((o.get("f1") or 0.0) < sf1 for o in others):
        return "Won on: higher F1 among grade-tied attempts"
    sfp = sel.get("fp")
    if sfp is not None and any((o.get("fp") or 0) > sfp for o in others):
        return "Won on: fewer false-positives among F1-tied attempts"
    sr = sel.get("n_tool_calls")
    if sr is not None and any((o.get("n_tool_calls") or 0) > sr for o in others):
        return "Won on: leaner (fewer tool rounds) among otherwise-tied attempts"
    return "Won on: tiebreak (grade + distance + leanness all equal — first best kept)"


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
                "best-of-K attempts",
                _pre(gen_prompt or next(iter(prompts)))))
        sel_raw = r.get("raw_model_output")
        if attempts:
            sel_att = next((a for a in attempts
                            if sel_raw is not None and a.get("raw_model_output") == sel_raw), None)
            for a in attempts:
                idx = a.get("attempt_idx")
                selected = a is sel_att
                met = (f"{_attempt_selection_metrics(a)} · "
                       f"status={_esc(a.get('status'))} · shape={_esc(_attempt_shape(a))}")
                win = ""
                if selected:
                    others = [o for o in attempts if o is not a]
                    win = "  ⭐ SELECTED — " + _esc(_win_reason(a, others))
                label = f"① attempt {idx} — {met}{win}"
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
        # Captured separately by the producer (rewrite_reasoning).
        rr = r.get("rewrite_reasoning")
        think_block = _details(
            "② REWRITE — teacher &lt;think&gt; (natural reasoning before the rewrite)",
            _pre(rr, empty="<no separate reasoning captured — server ran non-thinking, "
                 "or think was inline in the raw response below>"),
            open_=False) if "rewrite_reasoning" in r else ""
        out.append(
            "<h4>② Rewrite — GT-align pass (the rep VIDEO is attached in the real call)</h4>"
            f"<div style='font-size:13px;margin-bottom:4px'>{meta}</div>"
            + _details("② REWRITE PROMPT (verbatim, incl. embedded transcript)",
                       _pre(r.get("rewrite_prompt")))
            + think_block
            + _details("② REWRITE RAW RESPONSE (parsed content — post-think)",
                       _pre(r.get("rewrite_raw_response"))))
    elif "rewrite_applied" in r:
        out.append("<h4>② Rewrite</h4><div style='opacity:.7'>No rewrite — natural "
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
        # 'mode' removed (Sandra 2026-07-17): always 'complementary' now — the
        # legacy single 15-tag judge is gone, so it carried no information. Top-level
        # 'notes' also dropped from this header — per-pass notes live inside each
        # pass's RAW RESPONSE JSON (no duplicate caption).
        _fp = r.get("judge_failed_pass")
        header = (f"<div style='font-size:13px;margin-bottom:4px'>Final verdict: "
                  f"<b>{_esc(r.get('judge_verdict_kind'))}</b> · tags={_esc(r.get('judge_tags'))} · "
                  f"accepted_after_regen={r.get('judge_accepted_after_regen')}"
                  + (f" · failed_pass=<b>{_esc(_fp)}</b>" if _fp else "")
                  + "</div>")
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
                             empty="<no separate reasoning captured>"), open_=False))
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
                                 empty="<no separate reasoning captured>"), open_=False))
                    raw_body = _pre(p.get("judge_raw_response"))
                    # The parsed `notes` is ALREADY inside the raw JSON above (as
                    # "notes": "..."), so a separate caption duplicated it (Sandra
                    # 2026-07-17). Only surface it standalone as a FALLBACK when the
                    # raw response doesn't carry it verbatim (a parse-failed / bare
                    # blob) — otherwise the JSON is the single source.
                    _notes = p.get("notes")
                    _raw = p.get("judge_raw_response") or ""
                    if _notes and _notes not in _raw:
                        raw_body += (f"<div style='font-size:12px;opacity:.85;"
                                     f"margin:6px 0 0;padding-top:6px;"
                                     f"border-top:1px solid #e2e8f0'>Notes: "
                                     f"{_esc(_notes)}</div>")
                    inner.append(_details(
                        f"{lbl} RAW RESPONSE (verdict — post-think)", raw_body))
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
                         empty="<no separate reasoning captured>"), open_=False))
            parts.append(_details(f"③ attempt {n} — JUDGE RAW RESPONSE (verdict — post-think)",
                                  _pre(a.get("judge_raw_response"))))
        _title = "③ Gate-3 judge — 3-specialist cascade (+ stage-4 repair)"
        out.append(f"<h4>{_title}</h4>" + "".join(parts))
    else:
        out.append("<h4>③ Gate-3 judge</h4><div style='opacity:.7'>Not run for this row "
                   "(--no-judge run, or dropped before the judge)</div>")

    if not out:
        return "<div style='opacity:.7'>No pipeline-trail fields on this row</div>"
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
    # Wrapped in the SAME colored stage-card style as the ①②③ pipeline trail
    # (Sandra 2026-07-16: this section read as a bare unstyled line) — a
    # distinct 4th accent (emerald, matching the mermaid diagram's "keep" node
    # color) since this IS the pipeline's final "kept" outcome.
    accent = "#059669"
    header = "<b>FINAL shipped <code>messages</code></b> — what SFT actually trains on"
    dropped = r.get("drop_reason") not in (None, "")
    if dropped:
        body = ("<div style='color:#b91c1c;font-weight:700;font-size:14px'>"
                "⛔ NOT shipped — this clip was DROPPED "
                f"(<code>{_esc(r.get('drop_reason'))}</code>), so SFT trains on "
                "NOTHING from it.</div>"
                "<div style='opacity:.75;font-size:13px;margin-top:4px'>The failed "
                "trace is still visible in the pipeline trail above (kept for "
                "inspection — paid compute is never discarded), but it is NOT part "
                "of the training set.</div>")
    else:
        msgs = r.get("messages") or []
        if not msgs:
            body = ("<div style='color:#b91c1c;font-weight:700'>no `messages` on this row"
                    " (dropped before packing, or a partial stub)</div>")
        else:
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
                    open_=False))
            body = (f"<div style='font-size:13px;margin-bottom:4px'>{len(msgs)} turns — "
                    "this is what SFT trains on (loss masks all non-assistant roles)"
                    "</div>" + "".join(parts))
    return (f"<section style='margin:0 0 18px;padding:10px 14px;"
            f"border:1px solid #e2e8f0;border-left:4px solid {accent};"
            f"border-radius:8px;background:#fff'>"
            f"<h4 style='margin:0 0 8px'>{header}</h4>{body}</section>")


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

def show_row(abs_idx: int, disposition: str, flavor: str, origin: str, verdict: str,
            rewrite: str, stage4: str, j1: str, j2: str, j3: str):
    rows = STATE["rows"]
    sel = filtered(disposition, flavor, origin, verdict, rewrite, stage4, j1, j2, j3)
    if not rows:
        empty = "<div style='opacity:.6'>No rows loaded</div>"
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
        scope=_scope_label(disposition, flavor, origin, verdict, rewrite, stage4, j1, j2, j3))
    return (video, vstatus, render_header(r), render_step_metrics(r),
            render_trail(r), render_final_messages(r), render_other_fields(r),
            counter, abs_idx)


def download_sample_txt(abs_idx: int, disposition, flavor, origin, verdict,
                        rewrite, stage4, j1, j2, j3):
    """Write the CURRENTLY-DISPLAYED row to a well-formatted .txt (same layout as
    /preview-output) and return the path for gr.File to serve. Reuses the CLI
    previewer's render_sample — the single source of truth — so the download and
    the terminal preview never drift. Falls back to a pretty JSON dump if that
    module couldn't import."""
    rows = STATE["rows"]
    if not rows:
        return None
    sel = filtered(disposition, flavor, origin, verdict, rewrite, stage4, j1, j2, j3)
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


def nav(delta: Optional[int], abs_idx: int, disposition, flavor, origin, verdict,
       rewrite, stage4, j1, j2, j3):
    sel = filtered(disposition, flavor, origin, verdict, rewrite, stage4, j1, j2, j3)
    if delta is None:
        new = nav_widgets.random_filtered(sel, len(STATE["rows"]))
    else:
        new = nav_widgets.step_filtered(abs_idx, delta, sel)
    return show_row(new, disposition, flavor, origin, verdict, rewrite, stage4, j1, j2, j3)


# ---------------------------------------------------------------------------
# Run-level overview — REUSES the producer's summarize_step_metrics()
# ---------------------------------------------------------------------------

# Target mix proportions (from build_flavor_assignment.FLAVOR_QUOTAS) — the
# insight panel compares the run's ACTUAL kept-per-flavor share against these so
# Sandra sees at a glance whether the mix is on-balance. Imported lazily (the
# builder lives in the repo, not this app dir); falls back to the known 4k mix.
try:  # noqa: SIM105
    import sys as _sys
    if _VISUAL_OBS_DIR not in _sys.path:
        _sys.path.insert(0, _VISUAL_OBS_DIR)
    from build_flavor_assignment import FLAVOR_QUOTAS as _FLAVOR_QUOTAS  # type: ignore
except Exception:
    _FLAVOR_QUOTAS = {"A": 1188, "B": 1188, "C": 712, "D": 712, "E": 200}


def _pct(n, d):
    return (100.0 * n / d) if d else 0.0


def _bar(frac, color="#4f46e5", w=90):
    """A tiny inline proportion bar (frac 0..1) for at-a-glance reading."""
    frac = max(0.0, min(1.0, frac or 0.0))
    return (f"<span style='display:inline-block;width:{w}px;height:9px;"
            f"background:var(--border-color-primary);border-radius:5px;"
            f"vertical-align:middle;overflow:hidden'>"
            f"<span style='display:block;height:100%;width:{frac*100:.0f}%;"
            f"background:{color}'></span></span>")


def _kpi(label, value, sub="", color="#1e1b4b"):
    return (f"<div style='flex:1;min-width:150px;padding:10px 14px;border:1px solid "
            f"var(--border-color-primary);border-radius:10px'>"
            f"<div style='font-size:12px;opacity:.7'>{label}</div>"
            f"<div style='font-size:22px;font-weight:800;color:{color};line-height:1.2'>{value}</div>"
            f"<div style='font-size:11px;opacity:.65'>{sub}</div></div>")


def _run_insights_html(rows: List[Dict]) -> str:
    """AT-A-GLANCE run health (2026-07-16, Sandra) — the story the dense tables
    below make you dig for: keep-rate, the grade invariant, per-flavor balance vs
    the target mix, what's killing yield, and the rewrite/repair funnel. Computed
    from the loaded kept+dropped rows (works without step_metrics)."""
    if not rows:
        return ""
    kept = [r for r in rows if r.get("_disposition") == "kept"]
    dropped = [r for r in rows if r.get("_disposition") == "dropped"]
    nk, nd, nt = len(kept), len(dropped), len(rows)

    # grade invariant: every kept row should be full-exact under the reasoning-only
    # contract (the whole point — the final is GT-composed + appended). Count the
    # exceptions LOUDLY (a non-zero here means a real regression).
    fe = [r.get("final_full_exact") for r in kept]
    n_fe_true = sum(1 for v in fe if v is True)
    n_fe_bad = sum(1 for v in fe if v is False)   # explicit False (None = unknown/old)
    inv_color = "#059669" if n_fe_bad == 0 else "#b91c1c"

    keep_color = "#059669" if _pct(nk, nt) >= 80 else ("#d97706" if _pct(nk, nt) >= 60 else "#b91c1c")
    kpis = "".join([
        _kpi("Rows", f"{nt}", f"{nk} kept · {nd} dropped"),
        _kpi("Keep-rate", f"{_pct(nk, nt):.0f}%", "Kept ÷ (kept+dropped)", keep_color),
        _kpi("Grade invariant", f"{n_fe_true}/{nk}", (
            "final_full_exact on kept" if n_fe_bad == 0
            else f"⚠️ {n_fe_bad} NOT full-exact — regression!"), inv_color),
    ])

    # per-flavor keep-rate + share-vs-target. actual share = kept_f / total_kept;
    # target share = quota_f / sum(quota). A big gap = the mix is off-balance.
    tot_quota = sum(_FLAVOR_QUOTAS.values()) or 1
    kept_by_fl = Counter(str(r.get("flavor")) for r in kept)
    drop_by_fl = Counter(str(r.get("flavor")) for r in dropped)
    fl_rows = []
    for fl in sorted(set(kept_by_fl) | set(drop_by_fl) | set(_FLAVOR_QUOTAS)):
        k, d = kept_by_fl.get(fl, 0), drop_by_fl.get(fl, 0)
        kr = _pct(k, k + d)
        actual_share = _pct(k, nk)
        tgt_share = _pct(_FLAVOR_QUOTAS.get(fl, 0), tot_quota)
        gap = actual_share - tgt_share
        gap_str = (f"<span style='color:{'#059669' if abs(gap) <= 5 else '#d97706'}'>"
                   f"{'+' if gap >= 0 else ''}{gap:.0f}pp</span>")
        fl_rows.append(
            f"<tr><td style='padding:3px 8px;font-weight:700'>{_esc(fl)}</td>"
            f"<td style='padding:3px 8px'>{k}</td><td style='padding:3px 8px'>{d}</td>"
            f"<td style='padding:3px 8px'>{_bar(kr/100)} {kr:.0f}%</td>"
            f"<td style='padding:3px 8px'>{actual_share:.0f}%</td>"
            f"<td style='padding:3px 8px'>{tgt_share:.0f}%</td>"
            f"<td style='padding:3px 8px'>{gap_str}</td></tr>")
    flavor_tbl = (
        "<h4 style='margin:14px 0 4px'>Per-flavor: keep-rate &amp; mix balance</h4>"
        "<div style='font-size:11px;opacity:.65;margin-bottom:4px'>Dropped = ALL drops "
        "(judge-excluded + workflow) · share = this flavor's % of KEPT rows · target = its "
        "% of the 4k quota · Δ near 0pp = on-balance</div>"
        "<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:12.5px'>"
        "<tr><th style='padding:3px 8px;text-align:left'>flavor</th>"
        "<th style='padding:3px 8px;text-align:left'>kept</th>"
        "<th style='padding:3px 8px;text-align:left'>dropped (all)</th>"
        "<th style='padding:3px 8px;text-align:left'>keep-rate</th>"
        "<th style='padding:3px 8px;text-align:left'>share</th>"
        "<th style='padding:3px 8px;text-align:left'>target</th>"
        "<th style='padding:3px 8px;text-align:left'>Δ</th></tr>"
        + "".join(fl_rows) + "</table></div>")

    # what's killing yield — ranked drop reasons, SPLIT into the two categories the
    # filter distinguishes (Sandra 2026-07-17): JUDGE-EXCLUDED = the clip was built +
    # repaired, but the re-judge after the stage-4 rewrite STILL failed (drop_reason
    # `judge:*` — regen_still_failing is repaired-but-rejected-again); vs NON-JUDGE
    # ("workflow") drops = it never reached a clean judge verdict at all (honest opt-out,
    # a rewrite/tool-parts failure, a shape mismatch, an exception). Two subtotals so
    # 'is my yield lost to the JUDGE or to the WORKFLOW?' is answerable at a glance.
    drop_reasons = Counter(str(r.get("drop_reason")) for r in dropped if r.get("drop_reason"))
    if drop_reasons:
        mx = max(drop_reasons.values())
        judge_dr = {k: v for k, v in drop_reasons.items() if k.startswith("judge:")}
        wf_dr = {k: v for k, v in drop_reasons.items() if not k.startswith("judge:")}
        n_judge, n_wf = sum(judge_dr.values()), sum(wf_dr.values())

        def _dr_rows(d):
            return "".join(
                f"<tr><td style='padding:3px 8px'><code>{_esc(rsn)}</code></td>"
                f"<td style='padding:3px 8px;white-space:nowrap'>{n}</td>"
                f"<td style='padding:3px 8px'>{_bar(n/mx, '#b91c1c', 120)}</td>"
                f"<td style='padding:3px 8px;white-space:nowrap'>{_pct(n, nd):.0f}% of drops</td></tr>"
                for rsn, n in sorted(d.items(), key=lambda kv: -kv[1]))

        def _subhead(label, sub, n, color):
            # 3-cell subhead (label+sub span the text cols, subtotal aligned to the
            # share column) so it reads with the rows rather than floating.
            return (f"<tr style='border-top:2px solid {color}33'>"
                    f"<td colspan='3' style='padding:7px 8px 3px'>"
                    f"<span style='color:{color};font-weight:700'>{label}</span> "
                    f"<span style='opacity:.6;font-size:11.5px'>— {sub}</span></td>"
                    f"<td style='padding:7px 8px 3px;white-space:nowrap;"
                    f"font-weight:700;color:{color}'>{n} ({_pct(n, nd):.0f}%)</td></tr>")

        body = ""
        if judge_dr:
            body += _subhead("Judge-excluded", "built + repaired, but the re-judge "
                             "after the stage-4 rewrite failed AGAIN", n_judge, "#b91c1c")
            body += _dr_rows(judge_dr)
        if wf_dr:
            body += _subhead("Workflow drops (non-judge)", "never reached a clean verdict — "
                             "opt-out, rewrite/tool-parts failure, shape mismatch, exception",
                             n_wf, "#d97706")
            body += _dr_rows(wf_dr)
        drop_tbl = ("<h4 style='margin:14px 0 4px'>What's dropping rows — judge vs workflow</h4>"
                    "<div style='font-size:11.5px;opacity:.7;margin-bottom:4px'>"
                    "<b>Judge-excluded</b> = the repaired clip was re-judged and still failed "
                    "(<code>judge:*</code>) · <b>Workflow</b> = it never got a clean verdict "
                    "(opt-out / rewrite / shape / exception). The tab-1 filter treats these as "
                    "mutually exclusive.</div>"
                    "<div style='overflow-x:auto'><table style='border-collapse:collapse;"
                    "font-size:12.5px'><tr><th style='padding:3px 8px;text-align:left'>drop_reason</th>"
                    "<th style='padding:3px 8px;text-align:left'>n</th><th></th>"
                    "<th style='padding:3px 8px;text-align:left'>share</th></tr>"
                    + body + "</table></div>")
    else:
        drop_tbl = ("<h4 style='margin:14px 0 4px'>What's dropping rows</h4>"
                    "<div style='opacity:.7;font-size:12.5px'>No dropped rows loaded — "
                    "0 drops, or the <code>.dropped.jsonl</code> sidecar isn't present.</div>")

    # rewrite / repair funnel — how much teacher work each row cost + opt-out rate.
    n_rewritten = sum(1 for r in rows if r.get("rewrite_kind"))
    n_optout = sum(1 for r in dropped if str(r.get("drop_reason", "")).startswith("sample_excluded_gt_"))
    n_regen = sum(1 for r in rows
                  if ((r.get("step_metrics") or {}).get("judge") or {}).get("n_regen_calls"))
    calls = [((r.get("step_metrics") or {}).get("n_model_calls")) for r in rows]
    calls = [c for c in calls if isinstance(c, (int, float))]
    med_calls = sorted(calls)[len(calls) // 2] if calls else "—"
    funnel = ("<h4 style='margin:14px 0 4px'>Rewrite / repair funnel</h4>"
              "<div style='display:flex;gap:10px;flex-wrap:wrap'>"
              + _kpi("Needed stage-2 rewrite", f"{_pct(n_rewritten, nt):.0f}%",
                     f"{n_rewritten} of {nt} rows")
              + _kpi("Needed stage-4 regen", f"{n_regen}",
                     "Rows a judge-fail sent back")
              + _kpi("Honest opt-outs", f"{n_optout}",
                     "CANNOT_GROUND/RECONCILE drops")
              + _kpi("Median teacher calls", f"{med_calls}", "per row (gen+rewrite+judge)")
              + "</div>")

    # ── K-STRATEGY (best-of-K attempts) — Sandra 2026-07-16: is best-of-K earning
    # its cost per flavor? n_attempts = how many generation tries were spent before
    # the loop stopped (stops early on a severity-exact match, else runs to K).
    # A flavor that almost always stops at 1 doesn't need a big K; one whose mean
    # climbs toward K is USING the budget (and a max pinned at K may be K-starved —
    # bump K). n_degenerate = wasted empty/degenerate tries that DON'T count against
    # the K budget (context for why a row spent many calls). K itself is the ceiling.
    def _stats(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        if not xs:
            return None
        return (min(xs), max(xs), sum(xs) / len(xs), len(xs))
    k_ceiling = max((r.get("n_attempts") or 0) for r in rows) if rows else 0
    all_att = _stats([r.get("n_attempts") for r in rows])
    krow_all = ""
    if all_att:
        mn, mx, mean, n = all_att
        krow_all = _kpi("Attempts / row (best-of-K)", f"{mean:.1f} avg",
                        f"min {mn} · max {mx} · over {n} rows")
    n_hit_ceiling = sum(1 for r in rows if (r.get("n_attempts") or 0) >= k_ceiling and k_ceiling > 1)
    k_by_fl_rows = []
    for fl in sorted(set(str(r.get("flavor")) for r in rows)):
        frows = [r for r in rows if str(r.get("flavor")) == fl]
        st = _stats([r.get("n_attempts") for r in frows])
        dg = _stats([r.get("n_degenerate") for r in frows])
        if not st:
            continue
        mn, mx, mean, n = st
        deg_mean = dg[2] if dg else 0.0
        # bar shows mean attempts as a fraction of the max observed ceiling
        frac = (mean / k_ceiling) if k_ceiling else 0.0
        ceil_note = ("<span style='color:#d97706'> ⟵ at ceiling</span>"
                     if mx >= k_ceiling and k_ceiling > 1 else "")
        k_by_fl_rows.append(
            f"<tr><td style='padding:3px 8px;font-weight:700'>{_esc(fl)}</td>"
            f"<td style='padding:3px 8px'>{n}</td>"
            f"<td style='padding:3px 8px'>{mn}</td>"
            f"<td style='padding:3px 8px'>{mx}{ceil_note}</td>"
            f"<td style='padding:3px 8px'>{_bar(frac, '#0891b2')} {mean:.1f}</td>"
            f"<td style='padding:3px 8px'>{deg_mean:.1f}</td></tr>")
    kstrat = (
        "<h4 style='margin:14px 0 4px'>K-strategy — best-of-K attempts per flavor</h4>"
        "<div style='font-size:11px;opacity:.65;margin-bottom:4px'>"
        "Attempts = generation tries spent before stop-on-perfect (ceiling = K). "
        "Avg near 1 ⇒ K barely used for that flavor · avg climbing / max at ceiling ⇒ "
        "K is being used (maybe K-starved — bump K). Degen = wasted empty tries "
        "(don't count against K).</div>"
        "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px'>"
        + krow_all
        + _kpi("Observed K ceiling", f"{k_ceiling}", "max attempts any row spent")
        + _kpi("Rows at the ceiling", f"{n_hit_ceiling}",
               "used every try — candidates for a bigger K")
        + "</div>"
        "<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:12.5px'>"
        "<tr><th style='padding:3px 8px;text-align:left'>flavor</th>"
        "<th style='padding:3px 8px;text-align:left'>rows</th>"
        "<th style='padding:3px 8px;text-align:left'>min</th>"
        "<th style='padding:3px 8px;text-align:left'>max</th>"
        "<th style='padding:3px 8px;text-align:left'>mean attempts</th>"
        "<th style='padding:3px 8px;text-align:left'>degen avg</th></tr>"
        + "".join(k_by_fl_rows) + "</table></div>")

    # Box each insight group in a bordered card so the overview reads as distinct
    # blocks (Sandra 2026-07-17, "prefer bordered containers for accessibility") —
    # the same treatment as the guidance tab. Each group already leads with its own
    # <h4>; _card lifts that into the card's header band and boxes the rest.
    import re as _re
    def _card(fragment, accent="#6366f1"):
        m = _re.match(r"\s*<h4[^>]*>(.*?)</h4>(.*)", fragment, _re.DOTALL)
        title, body = (m.group(1), m.group(2)) if m else ("", fragment)
        head = (f"<div style='background:#eef2ff;padding:8px 14px;"
                f"border-bottom:1px solid #e2e8f0;color:#1e1b4b;font-weight:700;"
                f"font-size:14px'>{title}</div>" if title else "")
        return (f"<section style='border:1px solid #e2e8f0;border-left:4px solid {accent};"
                f"border-radius:10px;overflow:hidden;margin:0 0 14px;"
                f"box-shadow:0 1px 2px rgba(0,0,0,.04)'>{head}"
                f"<div style='padding:8px 14px'>{body}</div></section>")

    return (
        _card("<h4>Run health</h4>"
              "<div style='display:flex;gap:10px;flex-wrap:wrap'>" + kpis + "</div>",
              accent="#0891b2")
        + _card(flavor_tbl) + _card(drop_tbl) + _card(funnel) + _card(kstrat))


def overview():
    path = STATE["path"]
    if not path:
        return "<div>load a run first</div>", None
    # AT-A-GLANCE insight panel FIRST (independent of the producer import — it reads
    # the loaded rows, so it works even if step_metrics.py failed to import).
    htm: List[str] = [_run_insights_html(STATE["rows"])]

    if summarize_step_metrics is None:
        htm.append(f"<div style='color:#b91c1c'>🔴 could not import the producer's "
                   f"step_metrics.py: <code>{_esc(_SUMMARIZE_IMPORT_ERROR)}</code> — "
                   f"the schema-driven aggregate below is unavailable; the insight "
                   f"panel above is still valid.</div>")
        return "".join(htm), None

    paths = [path, path + ".dropped.jsonl"]
    lines: List[str] = []
    agg = summarize_step_metrics(paths, print_fn=lines.append)

    # COST & TIMING per flavor (2026-07-16 — Sandra "show less, easier to understand"):
    # the insight panel above already covers keep-rate / drops / rewrite+regen rates /
    # K-strategy in plain language, so this table shows ONLY what it doesn't: how LONG
    # each flavor takes per stage and how BIG its final trace is. Friendly labels; the
    # complete raw aggregate (all 14 producer keys) stays available in a details fold.
    if agg:
        # (raw_key, friendly label, unit-suffix) — median unless noted.
        SHOW = [
            ("gen_wall_s_p50", "Generate", "s"),
            ("rewrite_wall_s_p50", "Rewrite", "s"),
            ("judge_wall_s_p50", "Judge", "s"),
            ("final_output_tokens_p50", "Final size", " tok"),
        ]

        def _num(v, suf):
            if v is None:
                return "—"
            return (f"{v:.0f}{suf}" if isinstance(v, (int, float)) else f"{_esc(v)}{suf}")
        head = "".join(f"<th style='padding:3px 8px;text-align:left'>{lbl}</th>"
                       for _k, lbl, _s in SHOW)
        rows_h = []
        for fl in sorted(agg):
            d = agg[fl]
            total = sum((d.get(k) or 0) for k in ("gen_wall_s_p50", "rewrite_wall_s_p50",
                                                  "judge_wall_s_p50")
                        if isinstance(d.get(k), (int, float)))
            tds = "".join(f"<td style='padding:3px 8px'>{_num(d.get(k), s)}</td>"
                          for k, _lbl, s in SHOW)
            rows_h.append(
                f"<tr><td style='padding:3px 8px;font-weight:700'>{_esc(fl)} "
                f"<span style='opacity:.55;font-weight:400'>(n={d.get('n', 0)})</span></td>"
                f"{tds}<td style='padding:3px 8px;font-weight:600'>{total:.0f}s</td></tr>")
        htm.append(
            "<h4 style='margin:4px 0'>Cost &amp; timing per flavor "
            "<span style='font-weight:400;font-size:12px;opacity:.65'>(median per row)</span></h4>"
            "<div style='font-size:11px;opacity:.65;margin-bottom:4px'>How long each "
            "flavor spends in each teacher stage, and how big its final trace is — "
            "C is the slowest (always two-pass: generate→rewrite→judge).</div>"
            "<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:12.5px'>"
            f"<tr><th style='padding:3px 8px;text-align:left'>flavor</th>{head}"
            "<th style='padding:3px 8px;text-align:left'>total</th></tr>"
            + "".join(rows_h) + "</table></div>")
        # the full producer aggregate, folded away for anyone who wants every key.
        cols: List[str] = []
        for d in agg.values():
            for k in d:
                if k not in cols:
                    cols.append(k)
        rhead = "".join(f"<th style='padding:4px 8px;text-align:left'>{_esc(c)}</th>" for c in cols)
        rraw = []
        for fl in sorted(agg):
            tds = "".join(f"<td style='padding:4px 8px'>{_fmt_cell(agg[fl].get(c))}</td>"
                          for c in cols)
            rraw.append(f"<tr><td style='padding:4px 8px;font-weight:700'>{_esc(fl)}</td>{tds}</tr>")
        htm.append(_details(
            "full step_metrics aggregate (all producer keys)",
            "<div style='font-size:11px;opacity:.7;margin:4px 0'>From the producer's own "
            "<code>summarize_step_metrics()</code>; kept+dropped rows. "
            "<code>final_not_perfect_count</code> should be ~0 (the grade invariant); "
            "rates are 0–1.</div>"
            f"<div style='overflow-x:auto'><table style='border-collapse:collapse;"
            f"font-size:12px'><tr><th style='padding:4px 8px'>flavor</th>{rhead}</tr>"
            + "".join(rraw) + "</table></div>"))
    else:
        htm.append("<div style='color:#b91c1c;font-weight:700'>🔴 No rows with "
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
    # Folded away (2026-07-16 "show less"): the insight panel above already gives
    # keep-rate + ranked drops in plain language; this is the full per-flavor ×
    # judge-verdict × drop-reason cross-tab for anyone who wants the raw counts.
    htm.append(_details(
        "Per-flavor × judge-verdict × drop-reason cross-tab (raw counts)",
        f"<div style='overflow-x:auto'><table style='border-collapse:collapse;"
        f"font-size:12.5px'><tr><th style='padding:4px 8px'>flavor</th>{head}</tr>"
        + "".join(rows_h) + "</table></div>"))

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
        htm.append(f"<div style='opacity:.7'>Histogram unavailable: {_esc(e)}</div>")

    return "".join(htm), fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# Top-level tab buttons + the "FINAL shipped messages" section header get a
# distinct accent color each (Sandra 2026-07-16 — plain black-text tabs/headers
# didn't read as distinct areas). `elem_id="X"` on a gr.Tab puts id="X-button"
# on its actual tab button (documented Gradio hook) — targeted here, not a
# fragile text/position selector that breaks if a label changes.
_TAB_CSS = """
#tab-row-inspector-button { color: #4f46e5 !important; }
#tab-row-inspector-button.selected { border-color: #4f46e5 !important; }
#tab-overview-button { color: #059669 !important; }
#tab-overview-button.selected { border-color: #059669 !important; }
#tab-guidance-button { color: #7c3aed !important; }
#tab-guidance-button.selected { border-color: #7c3aed !important; }
.gloss-flash { animation: glossflash 1.6s ease-out; }
@keyframes glossflash {
  0%,20% { background: #ede9fe; box-shadow: 0 0 0 3px #c4b5fd inset; }
  100%   { background: transparent; box-shadow: none; }
}
"""

# Cross-tab glossary navigation (Sandra 2026-07-17). Gradio's gr.HTML strips
# <script>, so the click handler is installed via the Blocks `js=` hook, which runs
# once at app load and is NOT sanitized. A single delegated listener on document
# catches any `.gloss` link click: it switches to the App Guidance tab (clicking its
# tab button), then scrolls the matching `#gloss-<slug>` row into view and flashes
# it. A short retry loop covers the tab's content mounting lazily after the switch.
_GLOSS_JS = """
() => {
  document.addEventListener('click', (e) => {
    const a = e.target.closest && e.target.closest('a.gloss');
    if (!a) return;
    e.preventDefault();
    const slug = a.getAttribute('data-gloss');
    if (!slug) return;
    const btn = document.getElementById('tab-guidance-button');
    if (btn) btn.click();
    let tries = 0;
    const go = () => {
      const el = document.getElementById('gloss-' + slug);
      if (el) {
        el.scrollIntoView({behavior: 'smooth', block: 'center'});
        el.classList.remove('gloss-flash');
        void el.offsetWidth;            // reflow so the animation re-fires
        el.classList.add('gloss-flash');
      } else if (tries++ < 20) {
        setTimeout(go, 60);            // guidance tab content still mounting
      }
    };
    setTimeout(go, 80);
  }, true);
}
"""


def build_ui() -> gr.Blocks:
    # theme/css/js moved to launch() — Gradio 6.0 relocated them off the Blocks
    # constructor (a constructor-arg js= is IGNORED, so the gloss handler wouldn't
    # install). They're applied in main()'s demo.launch(...) instead.
    with gr.Blocks(title="pipeline-inspector — VObs-tool-SFT") as demo:
        gr.Markdown("## 🔬 pipeline-inspector — VObs-tool-SFT pipeline quality "
                    "(gen → rewrite → judge → regen · row video · step_metrics)")

        # Section ① — load a run. Visually separated from the filter/nav/display
        # controls below (Sandra 2026-07-16: the toolbar read as one flat stack
        # of same-styled widgets — group by PURPOSE instead).
        gr.Markdown("#### 1 · Load a run")
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
            with gr.Tab("Row inspector", elem_id="tab-row-inspector"):
                # Section ② — filter which rows are in scope.
                gr.Markdown("#### 2 · Filter")
                with gr.Row():
                    disp_dd = gr.Dropdown(choices=disposition_choices(), value=ALL,
                                          label="kept / dropped / judge-excluded")
                    flavor_dd = gr.Dropdown(choices=[ALL], value=ALL, label="flavor")
                    origin_dd = gr.Dropdown(choices=[ALL], value=ALL, label="prompt_origin")
                    verdict_dd = gr.Dropdown(choices=[ALL], value=ALL,
                                             label="judge_verdict_kind")
                with gr.Row():
                    rewrite_dd = gr.Dropdown(choices=YES_NO, value=ALL,
                                             label="Stage-2 GT-align rewrite applied?")
                    stage4_dd = gr.Dropdown(choices=YES_NO, value=ALL,
                                            label="Stage-4 judge repair fired?")
                    j1_dd = gr.Dropdown(choices=YES_NO, value=ALL,
                                        label="J1 grounding — ever failed?")
                    j2_dd = gr.Dropdown(choices=YES_NO, value=ALL,
                                        label="J2 format — ever failed?")
                    j3_dd = gr.Dropdown(choices=YES_NO, value=ALL,
                                        label="J3 flavor-purpose — ever failed?")

                # Section ③ — move through the filtered rows (step / random /
                # jump-to-index all live together, one visual group).
                gr.Markdown("#### 3 · Navigate")
                prev_btn, next_btn, random_btn, refresh_btn, counter_md = \
                    nav_widgets.make_nav_row()
                jump_input, jump_btn = nav_widgets.make_jump_row("Jump to row index (0-based)")

                # Section ④ — display + export the CURRENTLY shown row.
                gr.Markdown("#### 4 · Display & export this row")
                with gr.Row():
                    pre_fs = gr.Slider(8, 28, value=12, step=1,
                                       label="Prompt text size (px)", scale=3)
                    dl_btn = gr.Button("⬇ Download this sample (.txt)", scale=1)
                # Prompt text size sets the --pre-fs CSS var live (client-side, no
                # server round-trip) so every prompt/output <pre> scales.
                pre_fs.change(
                    None, pre_fs, None,
                    js="(v)=>{document.documentElement.style.setProperty("
                       "'--pre-fs', v+'px'); return [];}")
                # Full formatted dump of the displayed sample (same layout as
                # /preview-output). Appears when the button is clicked. height
                # caps the empty-state drop-zone (was a tall blank box before
                # any download — Sandra 2026-07-16).
                dl_file = gr.File(label="sample .txt (all data for this one sample)",
                                  visible=True, height=80)

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
                final_html = gr.HTML()
                with gr.Accordion("All row fields not rendered above",
                                  open=False):
                    other_html = gr.HTML()

            with gr.Tab("Run overview", elem_id="tab-overview"):
                overview_btn = gr.Button("Compute overview",
                                         variant="primary")
                overview_html = gr.HTML()
                overview_plot = gr.Plot(label="change-ratio distributions per flavor")

            with gr.Tab("App Guidance", elem_id="tab-guidance"):
                # Plain-language glossary of every idea in this app + the live
                # workflow diagram (read from the canonical .mmd, so it can't drift).
                gr.HTML(guidance_html())

        idx_state = gr.State(0)

        row_outputs = [video, video_status, header_html, metrics_html, trail_html,
                       final_html, other_html, counter_md, idx_state]
        filter_inputs = [disp_dd, flavor_dd, origin_dd, verdict_dd,
                         rewrite_dd, stage4_dd, j1_dd, j2_dd, j3_dd]

        def do_load(dd_path, tb_path):
            path = dd_path or tb_path
            status = load_run(path)
            upd = [gr.update(choices=_choices(f), value=ALL)
                   for f in ("flavor", "prompt_origin", "judge_verdict_kind")]
            # rewrite/stage4/J1/J2/J3 use static YES_NO choices — just reset the value.
            reset_yn = [gr.update(value=ALL) for _ in range(5)]
            first = show_row(0, ALL, ALL, ALL, ALL, ALL, ALL, ALL, ALL, ALL)
            return [status, gr.update(value=ALL)] + upd + reset_yn + list(first)

        load_btn.click(do_load, [run_dd, path_tb],
                       [load_status] + filter_inputs + row_outputs)

        # Re-discover runs (newest first) and repopulate the dropdown, selecting the
        # newest so a fresh smoke is one click away. Does NOT load — user hits Load.
        def do_rescan():
            runs = discover_runs()
            return gr.update(choices=runs, value=(runs[0] if runs else None))
        rescan_btn.click(do_rescan, [], [run_dd])

        for dd in filter_inputs:
            dd.change(lambda i, d, f, o, v, rw, s4, j1, j2, j3: show_row(i, d, f, o, v, rw, s4, j1, j2, j3),
                      [idx_state] + filter_inputs, row_outputs)
        prev_btn.click(lambda i, d, f, o, v, rw, s4, j1, j2, j3: nav(-1, i, d, f, o, v, rw, s4, j1, j2, j3),
                       [idx_state] + filter_inputs, row_outputs)
        next_btn.click(lambda i, d, f, o, v, rw, s4, j1, j2, j3: nav(+1, i, d, f, o, v, rw, s4, j1, j2, j3),
                       [idx_state] + filter_inputs, row_outputs)
        random_btn.click(lambda i, d, f, o, v, rw, s4, j1, j2, j3: nav(None, i, d, f, o, v, rw, s4, j1, j2, j3),
                         [idx_state] + filter_inputs, row_outputs)
        refresh_btn.click(lambda i, d, f, o, v, rw, s4, j1, j2, j3: show_row(i, d, f, o, v, rw, s4, j1, j2, j3),
                          [idx_state] + filter_inputs, row_outputs)
        jump_btn.click(lambda j, d, f, o, v, rw, s4, j1, j2, j3: show_row(int(j), d, f, o, v, rw, s4, j1, j2, j3),
                       [jump_input] + filter_inputs, row_outputs)

        dl_btn.click(download_sample_txt,
                     [idx_state] + filter_inputs, [dl_file])

        overview_btn.click(overview, [], [overview_html, overview_plot])

        demo.load(do_load, [run_dd, path_tb],
                  [load_status] + filter_inputs + row_outputs)
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=7880)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)
    demo = build_ui()
    # Gradio 6.0: theme/css/js belong on launch(), not the Blocks constructor.
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                allowed_paths=[VIDEO_CACHE_DIR],
                theme=gr.themes.Soft(), css=_TAB_CSS, js=_GLOSS_JS)


if __name__ == "__main__":
    main()
