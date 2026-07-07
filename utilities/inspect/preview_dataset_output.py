#!/usr/bin/env python3
"""Preview a generated dataset's full pipeline as plain text: GENERATION PROMPT ->
RAW MODEL OUTPUT -> PARSED OUTPUT, for N sample rows per leaf.

The pre-run gate behind /preview-output (Sandra, 2026-06-29): ALWAYS inspect what a
generator actually produces (prompt + raw + parsed) BEFORE launching a full run, so a
mismatch between intent and implementation is caught cheaply.

Works on:
  - a DatasetDict dir (has split subdirs),
  - a single saved split dir,
  - a PARENT dir containing several leaf datasets (auto-discovers leaves),
  - a generator JSONL sidecar (`*_traces_*.jsonl` or any `.jsonl` of row dicts),
  - a PARENT dir containing per-family sidecars (`*/_traces_*.jsonl`).

Each row renders FIVE sections: 0. KEY METADATA + completeness audit · 1. GENERATION
PROMPT · 2. MODEL REASONING · 3. RAW MODEL OUTPUT · 4. PARSED OUTPUT (messages).

Provenance columns it looks for (any subset): generation_prompt, raw_model_output,
messages, choices, correct_answer + the reasoning/judge metadata fields. If
generation_prompt / raw_model_output are absent it says so explicitly (never a
silent half-preview). The fields shown in section 0 are configurable with
`--meta-fields a,b,c`; defaults cover the text-reasoning + judge pipeline.

Reusable by BOTH /generate-reas (preview generated traces) and /vlm-judge
(preview judged rows: judge decision/tags surfaced in section 0).

Usage:
  python preview_dataset_output.py --root <dir-or-jsonl> --n 3 [--split train] [--out preview.txt]
  python preview_dataset_output.py --root /home/sgsilva/tmp/<gen_smoke> --n 3   # sidecars
  python preview_dataset_output.py --root <judged_dataset> --meta-fields reasoning_judge_decision,reasoning_repair_tags
  # diversity preview — one representative row per (style × model) cell (covers all combos,
  # not just the first-n rows which on a session-ordered leaf can all be one cell):
  python preview_dataset_output.py --root <gen_dir> --n 1 --group-by reasoning_prompt_style,reasoning_model
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from datasets import load_from_disk


class _JsonlDataset:
    """Minimal Dataset-like view over a JSONL of row dicts (generator sidecars),
    so _render works identically for HF dirs and sidecars. Skips rows whose
    `_row_status`/`_twin_status` is present and != 'ok' (sentinel failures)."""

    def __init__(self, rows):
        self.rows = rows
        cols = set()
        for r in rows:
            cols.update(r.keys())
        self.column_names = sorted(cols)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _load_jsonl(p: Path):
    import json
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = r.get("_row_status", r.get("_twin_status", "ok"))
        if status != "ok":
            continue
        # some generators (e.g. image reasoning) store `messages` JSON-stringified
        # rather than as a native list -- normalize so downstream code (which
        # assumes a list of dicts) doesn't crash on `.get()` against a str.
        if isinstance(r.get("messages"), str):
            try:
                r["messages"] = json.loads(r["messages"])
            except json.JSONDecodeError:
                pass
        rows.append(r)
    return _JsonlDataset(rows)


def _is_dataset_dir(p: Path) -> bool:
    return (p / "dataset_info.json").exists() or (p / "state.json").exists()


def _has_split_subdirs(p: Path) -> bool:
    return any(_is_dataset_dir(p / s) for s in ("train", "test", "validation"))


def _load_any(p: Path):
    """Return list of (label, Dataset). Handles a JSONL sidecar, DatasetDict,
    single split, or split subdir."""
    out = []
    if p.is_file() and p.suffix == ".jsonl":
        return [(p.parent.name + "/" + p.name, _load_jsonl(p))]
    try:
        ds = load_from_disk(str(p))
    except Exception:
        ds = None
    if ds is not None:
        if hasattr(ds, "keys"):  # DatasetDict
            for split in ds:
                out.append((f"{p.name}/{split}", ds[split]))
        else:  # single Dataset
            out.append((p.name, ds))
        return out
    # split subdirs saved separately (e.g. <leaf>/train, <leaf>/test)
    if _has_split_subdirs(p):
        for split in ("train", "test", "validation"):
            sd = p / split
            if _is_dataset_dir(sd):
                out.append((f"{p.name}/{split}", load_from_disk(str(sd))))
    return out


def _discover_leaves(root: Path):
    """Yield (label, Dataset) for root and any nested leaf datasets or JSONL
    sidecars (generator output: <root>/<family>/_traces_<split>.jsonl)."""
    if root.is_file():
        return _load_any(root)
    # generator sidecars under per-family subdirs
    sidecars = sorted(root.glob("*/_traces_*.jsonl")) + sorted(root.glob("_traces_*.jsonl"))
    if sidecars:
        out = []
        for sc in sidecars:
            out.extend(_load_any(sc))
        return out
    direct = _load_any(root)
    if direct:
        return direct
    # parent dir: scan children
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            out.extend(_load_any(child))
    return out


def _text_of(content) -> str:
    if isinstance(content, list):
        return " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    return str(content)


# Columns/keys where a thinking model's reasoning may live, in priority order.
# NOTE: `generation_reasoning` is DELIBERATELY EXCLUDED. On LLM-authored (v7) text
# families it holds the item-AUTHORING scratchpad ("Create TWO MCQs… Constraint 7…"),
# not the answer trace — showing it made section 2 lie for ~4/7 families while the
# clean trace sat in messages/raw output. Section 2 must mirror what SFT packs (the
# assistant `messages` <think>), so we read the trace from there FIRST.
# `[[project_text_2706_reasoning_twins]]`
_REASONING_COLS = ("reasoning_content", "reasoning", "reasoning_trace",
                   "think", "thinking", "cot")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _extract_reasoning(row, msgs) -> str:
    """Pull the reasoning SFT would actually train on. Priority: the <think> block in
    the assistant `messages` (what the packer reads) -> raw_model_output <think> ->
    a dedicated reasoning column. `generation_reasoning` is never trusted (see above).
    Returns '' if there is genuinely none (and the caller flags that)."""
    # 1) the assistant message <think> — exactly what nemo-rl SFT packs
    if msgs:
        for mm in msgs:
            if mm.get("role") == "assistant":
                mt = _text_of(mm.get("content", ""))
                mt_m = _THINK_RE.search(mt)
                if mt_m and mt_m.group(1).strip():
                    return mt_m.group(1).strip()
    # 2) raw model output <think>
    raw = row.get("raw_model_output", "") or ""
    m = _THINK_RE.search(raw)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # 3) a dedicated reasoning column, last resort
    for c in _REASONING_COLS:
        v = row.get(c)
        if v:
            return str(v).strip()
    if msgs:
        for mm in msgs:
            if mm.get("role") == "assistant":
                for c in _REASONING_COLS:
                    if mm.get(c):
                        return str(mm[c]).strip()
    return ""


# Default KEY metadata fields surfaced in section 0 (present-AND-filled audit).
# Covers the text-reasoning + judge pipelines; override with --meta-fields.
_DEFAULT_META_FIELDS = (
    "dataset_type", "source_dataset_family", "source_version", "exercise_id",
    "exercise_name", "body_region", "is_lr_pair", "task", "correct_answer",
    "reasoning_added", "reasoning_model",
    "reasoning_prompt_style", "reasoning_origin",
    "_pitfall_tags", "_pitfall_retry_count",
    "reasoning_judge_decision", "reasoning_regenerated", "reasoning_judge_model",
    "reasoning_repair_tags", "reasoning_judge_prompt",
    # vlm-judge sample/audit output (surfaced in §0 when present):
    "verdict_kind", "category", "answerability", "confidence", "evidence",
    "gt", "margin", "judge_model", "tier",
)


def _gen_prompt_of(row):
    """The reasoning-GENERATION prompt — `generation_prompt` (the teacher prompt
    that produced the <think> trace). The visual-observations generator
    (add_reasoning_traces_vlm.py) persists it as `reasoning_teacher_prompt`
    instead — additive fallback, backwards compatible."""
    return row.get("generation_prompt") or row.get("reasoning_teacher_prompt")


def _judge_prompt_of(row):
    """The reasoning-JUDGE prompt sent to the judge model. Persisted on judged
    datasets as `reasoning_judge_prompt` (judge_reasoning_text_mcqa.py); legacy
    vlm-judge sidecars use `judge_prompt`. Returns None on un-judged datasets so
    section 1b is simply omitted — backwards compatible."""
    return row.get("reasoning_judge_prompt") or row.get("judge_prompt")


def _prompt_of(row):
    """Any single prompt — gen first, else judge. Kept for callers/paths that
    want one prompt; section 1 renders both explicitly when both are present."""
    return _gen_prompt_of(row) or _judge_prompt_of(row)


def _raw_of(row):
    """Raw model output — `raw_model_output` (generators) OR `raw_response`
    (vlm-judge output) OR `reasoning_raw_response` (the visual-observations
    generator's raw teacher output) — additive fallback, backwards compatible."""
    return (row.get("raw_model_output") or row.get("raw_response")
            or row.get("reasoning_raw_response"))


def _row_indices_by_group(d, group_cols, n):
    """Row indices for a --group-by preview: the first `n` rows of EACH distinct
    value-tuple of group_cols (n=1 → one representative per cell). Shows every
    cell, so a diversity preview covers all (style × model) combos instead of the
    first-n rows — which, on a session-ordered leaf, can all be one cell. Falls
    back to the first-n rows if none of group_cols are present."""
    present = [c for c in group_cols if c in d.column_names]
    if not present:
        return list(range(min(n, len(d))))
    from collections import defaultdict
    per_cell = defaultdict(list)
    for i in range(len(d)):
        r = d[i]
        key = tuple(str(r.get(c, "")) for c in present)
        if len(per_cell[key]) < max(1, n):
            per_cell[key].append(i)
    return sorted(i for idxs in per_cell.values() for i in idxs)


def _render(label, d, n, w, meta_fields, group_cols=None):
    w(f"\n{'='*100}\n=== LEAF: {label}   (rows={len(d)}, cols={len(d.column_names)})\n{'='*100}")
    cols = set(d.column_names)
    # prompt/raw may be under generator OR judge column names — only warn if NEITHER
    # alias is present, so judge outputs (judge_prompt/raw_response) don't false-warn.
    missing = []
    if not (cols & {"generation_prompt", "judge_prompt", "reasoning_judge_prompt",
                    "reasoning_teacher_prompt"}):
        missing.append("generation_prompt/judge_prompt")
    if not (cols & {"raw_model_output", "raw_response", "reasoning_raw_response"}):
        missing.append("raw_model_output/raw_response")
    if missing:
        w(f"!! MISSING provenance columns: {missing} — showing what IS present "
          f"(at minimum messages). Columns: {d.column_names}")
    if group_cols:
        row_idxs = _row_indices_by_group(d, group_cols, n)
        present_gc = [c for c in group_cols if c in d.column_names]
        w(f"  [--group-by {','.join(group_cols)}] → {len(row_idxs)} rows across "
          f"{len(set(tuple(str(d[i].get(c,'')) for c in present_gc) for i in row_idxs))} cells")
    else:
        row_idxs = list(range(min(n, len(d))))
    for i in row_idxs:
        r = d[i]
        ex = r.get("exercise_name") or r.get("exercise_id") or ""
        # `task`/`variant` are the text-pipeline names; image judge sidecars use
        # `task_type`/`question_template` — fall back so the header isn't '?'.
        task = r.get("task") or r.get("task_type") or "?"
        variant = r.get("variant") or r.get("question_template") or "?"
        cell = (f"  cell=({','.join(str(r.get(c,'')) for c in group_cols if c in d.column_names)})"
                if group_cols else "")
        w(f"\n{'-'*100}\nROW {i}  exercise={ex!r}  task={task!r}  variant={variant!r}{cell}\n{'-'*100}")

        w("\n### 0. KEY METADATA (present-AND-filled audit) ###")
        empty = []
        for f in meta_fields:
            if f not in cols:
                continue
            v = r.get(f)
            filled = v not in (None, "", [], {})
            # Section 0 is a present-AND-filled audit, not a content dump: abbreviate
            # long values (e.g. reasoning_judge_prompt — full text is in section 1b).
            disp = repr(v)
            if len(disp) > 120:
                disp = disp[:117] + "…'"
            w(f"  {f:26s} = {disp}" + ("" if filled else "   ⚠ EMPTY"))
            if not filled and f in ("messages", "reasoning_model", "source_version",
                                    "dataset_type"):
                empty.append(f)
        if empty:
            w(f"  ⚠ LOAD-BEARING FIELDS EMPTY: {empty} — STOP, fix the generator.")

        # Section 1 renders BOTH prompts when present. Label by OPERATION, not a
        # fixed "reasoning" assumption: a REASONING-judge dataset carries a
        # `reasoning_judge_prompt` (+ the reasoning-GENERATION teacher prompt); a
        # SAMPLE-QUALITY judge sidecar carries `judge_prompt` with NO generation
        # prompt (it audits the sample/answer, not a <think> trace). Mislabeling a
        # sample-quality audit as a "reasoning judge" is the wrong operation
        # (judge-SAMPLE vs judge-REASONING — see /vlm-judge).
        gen_p = _gen_prompt_of(r)
        judge_p = _judge_prompt_of(r)
        is_reasoning_judge = bool(r.get("reasoning_judge_prompt"))
        if judge_p and (gen_p or is_reasoning_judge):
            # reasoning pipeline: generation trace + reasoning-judge verdict
            w("\n### 1a. REASONING-GENERATION PROMPT (teacher → <think>) ###")
            w(gen_p or "<no generation_prompt column>")
            w("\n### 1b. REASONING-JUDGE PROMPT (judge → trace verdict) ###")
            w(judge_p)
        elif judge_p:
            # sample-quality judge: one prompt, audits the sample/answer (no trace)
            w("\n### 1. SAMPLE-QUALITY JUDGE PROMPT (judge → sample verdict) ###")
            w(judge_p)
        else:
            w("\n### 1. GENERATION PROMPT (sent to the TEACHER only — includes the "
              "correct answer/reference + instructions the student never sees; NOT "
              "what the student model is trained on) ###")
            w(gen_p or _prompt_of(r) or "<no generation_prompt/judge_prompt column>")

        # For a reasoning pipeline this is the model's <think> trace; for a
        # sample-quality judge it's the judge's analysis preamble before the JSON.
        _sec2 = ("JUDGE ANALYSIS (prose before the verdict JSON)"
                 if (judge_p and not gen_p and not is_reasoning_judge)
                 else "MODEL REASONING (<think> / reasoning_content)")
        w(f"\n### 2. {_sec2} ###")
        reasoning = _extract_reasoning(r, r.get("messages"))
        if reasoning:
            w(reasoning)
        else:
            w("<none captured — model emitted no reasoning, OR the server's reasoning_parser "
              "stripped <think> and the generator did not store reasoning_content>")

        w("\n### 3. RAW MODEL OUTPUT ###")
        w(_raw_of(r) or "<no raw_model_output/raw_response column>")

        w("\n### 4. PARSED OUTPUT (messages) — the actual SFT row; the [user] turn "
          "here is what the student model is trained/served on, and MUST be "
          "byte-identical to the NR source row's [user] turn (the _2706 contract) ###")
        msgs = r.get("messages")
        if msgs:
            for m in msgs:
                w(f"[{m.get('role','?')}] {_text_of(m.get('content',''))}")
        else:
            w("<no messages column>")
        if "choices" in cols:
            w(f"\nchoices: {r.get('choices')}")
        if "correct_answer" in cols:
            w(f"correct_answer: {r.get('correct_answer')}")


def _coverage_table(leaves, w):
    """When the diversity provenance columns are present (reasoning_prompt_style /
    reasoning_model, or the combined reasoning_origin), print a style x model
    coverage table aggregated across ALL leaves — proves the (style, model)
    sampling space is actually being exercised and that both fields are
    present-AND-filled. No-op when those columns are absent (non-diversity data).
    `[[project_text_2706_reasoning_twins]]`"""
    from collections import Counter
    has_any = any(
        ("reasoning_prompt_style" in d.column_names
         or "reasoning_origin" in d.column_names)
        for _, d in leaves)
    if not has_any:
        return

    grid = Counter()          # (style, model) -> n
    style_tot = Counter()
    model_tot = Counter()
    empty = 0
    total = 0
    per_leaf = {}             # label -> Counter of origins
    for label, d in leaves:
        cols = set(d.column_names)
        leaf_c = Counter()
        for i in range(len(d)):
            r = d[i]
            style = r.get("reasoning_prompt_style")
            model = r.get("reasoning_model")
            origin = r.get("reasoning_origin")
            # derive style/model from origin ("model:style") if the split cols
            # are absent but the combined one is present.
            if (style is None or model is None) and isinstance(origin, str) and ":" in origin:
                om, _, os_ = origin.partition(":")
                model = model or om
                style = style or os_
            total += 1
            if not style or not model:
                empty += 1
                continue
            grid[(style, model)] += 1
            style_tot[style] += 1
            model_tot[model] += 1
            leaf_c[(style, model)] += 1
        per_leaf[label] = leaf_c

    styles = sorted(style_tot)
    models = sorted(model_tot)
    w(f"\n{'#'*100}\n### DIVERSITY COVERAGE (style x model)  total_ok_rows={total}"
      f"  empty_provenance={empty}\n{'#'*100}")
    header = f"{'style \\ model':22s}" + "".join(f"{m:>14s}" for m in models) + f"{'ROW TOT':>12s}"
    w(header)
    for s in styles:
        row = f"{s:22s}" + "".join(f"{grid.get((s, m), 0):>14d}" for m in models)
        row += f"{style_tot[s]:>12d}"
        w(row)
    w(f"{'COL TOT':22s}" + "".join(f"{model_tot[m]:>14d}" for m in models)
      + f"{total - empty:>12d}")

    # coverage warnings (a cell that never fires may signal a broken alias/style)
    missing = [(s, m) for s in styles for m in models if grid.get((s, m), 0) == 0]
    if missing:
        w(f"\n⚠ EMPTY CELLS (never sampled): {missing} — expected on a small "
          f"smoke; on a full run a persistently empty cell = broken style/alias.")
    if empty:
        w(f"⚠ {empty} row(s) had EMPTY style/model provenance — STOP, fix the "
          f"generator (both are always known at gen time).")
    w(f"\nper-leaf origin counts:")
    for label in sorted(per_leaf):
        c = per_leaf[label]
        pretty = ", ".join(f"{m}:{s}={n}" for (s, m), n in sorted(c.items()))
        w(f"  {label:56s} {pretty or '<none>'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset dir / split dir / parent dir")
    ap.add_argument("--n", type=int, default=3, help="rows per leaf (default 3)")
    ap.add_argument("--split", default=None, help="only preview this split (e.g. train)")
    ap.add_argument("--out", default=None,
                    help="txt output path; default: /home/sgsilva/tmp/previews/preview_<root>.txt")
    ap.add_argument("--meta-fields", default=None,
                    help="comma-separated metadata fields for section 0 "
                         "(default: text-reasoning + judge fields)")
    ap.add_argument("--group-by", default=None,
                    help="comma-separated provenance cols (e.g. "
                         "reasoning_prompt_style,reasoning_model) — show the first "
                         "--n rows of EACH distinct cell instead of the first-n "
                         "rows overall. Covers every (style×model) diversity combo; "
                         "avoids a session-ordered leaf showing only one cell.")
    args = ap.parse_args()
    group_cols = ([c.strip() for c in args.group_by.split(",")]
                  if args.group_by else None)
    meta_fields = (tuple(f.strip() for f in args.meta_fields.split(","))
                   if args.meta_fields else _DEFAULT_META_FIELDS)

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 2

    # Default the preview into the dedicated preview dir (Sandra, 2026-06-29).
    if args.out is None:
        args.out = f"/home/sgsilva/tmp/previews/preview_{root.name}.txt"

    leaves = _discover_leaves(root)
    if not leaves:
        print(f"ERROR: no loadable dataset under {root}", file=sys.stderr)
        return 2
    if args.split:
        leaves = [(lbl, d) for lbl, d in leaves if lbl.endswith("/" + args.split) or lbl == args.split]
        if not leaves:
            print(f"ERROR: split '{args.split}' not found", file=sys.stderr)
            return 2

    buf: list[str] = []

    def w(line: str):
        print(line)
        buf.append(line)

    w(f"PREVIEW  root={root}  n={args.n}  leaves={len(leaves)}")
    for label, d in leaves:
        _render(label, d, args.n, w, meta_fields, group_cols=group_cols)

    # Diversity coverage summary (no-op unless the style/model provenance cols
    # are present) — aggregated across all leaves.
    _coverage_table(leaves, w)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(buf))
        print(f"\n>>> preview written to: {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
