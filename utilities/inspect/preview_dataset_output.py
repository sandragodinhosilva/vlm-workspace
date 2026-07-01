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
_REASONING_COLS = ("generation_reasoning", "reasoning_content", "reasoning",
                   "reasoning_trace", "think", "thinking", "cot")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _extract_reasoning(row, msgs) -> str:
    """Pull the model's reasoning from wherever it lives: a dedicated column, an
    embedded <think> block in raw output, or an assistant-message reasoning field.
    Returns '' if there is genuinely none (and the caller flags that)."""
    for c in _REASONING_COLS:
        v = row.get(c)
        if v:
            return str(v).strip()
    raw = row.get("raw_model_output", "") or ""
    m = _THINK_RE.search(raw)
    if m and m.group(1).strip():
        return m.group(1).strip()
    if msgs:
        for mm in msgs:
            if mm.get("role") == "assistant":
                for c in _REASONING_COLS:
                    if mm.get(c):
                        return str(mm[c]).strip()
                mt = _text_of(mm.get("content", ""))
                mt_m = _THINK_RE.search(mt)
                if mt_m and mt_m.group(1).strip():
                    return mt_m.group(1).strip()
    return ""


# Default KEY metadata fields surfaced in section 0 (present-AND-filled audit).
# Covers the text-reasoning + judge pipelines; override with --meta-fields.
_DEFAULT_META_FIELDS = (
    "dataset_type", "source_dataset_family", "source_version", "exercise_id",
    "exercise_name", "body_region", "is_lr_pair", "task", "correct_answer",
    "reasoning_added", "reasoning_model",
    "reasoning_judge_decision", "reasoning_regenerated", "reasoning_judge_model",
    "reasoning_repair_tags",
)


def _render(label, d, n, w, meta_fields):
    w(f"\n{'='*100}\n=== LEAF: {label}   (rows={len(d)}, cols={len(d.column_names)})\n{'='*100}")
    cols = set(d.column_names)
    missing = [c for c in ("generation_prompt", "raw_model_output") if c not in cols]
    if missing:
        w(f"!! MISSING provenance columns: {missing} — showing what IS present "
          f"(at minimum messages). Columns: {d.column_names}")
    for i in range(min(n, len(d))):
        r = d[i]
        ex = r.get("exercise_name") or r.get("exercise_id") or ""
        w(f"\n{'-'*100}\nROW {i}  exercise={ex!r}  task={r.get('task','?')!r}  variant={r.get('variant','?')!r}\n{'-'*100}")

        w("\n### 0. KEY METADATA (present-AND-filled audit) ###")
        empty = []
        for f in meta_fields:
            if f not in cols:
                continue
            v = r.get(f)
            filled = v not in (None, "", [], {})
            w(f"  {f:26s} = {v!r}" + ("" if filled else "   ⚠ EMPTY"))
            if not filled and f in ("messages", "reasoning_model", "source_version",
                                    "dataset_type"):
                empty.append(f)
        if empty:
            w(f"  ⚠ LOAD-BEARING FIELDS EMPTY: {empty} — STOP, fix the generator.")

        w("\n### 1. GENERATION PROMPT ###")
        w(r.get("generation_prompt", "<no generation_prompt column>"))

        w("\n### 2. MODEL REASONING (<think> / reasoning_content) ###")
        reasoning = _extract_reasoning(r, r.get("messages"))
        if reasoning:
            w(reasoning)
        else:
            w("<none captured — model emitted no reasoning, OR the server's reasoning_parser "
              "stripped <think> and the generator did not store reasoning_content>")

        w("\n### 3. RAW MODEL OUTPUT ###")
        w(r.get("raw_model_output", "<no raw_model_output column>"))

        w("\n### 4. PARSED OUTPUT (messages) ###")
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
    args = ap.parse_args()
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
        _render(label, d, args.n, w, meta_fields)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(buf))
        print(f"\n>>> preview written to: {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
