#!/usr/bin/env python3
"""Convert an HF SFT/eval dataset (load_from_disk) into an app-compatible browse
JSONL for the video-sft-vlm gradio app (point the app at it via DEFAULT_JSONL=...).

This is the GENERIC converter — point it at any HF dataset that carries the
canonical thrive-vlm columns (messages + video_frames/image + fps/need_to_flip/
session_id/exercise_id/rep_index). It writes one JSONL row per example: the row's
columns verbatim + a `metadata` dict the app reads for its fps/flip/video_id caches
(app.py load_jsonl_samples reads metadata-first, else top-level). It is defensive —
only echoes columns that actually exist, so it stays correct for ANY dataset (a row
without session_id/exercise_id just omits those keys rather than fabricating "").

OUTPUT LOCATION RULE: write app browse datasets to
  /mnt/data/sgsilva/datasets/app_video_datasets/<name>_browse.jsonl
(see that dir's README). NEVER leave them in /tmp or /mnt/data/sgsilva/tmp.

WHAT IT HANDLES:
  - messages kept as a list (app re-parses string OR list content).
  - [VISUAL OBSERVATIONS] answer blocks: parsed into metadata.answers[] so the app's
    render_vo_block shows them; absent for MCQA/plain SFT (routes through normal render).
  - --old-reas-from <HF dataset>: optional join of a SOURCE reasoning_trace onto
    metadata.old_reas_trace by (exercise_id, example_id/session_id, question_idx,
    repetition/rep_index) — used for the 1805_binary set whose source trace was
    dropped from the SFT target but is still worth eyeballing.

USAGE:
  /home/sgsilva/vlm-post-training-home-venv/bin/python \
    /home/sgsilva/utilities/apps/scripts/make_app_video_dataset.py \
    --source /mnt/data/sgsilva/datasets/<hf_dataset> \
    --name   <name>           # -> /mnt/data/sgsilva/datasets/app_video_datasets/<name>_browse.jsonl

  # with the optional source-trace join (1805_binary style):
  ... --source <converted_hf> --name 1805_binary_converted \
      --old-reas-from /mnt/data/vramos/.../1805_binary_train_no_video/dataset

Then:
  cd /home/sgsilva/video-sft-vlm && source /home/sgsilva/video-sft-vlm-home-venv/bin/activate
  DEFAULT_JSONL=/mnt/data/sgsilva/datasets/app_video_datasets/<name>_browse.jsonl python app.py --port 7863
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_from_disk, DatasetDict

APP_DIR = "/mnt/data/sgsilva/datasets/app_video_datasets"
_NUM_RE = re.compile(r"^\s*(\d+)\.\s*(.*\S)\s*$")


def load_split(path, prefer=("test", "train")):
    ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        for s in prefer:
            if s in ds:
                return ds[s]
        return ds[next(iter(ds.keys()))]
    return ds


def _question_stems(user_content):
    stems = {}
    for line in (user_content or "").splitlines():
        m = _NUM_RE.match(line)
        if m:
            qn = int(m.group(1)) - 1
            stems.setdefault(qn, m.group(2).split("(pick one")[0].strip())
    return stems


def _parse_vo_answers(assistant_content, user_content):
    """Parse a [VISUAL OBSERVATIONS] answer block into render_vo_block's shape.
    Returns (answers, vo_block_text); answers=None when no VO block (distinct from [])."""
    if "[VISUAL OBSERVATIONS]" not in (assistant_content or ""):
        return None, ""
    block = assistant_content.split("</think>", 1)[-1]
    vo_block = block[block.index("[VISUAL OBSERVATIONS]"):].strip()
    stems = _question_stems(user_content)
    answers = []
    for line in vo_block.splitlines():
        m = _NUM_RE.match(line)
        if not m:
            continue
        qi = int(m.group(1)) - 1
        answers.append({
            "schema_question_index": qi, "chosen_option": m.group(2).strip(),
            "answered": True, "question_text": stems.get(qi, ""),
            "tier": "", "reason": "", "verification": {},
        })
    return answers, vo_block


def _build_old_reas_index(path):
    """Index a SOURCE dataset's reasoning_trace by true content-id for the optional
    metadata.old_reas_trace join. Key = (exercise_id, example_id, question_idx, repetition)."""
    src = load_split(path)
    if "reasoning_trace" not in src.column_names:
        print(f"ERROR: --old-reas-from has no reasoning_trace column: {path}", file=sys.stderr)
        sys.exit(1)
    idx = {}
    for r in src:
        k = (str(r.get("exercise_id")), str(r.get("example_id")),
             int(r["question_idx"]), int(r["repetition"]))
        idx[k] = r["reasoning_trace"] or ""
    return idx


def _to_record(row, cols, *, old_reas, echo_to_metadata, counters):
    """Build one app browse record: the row's columns verbatim + a `metadata` dict
    the app reads. Single source of truth for BOTH the full dump and the stratified
    subset, so they never diverge. `counters` is a dict mutated in place for the report."""
    rec = {c: row[c] for c in cols}
    fps, flip = row.get("fps"), row.get("need_to_flip")
    if fps is None:
        counters["missing_fps"] += 1  # distinct count, never silently defaulted

    meta = {"fps": fps, "need_to_flip": flip}
    if row.get("session_id") is not None:
        meta["video_id"] = row.get("session_id")
    if row.get("exercise_id") is not None:
        meta["exercise_code"] = str(row.get("exercise_id"))
    if row.get("rep_index") is not None:
        meta["rep_index"] = row.get("rep_index")

    # --echo-to-metadata: place named provenance columns UNDER metadata (only those
    # that exist in the source — don't fabricate). Mirrors the old merged_v2 builder.
    for c in echo_to_metadata:
        if c in cols:
            meta[c] = row.get(c)

    # optional source-trace join (distinct sentinel on miss, never blank-as-match)
    if old_reas is not None:
        k = (str(row.get("exercise_id")), str(row.get("session_id")),
             int(row.get("question_index", -1)), int(row.get("rep_index", -1)))
        t = old_reas.get(k)
        if t is None:
            meta["old_reas_trace"] = "OLDREAS_KEY_UNMATCHED"
            counters["old_reas_unmatched"] += 1
        else:
            meta["old_reas_trace"] = t
            if t.strip():
                counters["old_reas_matched"] += 1

    # VISUAL OBSERVATIONS answer block (only when present)
    msgs = row.get("messages") if isinstance(row.get("messages"), list) else []
    asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    usr = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    answers, vo_block = _parse_vo_answers(asst, usr)
    if answers is not None:
        meta["answers"], meta["vo_block"] = answers, vo_block
        counters["vo"] += 1

    rec["metadata"] = meta
    return rec


def _stratified_indices(ds, per_stratum, force_prov=20):
    """Pick a small, representative subset (mirrors the old merged_v2 subset builder):
    stratify by (schema, multiQ/perQ from question_index, description_tier), round-robin
    by exercise_id within each cell, then force-include up to `force_prov` rows carrying
    reasoning_regenerated_v2 so that provenance feature is visible. Returns sorted indices.
    Requires columns schema/question_index/description_tier/exercise_id/reasoning_regenerated_v2."""
    need = ["schema", "question_index", "description_tier", "exercise_id",
            "reasoning_regenerated_v2"]
    missing = [c for c in need if c not in ds.column_names]
    if missing:
        print(f"ERROR: --stratify needs columns {missing} (absent in source)", file=sys.stderr)
        sys.exit(1)
    schema = ds["schema"]; qi = ds["question_index"]; tier = ds["description_tier"]
    exid = ds["exercise_id"]; regen = ds["reasoning_regenerated_v2"]

    buckets = defaultdict(list); prov_idx = []
    for i in range(len(ds)):
        form = "multiQ" if int(qi[i]) == -1 else "perQ"
        buckets[(schema[i], form, tier[i])].append(i)
        if regen[i]:
            prov_idx.append(i)

    chosen = []; seen = set()
    for key, idxs in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        by_ex = defaultdict(list)
        for i in idxs:
            by_ex[exid[i]].append(i)
        order = sorted(by_ex.keys(), key=str); picked = []; pos = 0
        while len(picked) < per_stratum and order:
            e = order[pos % len(order)]
            if by_ex[e]:
                picked.append(by_ex[e].pop(0)); pos += 1
            else:
                order.remove(e)
        for i in picked:
            if i not in seen:
                seen.add(i); chosen.append(i)

    n_prov = 0
    for i in prov_idx:
        if n_prov >= force_prov:
            break
        if i not in seen:
            seen.add(i); chosen.append(i); n_prov += 1

    chosen.sort()
    print(f"  strata cells         : {len(buckets)} (per-stratum {per_stratum})")
    print(f"  provenance rows incl.: {n_prov}")
    print(f"  distinct exercises   : {len(set(exid[i] for i in chosen))}")
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="HF dataset dir (load_from_disk)")
    ap.add_argument("--name", required=True,
                    help="output stem; writes <APP_DIR>/<name>_browse.jsonl")
    ap.add_argument("--out", default=None, help="override full output path")
    ap.add_argument("--max-samples", type=int, default=None, help="cap rows (quick look)")
    ap.add_argument("--old-reas-from", default=None,
                    help="optional SOURCE HF dataset to join reasoning_trace onto "
                         "metadata.old_reas_trace (1805_binary style)")
    ap.add_argument("--echo-to-metadata", default="",
                    help="comma-separated source columns to ALSO place under metadata "
                         "(e.g. reasoning_regenerated_v2,judge_feedback,prev_reasoning_trace)")
    ap.add_argument("--stratify", action="store_true",
                    help="write a small stratified subset instead of the full dump "
                         "(schema x multiQ/perQ x tier, round-robin by exercise_id)")
    ap.add_argument("--per-stratum", type=int, default=12,
                    help="rows per stratum cell when --stratify (default 12)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path(APP_DIR) / f"{args.name}_browse.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_split(args.source)
    n_total = len(ds)
    cols = ds.column_names
    echo_cols = [c.strip() for c in args.echo_to_metadata.split(",") if c.strip()]
    old_reas = _build_old_reas_index(args.old_reas_from) if args.old_reas_from else None

    # row order: stratified subset, or sequential (optionally capped).
    if args.stratify:
        indices = _stratified_indices(ds, args.per_stratum)
    else:
        limit = args.max_samples or n_total
        indices = range(min(limit, n_total))

    counters = {"missing_fps": 0, "vo": 0, "old_reas_matched": 0, "old_reas_unmatched": 0}
    written = 0
    with open(out_path, "w") as f:
        for i in indices:
            rec = _to_record(ds[i], cols, old_reas=old_reas,
                             echo_to_metadata=echo_cols, counters=counters)
            f.write(json.dumps(rec) + "\n")
            written += 1

    print(f"source rows total      : {n_total}")
    print(f"rows written           : {written}")
    print(f"  missing fps          : {counters['missing_fps']}")
    print(f"  with VO answers      : {counters['vo']}")
    if echo_cols:
        print(f"  echoed to metadata   : {echo_cols}")
    if old_reas is not None:
        print(f"  old_reas matched     : {counters['old_reas_matched']}")
        print(f"  old_reas UNMATCHED   : {counters['old_reas_unmatched']}")
    print(f"OUT: {out_path}")
    on_disk = sum(1 for _ in open(out_path))
    if on_disk != written:
        print(f"ERROR: on-disk lines {on_disk} != written {written}", file=sys.stderr)
        sys.exit(1)
    print(f"verified on-disk lines : {on_disk}")


if __name__ == "__main__":
    main()
