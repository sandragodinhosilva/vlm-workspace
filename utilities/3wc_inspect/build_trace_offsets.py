#!/usr/bin/env python3
"""Build a turn_id -> byte-offset index for a huge Langfuse trace .jsonl.

The 3WC corpus files are 22-102 GB, so the repo's own `core.entry_by_id()` cold
start walks every line. For *inspection* of a handful of turns we only need to
seek, so this writes a compact sidecar index once:

    {"<turn_id>": [offset, length], ...}

Usage:
    python3 build_trace_offsets.py <trace.jsonl> [--out <index.json>] [--resume]

The sidecar defaults to <trace>.offsets.json NEXT TO THE OUTPUT DIR the caller
chooses -- never written beside the source, because the corpus lives on a shared
read-only mount owned by another user.

Only `id` and the byte span are stored (no member content), so the index itself
is not sensitive -- but the file it points into IS.
"""
import argparse
import json
import os
import sys


def build(src, out, *, resume=False, progress_every=2_000_000):
    """Stream src, recording each row's id -> (offset, length).

    Parses only the id via a cheap prefix scan when possible, falling back to a
    full json.loads. A row whose id cannot be determined is COUNTED as skipped
    and reported -- never silently dropped.
    """
    if resume and os.path.exists(out):
        print(f"index already exists, --resume given: {out}", file=sys.stderr)
        return 0

    index, skipped, n = {}, 0, 0
    size = os.path.getsize(src)
    tmp = out + ".partial"
    with open(src, "rb") as fh:
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            n += 1
            stripped = line.strip()
            if not stripped:
                skipped += 1
                continue
            tid = None
            # Fast path: `{"id": "…"` is the first key in this export.
            if stripped.startswith(b'{"id": "'):
                end = stripped.find(b'"', 8)
                if end > 8:
                    tid = stripped[8:end].decode("utf-8", "replace")
            if tid is None:
                try:
                    tid = json.loads(stripped).get("id")
                except Exception:
                    tid = None
            if not tid:
                skipped += 1
                continue
            index[tid] = [offset, len(line)]
            if n % progress_every == 0:
                print(f"  {n:>12,d} rows  {offset / size:6.1%}  "
                      f"{len(index):,d} ids", file=sys.stderr, flush=True)

    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    os.replace(tmp, out)  # atomic: a reader never sees a half-written index
    print(f"wrote {out}\n  rows={n:,d} ids={len(index):,d} skipped={skipped:,d}")
    if skipped:
        print(f"  NOTE {skipped:,d} row(s) had no usable id — they are NOT in the index.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="exit successfully if the index already exists")
    a = ap.parse_args()
    out = a.out or os.path.join(
        "/home/sgsilva/dawn-research/3wc/exploration/eval_previews/_offsets",
        os.path.basename(a.src) + ".offsets.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return build(a.src, out, resume=a.resume)


if __name__ == "__main__":
    sys.exit(main())
