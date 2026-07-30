#!/usr/bin/env python3
"""Companion to preview_eval_jsonl.py: show the CONVERSATION behind each eval row.

preview_eval_jsonl.py answers "what did the model do and how was it scored".
This answers "what was the model looking at when it did that" -- the system
events / member messages leading up to the graded turn, the model's own output,
and the judge's per-metric reasoning, side by side for the same rows.

The conversation is NOT stored in the run dir: an eval row carries only
`turn_id`, and the transcript lives in the Langfuse trace corpus under
$DATA_DIR/raw/traces/*.jsonl (22-102 GB per file). Rather than cold-start the
repo's full index, this seeks via the sidecar built by build_trace_offsets.py.

    python3 preview_conversations.py <judged_or_generations.jsonl> \
        --offsets <trace.jsonl.offsets.json> \
        --trace   <trace.jsonl> \
        [-n 20] [--status diverge] [--full]

CONTAINS REAL MEMBER CONVERSATIONS (names, clinical detail). Local inspection
only -- never send this output to a hosted model or into a shareable doc.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preview_eval_jsonl import load, scores_of, truncate  # noqa: E402  reuse, don't duplicate


def load_offsets(path):
    """turn_id -> [offset, length]. A missing index is fatal, never a silent
    fall-through to 'no conversation found'."""
    if not os.path.exists(path):
        print(f"MISSING offsets index: {path}\n"
              f"Build it first:\n"
              f"  python3 {os.path.dirname(os.path.abspath(__file__))}"
              f"/build_trace_offsets.py <trace.jsonl>", file=sys.stderr)
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_entry(trace_path, span):
    """Seek to a recorded byte span and parse that one trace row."""
    offset, length = span
    with open(trace_path, "rb") as fh:
        fh.seek(offset)
        raw = fh.read(length)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"_parse": "parse_failed", "_error": str(exc)}


def as_text(content):
    """Langfuse content is a str or a list of {type,text} blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                out.append(b.get("text") or b.get("content") or json.dumps(b))
            else:
                out.append(str(b))
        return "\n".join(out)
    if content is None:
        return ""
    return json.dumps(content)


def messages_of(entry, *, builder=None):
    """The conversation the model saw, as [(role, text)].

    The turn's prompt is NOT stored verbatim: production assembles it from the
    entry's `output` payload (short-term events + the agent's prior scratchpad).
    So we call the repo's own `core.build_turn_messages` -- reusing it means this
    view shows what the model ACTUALLY saw, including the recorded byte-exact
    system prompt, rather than an approximation that could drift from the eval.

    Returns None when the payload cannot be reconstructed, which stays distinct
    from [] ("reconstructed, but no messages").
    """
    payload = entry.get("output")
    if payload is None:
        return None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return [("_unparsed_output", truncate(payload, 400))]
    if not isinstance(payload, dict):
        return [("_unparsed_output", truncate(json.dumps(payload), 400))]
    payload.setdefault("_turn_id", entry.get("id"))

    if builder is None:
        return [("_no_builder",
                 "core.build_turn_messages unavailable — run from the "
                 "dawn-research/3wc dir so `core` is importable.")]
    try:
        msgs = builder(payload)
    except Exception as exc:  # a reconstruction failure is reported, never faked
        return [("_reconstruct_failed", f"{type(exc).__name__}: {exc}")]
    # build_turn_messages returns (agent_name, messages) — take the messages.
    if isinstance(msgs, tuple):
        msgs = msgs[1] if len(msgs) > 1 else msgs[0]
    if not isinstance(msgs, list):
        return [("_unexpected_builder_result", truncate(str(msgs), 300))]
    out = []
    for m in msgs:
        if isinstance(m, dict):
            out.append((m.get("role") or "?", as_text(m.get("content"))))
        else:
            out.append(("_raw", truncate(str(m), 400)))
    return out


def get_builder(repo="/home/sgsilva/dawn-research/3wc"):
    """core.build_turn_messages from the repo, or None with the reason printed.

    Imported lazily: `core` pulls in the corpus index module, and we only ever
    call build_turn_messages (which needs no index), so this stays cheap.
    """
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from core.messages import build_turn_messages
    except Exception as exc:
        print(f"⚠ could not import core.build_turn_messages from {repo}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return build_turn_messages


def print_conversation(row, entry, *, width, full, builder=None):
    """One row: the conversation, then what the model did, then the judge."""
    tid = row.get("turn_id")
    print()
    print("=" * 100)
    print(f"  line {row['_lineno']}  ·  turn_id={tid}")
    print(f"  {row.get('agent')}  ·  replay_status={row.get('replay_status')}"
          f"  ·  abstain={row.get('abstain')}"
          f"  ·  deterministic={'PASS' if row.get('deterministic_pass') else 'FAIL'}")
    if entry is None:
        print("  ⚠ NO TRACE ENTRY for this turn_id in the offsets index —")
        print("    the conversation could not be resolved (wrong trace file for this era?).")
        print("=" * 100)
        return "missing"
    if entry.get("_parse") == "parse_failed":
        print(f"  ⚠ TRACE ROW UNPARSEABLE: {entry.get('_error')}")
        print("=" * 100)
        return "parse_failed"

    meta = entry.get("metadata") or {}
    member = meta.get("member") or meta.get("member_name")
    if member:
        print(f"  member: {member}")
    print("=" * 100)

    msgs = messages_of(entry, builder=builder)
    if msgs is None:
        print("\n  ── CONVERSATION ──")
        print("  (this trace entry has input=null — nothing recorded)")
    elif not msgs:
        print("\n  ── CONVERSATION ──")
        print("  (no messages)")
    else:
        print(f"\n  ── CONVERSATION the model saw ({len(msgs)} messages) ──")
        for role, text in msgs:
            text = text or ""
            label = role.upper()
            if not full:
                # System prompts are ~40k of static rules; the per-turn events
                # are what actually varies, so cap the boilerplate and say so.
                cap = 600 if role == "system" else 1800
                if len(text) > cap:
                    text = text[:cap] + f"\n      …[{len(text) - cap:,d} more chars truncated]"
            print(f"\n  [{label}]")
            for line in text.splitlines() or [""]:
                while len(line) > width:
                    print(f"      {line[:width]}")
                    line = line[width:]
                print(f"      {line}")

    print(f"\n  ── WHAT THE MODEL DID ──")
    gen = row.get("gen_text") or "(empty)"
    if not full and len(gen) > 2500:
        gen = gen[:2500] + f"\n      …[{len(gen) - 2500:,d} more chars truncated]"
    for line in gen.splitlines() or [""]:
        while len(line) > width:
            print(f"      {line[:width]}")
            line = line[width:]
        print(f"      {line}")
    for c in (row.get("calls") or []):
        print(f"\n      → TOOL CALL: {c.get('name')}")
        for k, v in (c.get("args") or {}).items():
            s = v if isinstance(v, str) else json.dumps(v)
            print(f"          {k}: {truncate(s, 400 if not full else 100_000)}")
    if not (row.get("calls") or []):
        print("\n      → TOOL CALL: (none — abstained or message-only)")

    sc = scores_of(row)
    judges = row.get("judges") or []
    if sc or judges:
        print(f"\n  ── HOW THE JUDGE SCORED IT ──")
        for j in judges:
            v = j.get("verdict") or {}
            if j.get("error"):
                print(f"      judge {j.get('model')}: ERROR {truncate(j['error'], 200)}")
                continue
            print(f"      judge: {j.get('model')}   safety_gate={v.get('safety_gate', '-')}")
            pre = v.get("pre_scoring_analysis")
            if isinstance(pre, dict):
                for k, val in pre.items():
                    print(f"        · {k}: {truncate(as_text(val), 300)}")
            for metric, val in v.items():
                if isinstance(val, dict) and "score" in val:
                    print(f"        {metric:<30} {val.get('score')}  "
                          f"{truncate(val.get('comment') or '', 220)}")
    return "ok"


def main():
    ap = argparse.ArgumentParser(
        description="Show the conversation behind each 3WC eval row.")
    ap.add_argument("path", help="judged.jsonl or generations.jsonl")
    ap.add_argument("--trace", required=True, help="the trace .jsonl the turns came from")
    ap.add_argument("--offsets", default=None,
                    help="offsets sidecar (default: alongside eval_previews/_offsets)")
    ap.add_argument("-n", type=int, default=20, help="rows to show (default 20)")
    ap.add_argument("--agent")
    ap.add_argument("--status")
    ap.add_argument("--grep")
    ap.add_argument("--full", action="store_true",
                    help="do not truncate system prompts / gen_text / tool args")
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--repo", default="/home/sgsilva/dawn-research/3wc",
                    help="dawn-research/3wc dir, for core.build_turn_messages")
    a = ap.parse_args()

    offsets_path = a.offsets or os.path.join(
        "/home/sgsilva/dawn-research/3wc/exploration/eval_previews/_offsets",
        os.path.basename(a.trace) + ".offsets.json")
    offsets = load_offsets(offsets_path)
    if offsets is None:
        return 2

    rows, bad, scanned, truncated, _ = load(
        a.path, a.n, agent=a.agent, status=a.status, grep=a.grep)
    if not rows:
        print(f"NO ROWS matched in {a.path} (scanned {scanned:,d})", file=sys.stderr)
        return 1

    print(f"eval file : {a.path}")
    print(f"trace file: {a.trace}")
    print(f"offsets   : {offsets_path}  ({len(offsets):,d} ids)")
    print(f"showing   : {len(rows):,d} rows")
    if bad:
        print(f"⚠ {len(bad):,d} corrupt line(s) skipped in the scanned range "
              f"(lines {', '.join(str(b['_lineno']) for b in bad[:5])})")
    if truncated:
        print(f"⚠ scan stopped at {scanned:,d} rows — not the whole file")

    builder = get_builder(a.repo)
    stats = {"ok": 0, "missing": 0, "parse_failed": 0}
    for r in rows:
        span = offsets.get(r.get("turn_id"))
        entry = read_entry(a.trace, span) if span else None
        stats[print_conversation(r, entry, width=a.width, full=a.full,
                                 builder=builder)] += 1

    print()
    print("=" * 100)
    print(f"  resolved {stats['ok']}/{len(rows)} conversations"
          + (f"   ⚠ {stats['missing']} turn_id(s) not in this trace file" if stats["missing"] else "")
          + (f"   ⚠ {stats['parse_failed']} unparseable trace row(s)" if stats["parse_failed"] else ""))
    print("=" * 100)
    # A run whose turns mostly do not resolve means the WRONG trace file — say so
    # rather than returning 0 on a near-empty report.
    return 0 if stats["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
