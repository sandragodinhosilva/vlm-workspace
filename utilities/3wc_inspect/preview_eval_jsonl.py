#!/usr/bin/env python3
"""Preview the first N rows of a 3WC eval .jsonl (generations / judged).

Rows carry a ~3k gen_text and, for judged.jsonl, a ~44k `judges` blob, so a raw
dump is unreadable. This prints a per-row compact record plus a distribution
summary over the previewed slice.

    python3 preview_eval_jsonl.py <file.jsonl> [-n 100] [--full-text ROW]

CONTAINS REAL MEMBER CONVERSATION CONTENT -- local inspection only, never send
the output to a hosted model or paste it into a shareable doc.
"""
import argparse
import json
import os
import sys
from collections import Counter


def truncate(text, width):
    """One-line, width-capped preview of a possibly-multiline string."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def load(path, n, *, agent=None, status=None, grep=None, scan_limit=2_000_000):
    """Collect up to n rows matching the filters, streaming (files are ~1 GB).

    Distinct sentinels: a bad line is recorded as parse_failed / empty_line and
    counted, never skipped silently and never defaulted to an empty row.

    Filters apply to the SCANNED rows, so `-n 100 --status diverge` means "the
    first 100 diverge rows", not "diverge rows among the first 100". scan_limit
    bounds the read so a filter matching nothing cannot walk 16k heavy rows
    forever; a truncated scan is reported, never silently treated as exhaustive.
    """
    rows, bad, scanned, truncated = [], [], 0, False
    filtering = bool(agent or status or grep)
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if len(rows) >= n:
                break
            if scanned >= scan_limit:
                truncated = True
                break
            scanned += 1
            line = line.strip()
            if not line:
                bad.append({"_parse": "empty_line", "_lineno": i + 1})
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                # Corrupt rows are tracked SEPARATELY from matches. Counting them
                # as matches would let a filter that matches nothing fill its
                # quota with unrelated parse errors and read as success.
                bad.append({"_parse": "parse_failed", "_error": str(exc),
                            "_lineno": i + 1})
                continue
            if agent and row.get("agent") != agent:
                continue
            if status and row.get("replay_status") != status:
                continue
            if grep and grep.lower() not in (row.get("gen_text") or "").lower():
                continue
            row["_parse"] = "ok"
            row["_lineno"] = i + 1
            rows.append(row)
    # When filtering, a bad line is only reported if it could have matched --
    # unknowable for a row that never parsed, so report the count, not a verdict.
    return rows, bad, scanned, truncated, filtering


def list_run(run_dir):
    """Summarize every .jsonl in a run dir: size, rows, judged-or-not."""
    import glob
    import os
    # Also match .jsonl.bak / .jsonl.pre_*.bak — a stale backup sitting next to
    # a live file is exactly what you need to SEE (so you don't preview the
    # wrong one), so list it labeled rather than globbing it away.
    paths = sorted(set(glob.glob(os.path.join(run_dir, "*.jsonl"))
                       + glob.glob(os.path.join(run_dir, "*.jsonl.*"))))
    if not paths:
        print(f"no .jsonl files in {run_dir}", file=sys.stderr)
        return 1
    print(f"{'file':<44} {'size':>10} {'rows':>8}  judged?")
    print("-" * 78)
    for p in paths:
        size = os.path.getsize(p)
        with open(p, encoding="utf-8") as fh:
            rows = sum(1 for _ in fh)
            fh.seek(0)
            first = fh.readline()
        judged = "judges" in first[:200000] if first else False
        note = "judged" if judged else "-"
        if p.endswith(".bak"):
            note += "  (BACKUP)"
        print(f"{os.path.basename(p):<44} {size / 1e6:>9,.1f}M {rows:>8,d}  {note}")
    return 0


def scores_of(row):
    """Flatten judge_scores {judge: {metric: v}} -> {metric: v} for the judge
    named by judge_key, falling back to the sole judge when unambiguous."""
    js = row.get("judge_scores")
    if not isinstance(js, dict) or not js:
        return {}
    key = row.get("judge_key")
    if key in js and isinstance(js[key], dict):
        return js[key]
    if len(js) == 1:
        only = next(iter(js.values()))
        return only if isinstance(only, dict) else {}
    return {}


def main():
    ap = argparse.ArgumentParser(
        description="Preview rows of a 3WC eval .jsonl (generations / judged).")
    ap.add_argument("path", help="a .jsonl file, or a run dir with --list")
    ap.add_argument("-n", type=int, default=100, help="rows to preview (default 100)")
    ap.add_argument("--list", action="store_true",
                    help="PATH is a run dir: list its .jsonl files and row counts")
    ap.add_argument("--full-text", type=int, metavar="ROW",
                    help="print row ROW's full gen_text + judge verdicts, then exit")
    ap.add_argument("--agent", help="only rows with this agent (e.g. zero_to_chat)")
    ap.add_argument("--status", help="only rows with this replay_status "
                                     "(matched_all/over_action/diverge/abstained/stopped)")
    ap.add_argument("--grep", help="only rows whose gen_text contains this (case-insensitive)")
    ap.add_argument("--width", type=int, default=110, help="gen_text preview width")
    args = ap.parse_args()

    if args.list:
        return list_run(args.path)

    rows, bad, scanned, truncated, filtering = load(
        args.path, args.n, agent=args.agent, status=args.status, grep=args.grep)
    if not rows:
        filt = ", ".join(f"{k}={v}" for k, v in
                         (("agent", args.agent), ("status", args.status),
                          ("grep", args.grep)) if v)
        print(f"NO ROWS matched in {args.path}"
              + (f" (filters: {filt}; scanned {scanned:,d} rows)" if filt else ""),
              file=sys.stderr)
        return 1
    if truncated:
        print(f"!! scan limit hit after {scanned:,d} rows — "
              f"results are NOT the whole file", file=sys.stderr)

    if args.full_text is not None:
        match = [r for r in rows if r.get("_lineno") == args.full_text]
        if not match:
            print(f"row {args.full_text} not in the first {args.n}", file=sys.stderr)
            return 1
        row = match[0]
        print(f"===== row {args.full_text} | turn_id={row.get('turn_id')} =====")
        print(f"model={row.get('model')}  agent={row.get('agent')}  "
              f"replay_status={row.get('replay_status')}")
        print("\n----- gen_text -----")
        print(row.get("gen_text") or "(empty)")
        print("\n----- calls -----")
        print(json.dumps(row.get("calls"), indent=2)[:4000])
        for j in row.get("judges") or []:
            print(f"\n----- judge: {j.get('model')} -----")
            print(json.dumps(j.get("verdict"), indent=2)[:6000])
        return 0

    ok = rows  # load() already excludes unparseable lines from matches
    W = args.width

    # ---------- header ----------
    print("=" * 100)
    print(f"  {os.path.basename(args.path)}")
    print("=" * 100)
    filt = ", ".join(f"{k}={v}" for k, v in
                     (("agent", args.agent), ("status", args.status),
                      ("grep", args.grep)) if v)
    print(f"  path      {args.path}")
    print(f"  showing   {len(ok):,d} rows"
          + (f"  (matching {filt}, out of {scanned:,d} scanned)" if filt
             else f"  (the first {len(ok):,d} in the file)"))
    if bad:
        lines = ", ".join(str(b["_lineno"]) for b in bad[:8])
        more = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
        print(f"  ⚠ CORRUPT {len(bad):,d} line(s) in the scanned range could not be parsed:")
        print(f"            lines {lines}{more}")
        print(f"            these are NOT counted above — the file has genuinely bad rows.")
    if truncated:
        print(f"  ⚠ PARTIAL scan stopped at {scanned:,d} rows — NOT the whole file.")

    # ---------- summary ----------
    print()
    print("-" * 100)
    print("  SUMMARY OF THESE ROWS")
    print("-" * 100)
    for field in ("model", "agent", "mode"):
        vals = Counter(str(r.get(field)) for r in ok)
        shown = ", ".join(f"{v} ({c})" for v, c in vals.most_common(4))
        print(f"  {field:<20} {shown}")
        if len(vals) == 1 and len(ok) > 1:
            pass  # single-valued: the slice is homogeneous, worth noticing
    print()
    vals = Counter(str(r.get("replay_status")) for r in ok)
    print("  replay_status        how the model's action compared to the recorded one")
    for v, c in vals.most_common():
        bar = "█" * max(1, round(40 * c / max(1, len(ok))))
        print(f"    {v:<16} {c:>4}  {100 * c / max(1, len(ok)):>5.1f}%  {bar}")
    print()
    for field, good in (("deterministic_pass", "True"), ("matched_all", "True"),
                        ("abstain", "True")):
        vals = Counter(str(r.get(field)) for r in ok)
        print(f"  {field:<20} " + "  ".join(f"{v}={c}" for v, c in vals.most_common()))
    errs = [r for r in ok if r.get("error") is not None]
    print(f"  {'error != null':<20} {len(errs)}"
          + (f"   e.g. {truncate(errs[0].get('error'), 60)}" if errs else ""))
    empty = [r for r in ok if not (r.get("gen_text") or "").strip()]
    print(f"  {'gen_text empty':<20} {len(empty)}")

    metrics = sorted({m for r in ok for m in scores_of(r)} - {"avg"})
    if metrics:
        print()
        print("  JUDGE SCORES (mean over these rows; 1-5)")
        for m in metrics + ["avg"]:
            got = [scores_of(r)[m] for r in ok
                   if isinstance(scores_of(r).get(m), (int, float))]
            if not got:
                print(f"    {m:<30} n=0   (no numeric scores)")
                continue
            mean = sum(got) / len(got)
            bar = "█" * round(20 * (mean - 1) / 4) if mean >= 1 else ""
            flat = "  ← FLAT (no variance: metric may not discriminate)" \
                if min(got) == max(got) and len(got) > 5 else ""
            print(f"    {m:<30} {mean:>5.2f}  [{min(got)}-{max(got)}]  n={len(got):<4} "
                  f"{bar}{flat}")

    # ---------- per-row detail ----------
    print()
    print("-" * 100)
    print("  ROWS")
    print("-" * 100)
    for r in ok:
        sc = scores_of(r)
        flags = []
        if not r.get("deterministic_pass"):
            flags.append("DET-FAIL")
        if r.get("error") is not None:
            flags.append("ERROR")
        if not (r.get("gen_text") or "").strip():
            flags.append("EMPTY-GEN")
        flag_s = ("  ⚠ " + " ".join(flags)) if flags else ""

        print()
        print(f"  ┌─ line {r['_lineno']}  ·  {r.get('agent')}  ·  "
              f"{r.get('replay_status')}{flag_s}")
        meta = [f"turn_id={r.get('turn_id')}"]
        if r.get("n_steps") is not None:
            meta.append(f"steps={r.get('n_steps')}")
        if r.get("diverge_step") is not None:
            meta.append(f"diverge_step={r.get('diverge_step')}")
        meta.append(f"abstain={r.get('abstain')}")
        print(f"  │  {'  '.join(meta)}")
        if sc:
            avg = sc.get("avg")
            per = "  ".join(f"{m.split('_')[0][:6]}={sc.get(m, '-')}" for m in metrics)
            print(f"  │  judge: avg={avg if avg is not None else '-'}   {per}")
        # tool calls the model actually made — the decision, in one line each
        for c in (r.get("calls") or []):
            print(f"  │  → CALL {c.get('name')}({', '.join((c.get('args') or {}).keys())})")
        if not (r.get("calls") or []):
            print(f"  │  → CALL (none — abstained / message only)")
        body = truncate(r.get("gen_text") or "(empty)", W * 3)
        for i in range(0, len(body), W):
            print(f"  │  {body[i:i + W]}")
        print(f"  └─")
    return 0


if __name__ == "__main__":
    sys.exit(main())
