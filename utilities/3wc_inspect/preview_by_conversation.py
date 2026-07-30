#!/usr/bin/env python3
"""Group a 3WC eval run BY CONVERSATION: turn - eval - turn - eval - ...

The other two views are row-ordered, so one member's turns are scattered across
the file. This one reassembles a conversation: every graded turn for one
(account, program_uuid), in timestamp order, each turn immediately followed by
its eval (decision + judge scores/comments).

    python3 preview_by_conversation.py <judged.jsonl> --turns <turns_conv.jsonl> \
        [--conversation <account>] [--member "Name"] [--limit-convs 3] [--full]

Conversation identity comes from the TURNS file (account + program_uuid); eval
rows carry only turn_id, so the turns file is the join key and is REQUIRED.

CONTAINS REAL MEMBER CONVERSATIONS (names, clinical detail). Local inspection
only -- never send this output to a hosted model or into a shareable doc.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preview_eval_jsonl import scores_of, truncate  # noqa: E402  reuse the readers
from preview_conversations import load_offsets, read_entry  # noqa: E402


def load_turns(path):
    """turn_id -> turn metadata (account, program_uuid, member, timestamp, ...)."""
    turns, bad = {}, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            tid = t.get("turn_id")
            if tid:
                turns[tid] = t
            else:
                bad += 1
    return turns, bad


def load_evals(path, turns):
    """turn_id -> [eval rows]. One turn can have SEVERAL evals (different models,
    sample_index, or judge_key), which is exactly what we want to show together.

    Rows whose turn_id is absent from the turns file are counted as `orphans`
    rather than dropped silently -- an orphan means the wrong turns file.
    """
    by_turn, bad, orphans, n = defaultdict(list), 0, 0, 0
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            n += 1
            r["_lineno"] = i + 1
            tid = r.get("turn_id")
            if tid not in turns:
                orphans += 1
                continue
            by_turn[tid].append(r)
    return by_turn, n, bad, orphans


def wrap(text, width, indent):
    """Print text wrapped to width, prefixed with indent, preserving newlines."""
    for line in (text or "").splitlines() or [""]:
        while len(line) > width:
            print(f"{indent}{line[:width]}")
            line = line[width:]
        print(f"{indent}{line}")


def recorded_calls(entry_out):
    """Phoenix's OWN recorded tool-call names for this turn -- the ground truth
    `replay_status` is measured against (core/checks.py: matched_all / stopped /
    diverge / abstained / over_action all compare the model's call-name sequence
    to THIS). Without it, `diverge` and `over_action` are uninterpretable.

    Returns None when it cannot be reconstructed, distinct from [] ("Phoenix
    genuinely did nothing this turn", which is what over_action compares to).
    """
    if not entry_out:
        return None
    try:
        sys.path.insert(0, "/home/sgsilva/dawn-research/3wc")
        import core
        msgs = [m for _, m in core._agent_msgs_to_timeline(
            core.this_turn_reasoning(entry_out))]
    except Exception:
        return None
    names = []
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            fn = (tc.get("function") or {})
            if fn.get("name"):
                names.append(fn["name"])
    return names


def print_eval(r, *, width, full, metrics, recorded=None):
    """The eval half: what the model did on this turn, and how it scored."""
    tag = f"EVAL  model={r.get('model')}"
    if r.get("sample_index") is not None:
        tag += f"  sample={r.get('sample_index')}"
    if r.get("judge_key"):
        tag += f"  judge={r.get('judge_key')}"
    print(f"    ┌── {tag}   (line {r['_lineno']})")

    verdict = [f"replay_status={r.get('replay_status')}",
               f"matched_all={r.get('matched_all')}",
               f"abstain={r.get('abstain')}",
               f"det={'PASS' if r.get('deterministic_pass') else 'FAIL'}"]
    if r.get("diverge_step") is not None:
        verdict.append(f"diverge_step={r.get('diverge_step')}")
    print(f"    │  {'  '.join(verdict)}")
    # The comparison replay_status is actually made against, spelled out.
    model_names = [c.get("name") for c in (r.get("calls") or [])]
    if recorded is None:
        rec_s = "(could not reconstruct)"
    elif not recorded:
        rec_s = "(none — Phoenix took no action)"
    else:
        rec_s = " → ".join(recorded)
    print(f"    │  PHOENIX did : {rec_s}")
    print(f"    │  MODEL  did  : "
          f"{' → '.join(model_names) if model_names else '(none — abstained)'}")
    if r.get("error") is not None:
        print(f"    │  ⚠ ERROR: {truncate(r.get('error'), 200)}")

    sc = scores_of(r)
    if sc:
        avg = sc.get("avg")
        per = "  ".join(f"{m.split('_')[0][:6]}={sc.get(m, '-')}" for m in metrics)
        print(f"    │  judge avg={avg if avg is not None else '-'}   {per}")

    for c in (r.get("calls") or []):
        print(f"    │  → {c.get('name')}")
        for k, v in (c.get("args") or {}).items():
            s = v if isinstance(v, str) else json.dumps(v)
            print(f"    │      {k}: {truncate(s, 100_000 if full else 200)}")
    if not (r.get("calls") or []):
        print(f"    │  → (no tool call — abstained or message only)")

    gen = r.get("gen_text") or "(empty)"
    if not full and len(gen) > 700:
        gen = gen[:700] + f" …[{len(gen) - 700:,d} more chars]"
    print(f"    │  reasoning:")
    wrap(gen, width, "    │    ")

    # judge comments -- the "why this score" that a bare number cannot give
    for j in (r.get("judges") or []):
        v = j.get("verdict") or {}
        if j.get("error"):
            print(f"    │  judge {j.get('model')}: ERROR {truncate(j['error'], 160)}")
            continue
        for metric, val in v.items():
            if isinstance(val, dict) and "score" in val:
                print(f"    │    {metric:<28} {val.get('score')}  "
                      f"{truncate(val.get('comment') or '', 100_000 if full else 200)}")
    print(f"    └──")


def main():
    ap = argparse.ArgumentParser(
        description="Group a 3WC eval run by conversation: turn - eval - turn - eval.")
    ap.add_argument("path", help="judged.jsonl or generations.jsonl")
    ap.add_argument("--turns", required=True,
                    help="the turns file the eval set was built from "
                         "(e.g. .../eval_usersplit_0907/turns_conv.jsonl)")
    ap.add_argument("--conversation", help="only this account uuid (or a unique prefix)")
    ap.add_argument("--member", help="only conversations whose member matches this (substring)")
    ap.add_argument("--limit-convs", type=int, default=3,
                    help="max conversations to print (default 3; 0 = all)")
    ap.add_argument("--min-turns", type=int, default=1,
                    help="skip conversations with fewer graded turns than this")
    ap.add_argument("--full", action="store_true", help="do not truncate")
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--trace", default=None,
                    help="trace .jsonl — enables the 'PHOENIX did' ground-truth line")
    ap.add_argument("--offsets", default=None,
                    help="offsets sidecar for --trace (default: eval_previews/_offsets)")
    a = ap.parse_args()

    turns, bad_turns = load_turns(a.turns)
    if not turns:
        print(f"NO TURNS loaded from {a.turns}", file=sys.stderr)
        return 2
    by_turn, n_eval, bad, orphans = load_evals(a.path, turns)
    if not by_turn:
        print(f"NO EVAL ROWS joined to {a.turns} — wrong turns file for this run?\n"
              f"  eval rows read: {n_eval:,d}   orphans (turn_id not in turns): {orphans:,d}",
              file=sys.stderr)
        return 1

    # group graded turns by conversation
    convs = defaultdict(list)
    for tid in by_turn:
        t = turns[tid]
        convs[(t.get("account"), t.get("program_uuid"))].append(tid)

    def member_of(key):
        for tid in convs[key]:
            m = turns[tid].get("member")
            if m:
                return m
        return "(unknown)"

    keys = list(convs)
    if a.conversation:
        keys = [k for k in keys if k[0] and str(k[0]).startswith(a.conversation)]
    if a.member:
        keys = [k for k in keys if a.member.lower() in member_of(k).lower()]
    keys = [k for k in keys if len(convs[k]) >= a.min_turns]
    if not keys:
        print(f"NO CONVERSATIONS matched "
              f"(conversation={a.conversation} member={a.member} "
              f"min-turns={a.min_turns})", file=sys.stderr)
        return 1
    # longest first: the multi-turn ones are what this view is for
    keys.sort(key=lambda k: -len(convs[k]))
    total_matched = len(keys)
    if a.limit_convs:
        keys = keys[:a.limit_convs]

    offsets = None
    if a.trace:
        offsets_path = a.offsets or os.path.join(
            "/home/sgsilva/dawn-research/3wc/exploration/eval_previews/_offsets",
            os.path.basename(a.trace) + ".offsets.json")
        offsets = load_offsets(offsets_path)
        if offsets is None:
            return 2

    metrics = sorted({m for rs in by_turn.values() for r in rs
                      for m in scores_of(r)} - {"avg"})

    print(f"eval file : {a.path}")
    print(f"turns file: {a.turns}")
    print(f"joined    : {sum(len(v) for v in by_turn.values()):,d} eval rows on "
          f"{len(by_turn):,d} turns across {len(convs):,d} conversations")
    if orphans:
        print(f"⚠ {orphans:,d} eval row(s) had a turn_id NOT in the turns file "
              f"(wrong turns file, or a mixed run)")
    if bad or bad_turns:
        print(f"⚠ unparseable lines skipped: {bad:,d} in eval, {bad_turns:,d} in turns")
    print(f"showing   : {len(keys)} of {total_matched} matching conversations "
          f"(longest first)")

    for key in keys:
        tids = sorted(convs[key], key=lambda t: turns[t].get("timestamp") or "")
        n_ev = sum(len(by_turn[t]) for t in tids)
        print()
        print("█" * 100)
        print(f"  CONVERSATION  member={member_of(key)}")
        print(f"  account={key[0]}")
        print(f"  program={key[1]}")
        print(f"  {len(tids)} graded turns · {n_ev} eval rows")
        print("█" * 100)

        for i, tid in enumerate(tids, 1):
            t = turns[tid]
            print()
            print(f"  ══ TURN {i}/{len(tids)} ══ {t.get('timestamp')}  "
                  f"agent={t.get('agent')}  type={t.get('turn_type')}")
            print(f"     turn_id={tid}  kind={t.get('kind')}  "
                  f"n_real_tools={t.get('n_real_tools')}")
            rec = None
            if offsets is not None:
                span = offsets.get(tid)
                entry = read_entry(a.trace, span) if span else None
                payload = (entry or {}).get("output")
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        payload = None
                rec = recorded_calls(payload if isinstance(payload, dict) else None)
            for r in by_turn[tid]:
                print_eval(r, width=a.width, full=a.full, metrics=metrics,
                           recorded=rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
