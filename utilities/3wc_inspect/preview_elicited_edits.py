#!/usr/bin/env python3
"""Read the ELICITED rows of a 3WC sample package as plain text, for plausibility review.

The sibling previewers in this dir read an EVAL run (judged rows, replay_status, judge scores).
An elicited GRPO row has none of that: it is a training prompt whose prefix was MUTATED, and the
question a reviewer asks is not "what did the judge think" but "could a real clinician have written
this". So this view puts the edit first -- BEFORE / AFTER verbatim, the mutator's own account, the
conversation around the edited message -- and the rollouts last, as evidence the edit did what it
was built to do.

⚠️ REAL MEMBER DATA. Local inspection only (see the skill header / /data-anonymize).

    python3 preview_elicited_edits.py <package dir> [--mode specialist_edit] [--review pending]
                                      [--context 6] [--rollouts] [-n N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import textwrap

BAR = "=" * 100
SUB = "-" * 100


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def wrap(s, indent="    ", width=96):
    if not s:
        return f"{indent}(empty)"
    return "\n".join(textwrap.fill(ln, width, initial_indent=indent, subsequent_indent=indent)
                     or indent for ln in str(s).splitlines())


def rollouts_of(root, r):
    """The archived rollouts this row was graded on, from the elicitation run's attempts.jsonl."""
    d = os.path.join(root, r.get("elicit_run") or r.get("screen_run") or "", r.get("point_id") or "")
    n = r.get("elicited_at_attempt")
    try:
        atts = [json.loads(l) for l in open(os.path.join(d, "attempts.jsonl")) if l.strip()]
    except (OSError, json.JSONDecodeError):
        return [], d
    att = next((a for a in atts if a.get("attempt") == n), None)
    return ((att or {}).get("verdicts") or []), d


def context_around(root, r, n_ctx):
    """The messages around the EDITED one, from the archived prompt the policy actually saw."""
    d = os.path.join(root, r.get("elicit_run") or r.get("screen_run") or "", r.get("point_id") or "")
    n = r.get("elicited_at_attempt")
    try:
        io = json.load(open(os.path.join(d, "io", f"rollout_a{n}_00.json")))
    except (OSError, json.JSONDecodeError):
        return []
    msgs = [m for m in (io.get("messages") or []) if isinstance(m, dict)]
    i = (r.get("edits") or [{}])[0].get("msg_index")
    if i is None or i >= len(msgs):
        return []
    return [(j, msgs[j]) for j in range(max(0, i - n_ctx), min(len(msgs), i + 2))]


def head_of(m):
    c = m.get("content")
    c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return c.strip().split("\n", 1)[0][:110]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package", help="sample package dir (holds dataset/provenance.jsonl)")
    ap.add_argument("--elicit-root", default="/mnt/data/sgsilva/3wc/failure_modes/results/dormancy_pt_20260907")
    ap.add_argument("--mode", default="specialist_edit", help="'all' for every elicited row")
    ap.add_argument("--review", default="pending", help="plausibility_review value, or 'all'")
    ap.add_argument("--context", type=int, default=6, help="messages of context before the edit")
    ap.add_argument("--rollouts", action="store_true", help="also print each rollout's verdict")
    ap.add_argument("-n", type=int, default=0, help="stop after N rows (0 = all)")
    a = ap.parse_args()

    rows = read_jsonl(os.path.join(a.package, "dataset", "provenance.jsonl"))
    sel = [r for r in rows if r.get("origin") == "elicited"
           and (a.mode == "all" or r.get("mode") == a.mode)
           and (a.review == "all" or r.get("plausibility_review") == a.review)]
    print(BAR)
    print(f"ELICITED ROWS FOR PLAUSIBILITY REVIEW — {a.package}")
    print(f"{len(sel)} of {len(rows)} rows  |  mode={a.mode}  review={a.review}")
    print("⚠️ REAL MEMBER DATA — local only.")
    print("Ask of each row: same voice? every clinical fact unchanged? member tag intact?")
    print("                 does it now leave NOTHING pending on the member?")
    print(BAR)

    for k, r in enumerate(sel, 1):
        if a.n and k > a.n:
            print(f"\n... {len(sel) - a.n} more rows (raise -n)")
            break
        e = (r.get("edits") or [{}])[0]
        print(f"\n{BAR}\n[{k}/{len(sel)}] {r.get('example_id')}")
        print(f"  agent={r.get('agent')}  scenario={r.get('scenario')}  trigger={r.get('trigger_ts')}")
        print(f"  k_bad={r.get('k_bad')}/{r.get('n_rollouts')} (attempt {r.get('elicited_at_attempt')} of "
              f"{r.get('attempts')})  mutator={r.get('mutator_model')}")
        print(f"  {'✨ NOVEL — this conversation does not fail unmutated' if r.get('novel_failure') else '⚠️ VARIANT of an already-failing conversation'}")
        ctx = context_around(a.elicit_root, r, a.context)
        if ctx:
            print(f"\n  CONTEXT (the {len(ctx)-1} messages before the edited one):")
            for j, m in ctx[:-1]:
                print(f"    [{j}] {head_of(m)}")
        print(f"\n{SUB}\n  BEFORE — what the specialist actually wrote:\n{SUB}")
        print(wrap(e.get("original")))
        print(f"\n{SUB}\n  AFTER — what the policy was shown:\n{SUB}")
        print(wrap(e.get("member_message")))
        if ctx:
            print(f"\n  THEN: {head_of(ctx[-1][1])}")
        print(f"\n  MUTATOR SAYS")
        print(f"    changed:     {r.get('what_changed')}")
        print(f"    plausible?:  {r.get('plausibility')}")
        if a.rollouts:
            vs, d = rollouts_of(a.elicit_root, r)
            print(f"\n  ROLLOUTS ({len(vs)}) — {d}")
            for v in vs:
                print(f"    s{v.get('sample_index')}: rule={v.get('rule_verdict'):<14s} "
                      f"action={v.get('rule_actual')}  tools={v.get('tools_called')}")
    print(f"\n{BAR}\nVerdicts → /mnt/data/sgsilva/3wc/dormancy_samples/dataset/plausibility_review.jsonl")
    print("  plausible | implausible | unsure  (append-only, keyed by example_id; 'unsure' is a real option)")
    print(BAR)


if __name__ == "__main__":
    raise SystemExit(main())
