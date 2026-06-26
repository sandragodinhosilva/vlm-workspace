#!/usr/bin/env python3
"""Query the dataset registry by status. Backs the /datasets skill.

Reads ~/.claude/datasets_index.json (rebuilt from DATASETS.md by
rebuild_datasets_index.py) — the machine-readable registry. Each record has a
`status` field: canonical | component | superseded (+ optional superseded_by).

Usage:
    datasets_query.py resolve  <keyword>      # canonical entries matching name/purpose
    datasets_query.py guard    <path-or-name> # is this safe to train/eval on?
    datasets_query.py list     [status]       # registry grouped by status
"""
import json
import sys
from pathlib import Path

INDEX = Path.home() / '.claude' / 'datasets_index.json'


def load():
    if not INDEX.exists():
        sys.exit(f"datasets_query: {INDEX} missing — run rebuild_datasets_index.py first")
    return json.loads(INDEX.read_text())['datasets']


def fmt(d):
    line = f"  {d['status']:<10} {d.get('rows', '?'):>8}  {d['name']}  ({d['date']})"
    if d.get('superseded_by'):
        line += f"  -> superseded by {d['superseded_by']}"
    return line


def cmd_resolve(kw):
    # Token-based: every whitespace-separated word must appear somewhere in the
    # name or purpose (order-independent), so "12k mix" finds mix_12k_1506.
    toks = kw.lower().split()
    rows = load()
    hits = [d for d in rows
            if all(t in (d['name'] + ' ' + d.get('purpose', '')).lower() for t in toks)]
    if not hits:
        print(f"No registry match for '{kw}'. Try `list` to see all names.")
        return
    canon = [d for d in hits if d['status'] == 'canonical']
    other = [d for d in hits if d['status'] != 'canonical']
    if canon:
        print(f"CANONICAL match(es) for '{kw}' (newest first):\n")
        for d in canon:
            print(f"  • {d['name']}  ({d['date']}, {d.get('rows','?')} rows)")
            print(f"    path: {d['path']}")
            print(f"    {d.get('purpose','')}\n")
    else:
        print(f"⚠️  NO canonical dataset matches '{kw}' — only "
              f"{'/'.join(sorted({d['status'] for d in other}))} entries.")
    if other:
        print("Also matched (NOT directly usable as a standalone train/eval set):")
        for d in other:
            print(fmt(d))


def cmd_guard(target):
    rows = load()
    t = target.rstrip('/')
    base = Path(t).name
    # Match by exact path, path prefix (e.g. .../name/train), or exact name.
    match = None
    for d in rows:
        dp = d.get('path', '').rstrip('/')
        if t == dp or t == d['name'] or t.startswith(dp + '/') or base == d['name']:
            match = d
            break

    if match is None:
        print(f"⚠️  UNREGISTERED: '{target}' is NOT in DATASETS.md.")
        print("    Either it's scratch/unregistered, or it should be dlog'd first.")
        print("    Verify this is the dataset you mean before launching.")
        sys.exit(3)

    st = match['status']
    if st == 'canonical':
        print(f"✅ CANONICAL — safe to use: {match['name']} ({match.get('rows','?')} rows)")
        print(f"    {match['path']}")
        sys.exit(0)
    elif st == 'component':
        print(f"⚠️  COMPONENT — '{match['name']}' is a source folded into a mix/union,")
        print(f"    NOT a standalone train/eval set. You probably want the mix that consumes it.")
        print(f"    purpose: {match.get('purpose','')}")
        print("    Confirm explicitly before launching on the raw component.")
        sys.exit(2)
    elif st == 'superseded':
        repl = match.get('superseded_by', '(unknown)')
        print(f"⛔ SUPERSEDED — do NOT train on '{match['name']}'. Use '{repl}' instead.")
        # Show the replacement's path if it's in the registry.
        rd = next((d for d in rows if d['name'] == repl), None)
        if rd:
            print(f"    replacement: {rd['path']}  ({rd.get('rows','?')} rows)")
        print("    Confirm explicitly if you really intend the superseded one.")
        sys.exit(2)
    else:
        print(f"❓ Unknown status '{st}' for {match['name']} — inspect DATASETS.md.")
        sys.exit(3)


def cmd_list(status_filter=None):
    rows = load()
    if status_filter:
        rows = [d for d in rows if d['status'] == status_filter]
    order = {'canonical': 0, 'component': 1, 'superseded': 2}
    for st in ['canonical', 'component', 'superseded']:
        grp = [d for d in rows if d['status'] == st]
        if not grp:
            continue
        print(f"\n=== {st.upper()} ({len(grp)}) ===")
        for d in sorted(grp, key=lambda x: x['date'], reverse=True):
            print(fmt(d))


def main():
    if len(sys.argv) < 2:
        cmd_list()
        return
    cmd = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == 'resolve':
        if not arg:
            sys.exit("resolve needs a keyword")
        cmd_resolve(arg)
    elif cmd == 'guard':
        if not arg:
            sys.exit("guard needs a path or name")
        cmd_guard(arg)
    elif cmd == 'list':
        cmd_list(arg)
    else:
        sys.exit(f"unknown command '{cmd}' (use resolve|guard|list)")


if __name__ == '__main__':
    main()
