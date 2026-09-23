#!/usr/bin/env python3
"""Find (and optionally repair) broken file links across the ~/.claude vault.

Run it after any rename/move of a report, plan, handoff or memory file.

Checks three link forms in every .md under ~/.claude and ~/utilities:
  - vault paths written out in text: ~/.claude/..., $HOME/.claude/..., or a bare
    reports/ plans/ handoffs/ skills/ path (resolved against ~/.claude)
  - relative markdown links: [text](../../reports/x.md), resolved against the linking file
  - [[wikilinks]] to memory notes (reported separately: an unmatched wikilink is allowed
    by the memory convention -- it marks a note worth writing -- so it is not an error)

A broken path is FIXABLE when its basename exists at exactly one place in the vault
(the file moved, e.g. into _archive/ or a category subdir). Dry-run by default;
--fix rewrites fixable links in place. Ambiguous or missing targets are only reported.

Usage:
  python3 vault_link_check.py            # report
  python3 vault_link_check.py --fix      # repoint fixable links
  python3 vault_link_check.py --wikilinks  # also list unmatched [[wikilinks]]
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HOME = Path.home()
VAULT = HOME / '.claude'
MEMORY = VAULT / 'projects/-home-sgsilva/memory'

SCAN_ROOTS = [VAULT, HOME / 'utilities']
# _backups/ are point-in-time snapshots and projects/<session-uuid>/ is session history: both
# are immutable, so a stale name inside them is correct and never gets rewritten.
SKIP_DIRS = {'.git', 'node_modules', '__pycache__', 'synced', '.session-consolidate', '_backups',
             'file-history', 'shell-snapshots', 'todos', 'statsig', 'ide', 'debug', 'cache',
             '.venv', 'venv', 'site-packages'}
SESSION_DIR = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
TEXT_EXT = {'.md'}
LINKABLE_EXT = ('.md', '.txt', '.csv', '.jsonl', '.json', '.py', '.sh', '.js', '.html', '.pdf', '.png')

# ~/.claude/<path>  or  $HOME/.claude/<path>  or bare reports|plans|handoffs|skills/<path>
PATH_RE = re.compile(
    r'(?P<prefix>~/\.claude/|' + re.escape(str(HOME)) + r'/\.claude/|(?<![\w./-])(?=(?:reports|plans|handoffs|skills)/))'
    r'(?P<rel>[A-Za-z0-9_./-]+\.(?:md|txt|csv|jsonl|json|py|sh|js|html|pdf|png))'
)
MDLINK_RE = re.compile(r'\]\((?P<target>[^)\s]+?)\)')
WIKI_RE = re.compile(r'\[\[(?P<name>[^\]|#]+)(?:[|#][^\]]*)?\]\]')
PLACEHOLDER = re.compile(r'[<>*{}]|YYYY|MMDD|\bNAME\b|\.\.\.|\d{4}-\d{2}-(?:XX|\dX)')


def text_files():
    for root in SCAN_ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not SESSION_DIR.match(d)]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix in TEXT_EXT:
                    yield p


def basename_index():
    idx = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(VAULT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(LINKABLE_EXT):
                idx[fn].append(Path(dirpath) / fn)
    return idx


def memory_names():
    names = set()
    for p in MEMORY.rglob('*.md'):
        names.add(p.stem)
        try:
            head = p.read_text(errors='replace')[:600]
        except OSError:
            continue
        m = re.search(r'^name:\s*(\S+)', head, re.M)
        if m:
            names.add(m.group(1).strip('"\''))
    return names


def check_file(p, idx):
    """Yield (kind, raw, resolved_path, fix_to) for each broken link in p."""
    try:
        text = p.read_text(errors='replace')
    except OSError:
        return
    seen = set()
    for m in PATH_RE.finditer(text):
        rel = m.group('rel')
        if PLACEHOLDER.search(rel):
            continue
        raw = m.group(0)
        if raw in seen:
            continue
        seen.add(raw)
        target = VAULT / rel
        if target.exists():
            continue
        yield ('path', raw, target, idx.get(target.name, []))
    for m in MDLINK_RE.finditer(text):
        t = m.group('target').split('#')[0]
        if not t or re.match(r'[a-z]+:', t) or PLACEHOLDER.search(t):
            continue
        if not t.endswith(LINKABLE_EXT) or t.startswith(('~/', '/')):
            continue  # absolute forms are covered by PATH_RE
        if ('md', t) in seen:
            continue
        seen.add(('md', t))
        target = (p.parent / t).resolve()
        if target.exists():
            continue
        yield ('mdlink', t, target, idx.get(target.name, []))


def fixed_ref(p, kind, raw, dest):
    if kind == 'path':
        m = PATH_RE.match(raw)
        return m.group('prefix') + str(dest.relative_to(VAULT))
    return os.path.relpath(dest, p.parent)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fix', action='store_true', help='rewrite links whose target exists at exactly one path')
    ap.add_argument('--wikilinks', action='store_true', help='also list unmatched [[wikilinks]]')
    args = ap.parse_args()

    idx = basename_index()
    fixable, ambiguous, missing = [], [], []
    for p in text_files():
        for kind, raw, target, cands in check_file(p, idx):
            rec = (p, kind, raw, target, cands)
            (fixable if len(cands) == 1 else ambiguous if cands else missing).append(rec)

    def show(title, recs):
        print(f'\n=== {title}: {len(recs)} ===')
        for p, kind, raw, target, cands in recs:
            print(f'  {p.relative_to(HOME)}  [{kind}]  {raw}')
            for c in cands:
                print(f'      -> exists at {c.relative_to(HOME)}')

    show('FIXABLE (target moved; exactly one match)', fixable)
    show('AMBIGUOUS (several files share the basename)', ambiguous)
    show('MISSING (no file with that name anywhere in the vault)', missing)

    if args.wikilinks:
        names = memory_names()
        unmatched = defaultdict(set)
        for p in MEMORY.rglob('*.md'):
            for m in WIKI_RE.finditer(p.read_text(errors='replace')):
                n = m.group('name').strip()
                if n not in names:
                    unmatched[n].add(p.relative_to(MEMORY))
        print(f'\n=== UNMATCHED [[wikilinks]] (allowed; informational): {len(unmatched)} ===')
        for n, srcs in sorted(unmatched.items()):
            print(f'  [[{n}]]  <- {len(srcs)} file(s)')

    if args.fix and fixable:
        by_file = defaultdict(list)
        for rec in fixable:
            by_file[rec[0]].append(rec)
        for p, recs in by_file.items():
            s = p.read_text()
            for _, kind, raw, _, cands in recs:
                new = fixed_ref(p, kind, raw, cands[0])
                # Anchor the match: a plain str.replace of '../reports/x.md' would also hit
                # the tail of an already-correct '../../reports/x.md' and over-deepen it.
                if kind == 'mdlink':
                    s = re.sub(r'\]\(' + re.escape(raw) + r'(?=[)#])', lambda _m: '](' + new, s)
                else:
                    s = re.sub(r'(?<![\w./-])' + re.escape(raw), lambda _m: new, s)
            p.write_text(s)
        print(f'\nfixed {len(fixable)} link(s) in {len(by_file)} file(s)')
    elif fixable:
        print('\n(dry run -- pass --fix to repoint the FIXABLE links)')

    sys.exit(1 if (ambiguous or missing or (fixable and not args.fix)) else 0)


if __name__ == '__main__':
    main()
