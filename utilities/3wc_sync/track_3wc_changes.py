#!/usr/bin/env python3
"""Daily 3WC upstream tracker: pull, then log WHAT CHANGED and WHO is working on it.

`sync_3wc.sh pull` moves the repos; it does not tell you what moved. This runs after
it and appends a dated entry to a running markdown log, so "what is actively being
changed upstream" is answerable without re-deriving it from git each time.

    python3 track_3wc_changes.py              # pull, then log (the daily use)
    python3 track_3wc_changes.py --no-pull    # log against current state only
    python3 track_3wc_changes.py --dry-run    # print the entry, write nothing

Baselines are recorded in a sidecar (`.track_state.json`) at every run, so the next
diff starts from a KNOWN sha. The reference clones are `--depth 1`, and a shallow
fetch is not obliged to keep the old commit reachable -- when it does not, the entry
says so explicitly rather than silently reporting "no changes", which would read as
"upstream was quiet" when it actually means "I could not tell".

Read-only w.r.t. the SWORD repos: it runs `git log`/`diff`, never a write command.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPOS = {
    "dawn-research": "/home/sgsilva/dawn-research",
    "ai-services": "/home/sgsilva/dawn-research/3wc/ai-services",
    "ai-documentation": "/home/sgsilva/dawn-research/3wc/ai-documentation",
    "phoenix-gym": "/home/sgsilva/dawn-research/3wc/phoenix-gym",
}
SYNC = "/home/sgsilva/utilities/3wc_sync/sync_3wc.sh"
STATE = "/home/sgsilva/utilities/3wc_sync/.track_state.json"
LOG = "/home/sgsilva/.claude/reports/3wc/3WC_UPSTREAM_CHANGELOG.md"

# Paths whose churn changes what a 3WC eval measures. A commit touching these is
# flagged, because prompt-semantics drift makes decision_correctness /
# protocol_adherence non-comparable across eras
# (reports/3wc/2026-07-29_3wc_prompt_layer.md §6).
HOT = {
    "prompt layer": ("prompts/", "registry/specialized/", ".jinja", "monalisa/registry"),
    "eval harness": ("evals/", "scenarios/", "eval_", "/tests/"),
    "poc": ("precision_of_care",),
}


def git(repo, *args, check=False):
    """Run a read-only git command; return stdout ('' on failure)."""
    try:
        r = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                           text=True, timeout=120)
    except Exception as exc:
        return f"__ERROR__ {exc}"
    if r.returncode and check:
        return f"__ERROR__ {r.stderr.strip()[:200]}"
    return r.stdout.strip()


def reflog_baseline(path, before_today=True):
    """The sha this repo was at BEFORE the most recent pull, read from the reflog.

    Used to backfill a baseline when the sidecar has none (first run, or a run that
    recorded baselines only). The reflog is local truth about where HEAD actually
    was, so this is a real baseline -- not a guess. Returns None when the reflog
    has no prior entry, and the caller then reports "first run" rather than
    inventing a diff.
    """
    out = git(path, "reflog", "show", "--date=iso", "-40")
    if out.startswith("__ERROR__") or not out:
        return None
    entries = []
    for line in out.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[0]:
            entries.append(parts[0])
    if len(entries) < 2:
        return None
    # entries[0] is current HEAD; the next distinct sha is where we were before.
    head = entries[0]
    for sha in entries[1:]:
        if sha != head:
            return sha
    return None


def load_state():
    if not os.path.exists(STATE):
        return {}
    try:
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"WARNING unreadable state file {STATE}: {exc} — treating as empty",
              file=sys.stderr)
        return {}


def save_state(state):
    tmp = STATE + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    os.replace(tmp, tmp.replace(".partial", ""))


def classify(files):
    """Which hot areas these changed paths touch."""
    hit = []
    for label, needles in HOT.items():
        if any(any(nd in f for nd in needles) for f in files):
            hit.append(label)
    return hit


def describe(repo, path, prev_sha):
    """(head_sha, date, lines[]) describing what moved since prev_sha."""
    head = git(path, "rev-parse", "HEAD")
    if head.startswith("__ERROR__"):
        return None, None, [f"⚠️ could not read HEAD: {head[10:]}"]
    date = git(path, "log", "-1", "--format=%cs")
    branch = git(path, "rev-parse", "--abbrev-ref", "HEAD")

    if not prev_sha:
        return head, date, [f"_first run — baseline recorded at `{head[:7]}` ({branch}, {date})._"]
    if prev_sha == head:
        return head, date, ["_no change._"]

    # Can we still reach the old commit? A shallow clone may have dropped it.
    if git(path, "cat-file", "-t", prev_sha).strip() != "commit":
        return head, date, [
            f"⚠️ **moved `{prev_sha[:7]}` → `{head[:7]}`, but the old commit is no longer in "
            f"this shallow clone** — cannot list what changed. "
            f"(`git -C {path} fetch --unshallow` to get full history.)"]

    log = git(path, "log", "--format=%h|%an|%cs|%s", f"{prev_sha}..HEAD")
    files = [f for f in git(path, "diff", "--name-only", f"{prev_sha}..HEAD").splitlines() if f]
    stat = git(path, "diff", "--shortstat", f"{prev_sha}..HEAD")

    lines = [f"`{prev_sha[:7]}` → `{head[:7]}` on `{branch}` — {stat or 'no file changes'}", ""]

    # dawn-research carries LOCAL commits, so HEAD movement is not the same thing as
    # upstream movement. Report what is fetched-but-unmerged separately — conflating
    # "I committed" with "the team pushed" is the whole question this log answers.
    unmerged = git(path, "log", "--format=%h|%an|%cs|%s", "HEAD..@{u}")
    if unmerged and not unmerged.startswith("__ERROR__"):
        rows = [r for r in unmerged.splitlines() if r.count("|") >= 3]
        if rows:
            lines.append(f"**⬇ {len(rows)} commit(s) fetched from upstream, NOT yet merged** "
                         f"(`git -C {path} merge --ff-only @{{u}}`):")
            for row in rows:
                sha, author, cdate, subject = row.split("|", 3)
                lines.append(f"- `{sha}` {cdate} **{author}** — {subject}")
            lines.append("")
            lines.append("Commits below are **local** (mine / this box):")
            lines.append("")

    authors = {}
    commits = []
    for row in log.splitlines():
        parts = row.split("|", 3)
        if len(parts) != 4:
            continue
        sha, author, cdate, subject = parts
        authors[author] = authors.get(author, 0) + 1
        commits.append((sha, author, cdate, subject))

    if authors:
        who = ", ".join(f"**{a}** ({n})" for a, n in
                        sorted(authors.items(), key=lambda kv: -kv[1]))
        lines.append(f"Active: {who}")
        lines.append("")

    hot = classify(files)
    if hot:
        lines.append(f"⚠️ touches: **{', '.join(hot)}** — "
                     f"see `2026-07-29_3wc_prompt_layer.md` §6 on era comparability")
        lines.append("")

    for sha, author, cdate, subject in commits:
        lines.append(f"- `{sha}` {cdate} **{author}** — {subject}")

    # the hot files themselves, so a prompt change is visible without a second command
    hot_files = [f for f in files
                 if any(nd in f for nds in HOT.values() for nd in nds)]
    if hot_files:
        lines.append("")
        lines.append(f"<details><summary>{len(hot_files)} file(s) in tracked areas</summary>")
        lines.append("")
        for f in sorted(hot_files)[:60]:
            lines.append(f"  - `{f}`")
        if len(hot_files) > 60:
            lines.append(f"  - …and {len(hot_files) - 60} more")
        lines.append("")
        lines.append("</details>")
    return head, date, lines


PROMPT_STREAMS = ("general", "onboarding", "active_treatment", "zero_to_chat",
                  "precision_of_care")
REVISIONS = "/home/sgsilva/dawn-research/3wc/prompts_and_rubrics/{}/revisions/INDEX.md"
UPSTREAM_PROMPTS = ("src/monalisa/registry/specialized/thrive/",
                    "src/monalisa/prompts/")


def newest_observed(stream):
    """Latest `last_seen` date in a stream's revision INDEX.md.

    `prompts_and_rubrics/` is distilled from PRODUCTION TRACES by
    save_prompt_revisions.py -- it is NOT a copy of ai-services. So this date is
    "the newest prompt production was actually observed serving", which is the
    thing the scenario catalogue describes. Returns None when unreadable, and the
    caller then says so rather than implying the catalogue is current.
    """
    path = REVISIONS.format(stream)
    if not os.path.exists(path):
        return None
    newest = None
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                cells = [c.strip() for c in line.split("|")]
                if len(cells) < 7:
                    continue
                # columns: | file | tokens | n | first_seen | last_seen | turns |
                for c in cells:
                    if len(c) == 10 and c[4] == "-" and c[7] == "-" and c[:4].isdigit():
                        newest = c if (newest is None or c > newest) else newest
    except Exception:
        return None
    return newest


def prompt_lag(repo_path):
    """Has an upstream prompt change NOT yet appeared in production revisions?

    The catalogue (and anything labelled against it) describes what production
    SERVED. An ai-services prompt commit is a leading indicator: it only matters
    once a new Langfuse export shows it in the wild. Reporting the two dates side
    by side keeps 'merged' from being mistaken for 'in production'.
    """
    last_commit = git(repo_path, "log", "-1", "--format=%cs", "--", *UPSTREAM_PROMPTS)
    if not last_commit or last_commit.startswith("__ERROR__"):
        return []
    observed = {s: newest_observed(s) for s in PROMPT_STREAMS}
    known = [d for d in observed.values() if d]
    if not known:
        return [f"⚠️ upstream prompts last changed **{last_commit}**, but no revision "
                f"`INDEX.md` could be read — cannot tell whether production has caught up."]
    newest_seen = max(known)
    if newest_seen >= last_commit:
        return [f"✅ prompt sync: upstream prompt change {last_commit} ≤ newest observed "
                f"production revision {newest_seen} — the catalogue reflects it."]
    per = "  ·  ".join(f"{s}={observed[s] or '?'}" for s in PROMPT_STREAMS)
    return [
        f"⚠️ **prompt export lag** — upstream prompts changed **{last_commit}**, but the newest "
        f"revision distilled from production is **{newest_seen}**.",
        "",
        f"Production has not been observed serving the new prompt yet, so "
        f"`prompts_and_rubrics/` — and the {87} scenarios extracted from it — still describe the "
        f"OLDER behaviour. That is correct, not stale: refresh is triggered by a new Langfuse "
        f"export, not by an ai-services merge.",
        "",
        f"Per stream (newest `last_seen`): {per}",
        "",
        f"When a new export lands: `save_prompt_revisions.py` → `dump_scenarios.py` "
        f"(see `2026-07-29_scenario_pipeline_review.md`).",
    ]


def local_state():
    """Sandra's own uncommitted work — so the log says what SHE has in flight too."""
    out = []
    for name, path in REPOS.items():
        if name != "dawn-research":
            continue
        dirty = git(path, "status", "--porcelain")
        if dirty and not dirty.startswith("__ERROR__"):
            rows = [r for r in dirty.splitlines() if r.strip()]
            out.append(f"**{name}**: {len(rows)} uncommitted path(s)")
            for r in rows[:12]:
                out.append(f"  - `{r.strip()}`")
        behind = git(path, "rev-list", "--count", "HEAD..@{u}")
        if behind.isdigit() and int(behind) > 0:
            out.append(f"**{name}**: {behind} commit(s) behind upstream "
                       f"(`git -C {path} merge --ff-only @{{u}}`)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true", help="skip sync_3wc.sh pull")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--since-reflog", action="store_true",
                    help="for repos with no recorded baseline, backfill it from the reflog "
                         "(where HEAD was before the last pull) instead of reporting 'first run'")
    a = ap.parse_args()

    pull_note = ""
    if not a.no_pull:
        try:
            r = subprocess.run([SYNC, "pull"], capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                pull_note = (f"⚠️ `sync_3wc.sh pull` exited {r.returncode} — "
                             f"log reflects whatever state the repos are in.\n"
                             f"```\n{(r.stderr or r.stdout).strip()[-600:]}\n```")
                print(pull_note, file=sys.stderr)
        except Exception as exc:
            pull_note = f"⚠️ pull failed to run: {exc}"
            print(pull_note, file=sys.stderr)

    state = load_state()
    prev = dict(state.get("heads", {}))
    if a.since_reflog:
        for _name, _path in REPOS.items():
            if prev.get(_name) or not os.path.isdir(_path):
                continue
            _base = reflog_baseline(_path)
            if _base:
                prev[_name] = _base
                print(f"baseline for {_name} backfilled from reflog: {_base[:7]}",
                      file=sys.stderr)
            else:
                print(f"no reflog baseline for {_name} — will report as first run",
                      file=sys.stderr)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = [f"## {today}", "", f"_synced {stamp}_", ""]
    if pull_note:
        body += [pull_note, ""]

    new_heads = {}
    changed_any = False
    for name, path in REPOS.items():
        if not os.path.isdir(path):
            body += [f"### {name}", "", f"⚠️ missing at `{path}`", ""]
            continue
        head, _date, lines = describe(name, path, prev.get(name))
        if head:
            new_heads[name] = head
        if lines and lines != ["_no change._"]:
            changed_any = True
        body += [f"### {name}", ""] + lines + [""]

    lag = prompt_lag(REPOS["ai-services"])
    if lag:
        body += ["### prompt → production sync", ""] + lag + [""]

    mine = local_state()
    if mine:
        body += ["### my working state", ""] + mine + [""]

    if not changed_any:
        body.insert(3, "_No upstream movement in any repo._")
        body.insert(4, "")

    entry = "\n".join(body).rstrip() + "\n\n---\n"

    if a.dry_run:
        print(entry)
        print(f"[dry-run] would append to {a.log}; state NOT written", file=sys.stderr)
        return 0

    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    if not os.path.exists(a.log):
        header = (
            "# 3WC upstream changelog\n\n"
            "> What moved in the four SWORD repos, newest first. Appended by\n"
            "> `~/utilities/3wc_sync/track_3wc_changes.py` (skill: `/3wc-daily`), which pulls\n"
            "> first via `sync_3wc.sh pull`. Baselines live in `3wc_sync/.track_state.json`.\n"
            ">\n"
            "> ⚠️ The three reference clones are `--depth 1`. When a fetch drops the previous\n"
            "> commit, the entry says so rather than reporting a silent \"no changes\".\n\n"
            "---\n\n")
        with open(a.log, "w", encoding="utf-8") as fh:
            fh.write(header)

    # newest-first: insert after the header block
    with open(a.log, encoding="utf-8") as fh:
        cur = fh.read()
    marker = "---\n\n"
    i = cur.find(marker)
    if i == -1:
        cur = cur + "\n" + entry
    else:
        j = i + len(marker)
        cur = cur[:j] + entry + "\n" + cur[j:]
    with open(a.log, "w", encoding="utf-8") as fh:
        fh.write(cur)

    state["heads"] = new_heads
    state["last_run"] = stamp
    save_state(state)
    print(f"appended {today} entry -> {a.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
