# `~/utilities/3wc_sync` — 3WC repo sync + upstream tracking

| Script | Does | Skill |
|---|---|---|
| `sync_3wc.sh` | **PULL** the 4 SWORD repos (dawn-research + 3 reference clones); **PUSH** only Sandra's `3wc/` subtree to her personal fork | `/3wc-sync` |
| `track_3wc_changes.py` | Pull, then append **what changed + who is changing it** to a running changelog | `/3wc-daily` |
| `daily_digest_3wc.sh` | Emit the deterministic facts for ONE day of **Sandra's own** 3WC work (her `dawn-research` commits, areas touched, 3WC reports, artifacts) as raw material for a prose journal entry | `/3wc-daily-journal` |

State/output:

- `.track_state.json` — per-repo baseline sha **and** an `(mtime, size)` snapshot of the watched
  export artifacts, rewritten each tracker run (lives here, outside any repo)
- `~/.claude/reports/3wc/3WC_UPSTREAM_CHANGELOG.md` — the dated log, newest first

## The three questions one entry answers

1. **What moved in the repos** — commits, authors, hot-area flag. `dawn-research` splits
   fetched-but-unmerged (the team) from local (Sandra's own).
2. **Did a new Langfuse export land** — 10 artifacts under `/mnt/data/shared/3wc/` watched by
   `(mtime, size)`: `raw/traces/*.jsonl`, `recorded_system_prompts{,_poc}.jsonl`,
   `turn_prompt_version.json`, `.index_files`. Nothing announces an export; this is the only signal.
3. **Is production serving the new prompts yet** — newest upstream prompt-commit date vs newest
   `last_seen` in `prompts_and_rubrics/*/revisions/INDEX.md`.

(2) and (3) exist because **`prompts_and_rubrics/` is distilled from production traces, not copied
from `ai-services`** — so an upstream prompt merge is a leading indicator, and only a new export
advances the 87-scenario catalogue. `ai-documentation` contains no agent prompts at all.

## The asymmetry (enforced, not conventional)

Four repos come **in** from SWORD; exactly one destination goes **out**, and it is never SWORD.
`assert_not_sword()` runs before every push and hard-fails on any `SWORDHealth`/`swordhealth` URL,
then requires the target to match `sandragodinhosilva`. Neither script writes to a SWORD repo.

## Why the tracker records baselines

The three reference clones are `--depth 1`. A shallow fetch may drop the previous commit, and then
`git log old..HEAD` returns nothing — indistinguishable from "upstream was quiet". The sidecar makes
the baseline explicit, and the entry states when history is unreachable instead of implying calm.

## The hot-area flag

`track_3wc_changes.py` flags commits touching `prompts/`, `registry/specialized/`, `*.jinja`,
`evals/`, `scenarios/`, `precision_of_care`. Prompt-semantics drift makes `decision_correctness` /
`protocol_adherence` non-comparable across eval eras — see
`~/.claude/reports/3wc/2026-07-29_3wc_prompt_layer.md` §6. Extend the `HOT` dict when a new area
starts mattering.

## Why these live here, not in the repo

`dawn-research` is a SWORDHealth repo that is never pushed from this box. Keeping the tooling in
`~/utilities/` means no untracked file sits in it waiting to be swept into a commit — the same
rationale as `~/utilities/3wc_apps/run_app.sh`.
