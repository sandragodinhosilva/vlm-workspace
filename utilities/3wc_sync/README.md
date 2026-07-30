# `~/utilities/3wc_sync` — 3WC repo sync + upstream tracking

| Script | Does | Skill |
|---|---|---|
| `sync_3wc.sh` | **PULL** the 4 SWORD repos (dawn-research + 3 reference clones); **PUSH** only Sandra's `3wc/` subtree to her personal fork | `/3wc-sync` |
| `track_3wc_changes.py` | Pull, then append **what changed + who is changing it** to a running changelog | `/3wc-daily` |

State/output:

- `.track_state.json` — per-repo baseline sha, rewritten each tracker run (git-ignored by nothing; it lives here, outside any repo)
- `~/.claude/reports/3wc/3WC_UPSTREAM_CHANGELOG.md` — the dated log, newest first

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
