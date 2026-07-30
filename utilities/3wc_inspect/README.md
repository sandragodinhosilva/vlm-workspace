# 3wc_inspect — read-only inspection of 3WC eval output

Fronted by the `/3wc-preview-jsonl` skill. These tools **only read**; they never
write into `/mnt/data/shared/3wc/` (pmartins-owned) and never mutate a run dir.

| Script | Does |
| --- | --- |
| `preview_eval_jsonl.py` | Readable preview of a run's `generations*.jsonl` / `judged.jsonl` — summary (replay_status mix, judge-score means, flat-metric warning) + per-row block (decision, tool call, judge scores, gen_text). Also `--list` for a whole run dir. |
| `preview_conversations.py` | The companion "why" view: the **conversation the model saw**, its output, and the judge's per-metric comments, for the same rows. Imports the reader from `preview_eval_jsonl.py`. |
| `preview_by_conversation.py` | **turn → eval → turn → eval** for one member: every graded turn of one `(account, program_uuid)` in timestamp order, each followed by all its eval rows, with `PHOENIX did` / `MODEL did` side by side. Needs `--turns`; `--trace` adds the ground-truth line. |
| `build_trace_offsets.py` | One-time `turn_id → byte-offset` sidecar for a 22–102 GB Langfuse trace file, so the above can *seek* instead of cold-starting the repo's full corpus index. |

## What `replay_status` actually measures

`core/checks.py:302-318` compares the model's **tool-call name sequence** to Phoenix's
recorded one: `matched_all` (equal) · `stopped` (model's is a strict prefix) · `diverge`
(mismatch at `diverge_step`, or extra) · `abstained` (both empty) · `over_action`
(**recorded empty, model acted**). So `over_action`/`abstained` are comparisons to a
recorded no-op, not quality verdicts — which is why `preview_by_conversation.py` prints the
recorded action next to the model's.

`model=phoenix` rows (`replay_status=recorded`) are Phoenix's OWN action scored by the same
judge — a reference ceiling, not a model under test. Don't average it in.

One turn has **many** eval rows (~8 in `eval_0907`: models × judges × `sample_index`), so a
file's line count is not the number of turns evaluated.

## Why the offsets sidecar exists

An eval row carries only `turn_id` — the transcript lives in
`$DATA_DIR/raw/traces/*.jsonl`. The repo's own `core._load_out()` resolves that
correctly, but it indexes **every** manifest file on first call (~138 GB total,
one file is 102 GB), which is minutes per cold start. For inspecting a handful
of turns, a per-file offsets index is the cheap path. Use `core` (not this) for
anything that must match production resolution semantics, including era
precedence across overlapping exports.

## Typical use

```bash
# what's in the run
python3 preview_eval_jsonl.py /home/sgsilva/dawn-research/3wc/runs/eval_0907 --list

# readable preview of the first 100 judged rows
python3 preview_eval_jsonl.py /home/sgsilva/dawn-research/3wc/runs/eval_0907/judged.jsonl -n 100

# only the interesting failures
python3 preview_eval_jsonl.py .../judged.jsonl -n 20 --status diverge

# one-time index, then the conversation view
python3 build_trace_offsets.py /mnt/data/shared/3wc/raw/traces/3_agents_0907.jsonl
python3 preview_conversations.py .../judged.jsonl \
    --trace /mnt/data/shared/3wc/raw/traces/3_agents_0907.jsonl -n 5
```

## ⚠ Member data

Output contains **real member names and clinical detail**. Previews land in
`dawn-research/3wc/exploration/eval_previews/`, which is git-ignored for exactly
that reason. Never paste this output into a hosted model, a shared doc, or an
Artifact. See `/data-anonymize` before any of it leaves the machine.

## Which trace file for which run

`turn_id`s only resolve against the export era the eval set was built from —
`eval_0907` turns come from `3_agents_0907.jsonl`. A mostly-unresolved report
means the wrong trace file, and `preview_conversations.py` exits non-zero rather
than printing an empty report. Confirm with the overlap check in
`/3wc-preview-jsonl`.
