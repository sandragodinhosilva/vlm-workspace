# Model export naming convention (single source of truth)

Applies to every checkpoint exported to `/mnt/data/sgsilva/models/` by
`export_and_cleanup_nvidia_rl.sh` (which calls `nemo-rl-vlm/scripts/export_all_checkpoints.sh`).
The eval pipeline (`data_preparation/canonical_csv_columns.py:step_from_model_path`,
`build_formatted_csv.py`, the visual-obs-sft CSVs) parses these names — keep them exact.

## Shape

```
qwen35-<size>-<task...>[-reasoning]-<date>-<runtype>-step<N>[_thinkon]
```

| Segment        | Values / rule                                                             |
|----------------|---------------------------------------------------------------------------|
| family         | `qwen35` — NO dot (never `qwen3.5`; the dotted form breaks the exporter's already-exported check) |
| `<size>`       | `4b`, `27b`                                                                |
| `<task...>`    | task tokens, `-`-separated (e.g. `oracle-obs-cat`, `mix-12k-2005`, `visual-obs-cat`) |
| `reasoning`    | present iff the run trained real `<think>` traces (see thinking, below)    |
| `<date>`       | run date stamp if the task carries one (e.g. `1105`, `2705`)               |
| **`<runtype>`**| **`sft` \| `grpo` \| `sft_grpo`** — ALWAYS present; sits right before `-step` |
| `-step<N>`     | appended by `export_all_checkpoints.sh`; `<N>` is the megatron step number |
| `_thinkon`     | post-step suffix, ONLY for reasoning runs; thinkoff is unmarked (default)  |

### Runtype (CRUCIAL — never drop)
- `sft`     — `run_vlm_sft.py` output.
- `grpo`    — GRPO initialized from the raw base model (`Qwen3.5-*`).
- `sft_grpo` — GRPO initialized from one of OUR SFT exports (went SFT → then GRPO).

Detection (in `derive_model_prefix`):
1. run dir name contains both `grpo` and `sft` (e.g. `grpo_sft_...`) → `sft_grpo`;
2. else a `grpo` run whose **exact** config `policy.model_name` points at a path
   containing `sft` → `sft_grpo`; pointing at the raw base → `grpo`;
3. `grpo` token only → `grpo`; otherwise → `sft`.

### Thinking flag (`_thinkon` / unmarked)
A reasoning run trains real `<think>` and MUST be served `ENABLE_THINKING=1`; a
non-reasoning run has empty `<think>` and MUST be served `ENABLE_THINKING=0`
(wrong mode loops or collapses ~26 pts). The reasoning token is written
`reasoning`, `reas`, or `thinkon` across configs — all three mean thinkON and
are normalized to `reasoning` in the prefix, with `_thinkon` appended after the
step. thinkoff is the unmarked default (matches the existing `...-step357`).

### Dropped scaffolding (for consistency across experiments)
These config tokens carry no model meaning and are stripped: `local`, `megatron`,
`vlm`, `<N>gpu`. Stray `sft`/`grpo`/`thinkon`/`thinkoff` tokens inside the task
part are also dropped and re-added canonically, so the same experiment always
yields the same stem regardless of how its config file was named.

## Examples

| Training run (config / ckpt dir)                                | Exported name                                                    |
|----------------------------------------------------------------|-----------------------------------------------------------------|
| `sft_vlm_qwen35_4b_oracle_obs_cat_local_megatron`              | `qwen35-4b-oracle-obs-cat-sft-step357`                           |
| `sft_vlm_qwen35_4b_oracle_obs_cat_reasoning_local_megatron`   | `qwen35-4b-oracle-obs-cat-reasoning-sft-step336_thinkon`         |
| `sft_vlm_qwen35_4b_mix_reas_12k_full_2005_megatron`           | `qwen35-4b-mix-reasoning-12k-full-2005-sft-step<N>_thinkon`      |
| `grpo_visual_obs_cat_1105_4b`                                  | `visual-obs-cat-1105-4b-grpo-step<N>`                            |
| `grpo_qwen35_4b_oracle_obs_cat_reasoning_1105`                | `qwen35-4b-oracle-obs-cat-reasoning-1105-grpo-step<N>_thinkon`   |
| `grpo_sft_qwen35_4b_repetition_feedback_severity_2603_reasoning` | `qwen35-4b-repetition-feedback-severity-2603-reasoning-sft_grpo-step<N>_thinkon` |

## Migration note (2026-06-03)
Pre-convention exports lacked the runtype token and (for reasoning) the thinking
flag. The 1105 reasoning SFT was renamed on disk by hand:
`qwen35-4b-oracle-obs-cat-reasoning-1105-step{112,224,336}` → `..._thinkon`.
Under the full convention these would also carry `-sft-` before `-step`. Existing
older names without `-sft-` are grandfathered (the eval registry references them
literally); apply the full shape to all NEW exports.

## Why the flags live where they do
- `export_all_checkpoints.sh` hard-appends `-step<N>` to the prefix → the runtype
  token must be in the prefix, before `-step`.
- `step_from_model_path` regex is `step[_-]?(\d+)` on the basename → `-step<N>`
  (and `_thinkon` after it) parse cleanly.
- `_thinkon` after the step matches the established on-disk convention, so the
  formatted-CSV builder and agreement-file naming stay compatible. The wrapper
  renames `<prefix>-step<N>` → `<prefix>-step<N>_thinkon` post-export for
  reasoning runs.

## Pre-launch linter (2026-06-17)
`/home/sgsilva/utilities/chains/lint_model_name.sh` asserts a checkpoint-dir basename derives to a
compliant prefix BEFORE launch/export. It SOURCES this exporter's real `derive_model_prefix` (never
reimplements), then checks the shape `qwen35-(4b|27b)-<task>-(sft|grpo|sft_grpo)`, no dotted family,
no leaked scaffolding, and reasoning<->thinkon coherence. Run:
  bash /home/sgsilva/utilities/chains/lint_model_name.sh --config <cfg>.yaml
  bash /home/sgsilva/utilities/chains/lint_model_name.sh <checkpoint_dir>
Exit 1 => fix the checkpoint_dir basename (NOT the exporter). Wired into the /launch-sft preflight.
