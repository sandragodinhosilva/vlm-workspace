# Eval master — V2 era (started 2026-06-18)

Authoritative spec for the **new evaluation era**. The V1 era (all historical scattered
single-axis runs) is **frozen** under `/mnt/data/sgsilva/results/master/v1/<date>/`. The LIVE
`eval_master*.csv` files now hold only a **curated, allowlisted** set of models, each measured on
**all 3 axes through one pipeline** (`eval_all.sh --serve`). Nothing is auto-added.

## Why a V2 era
The V1 board accreted 123 rows of partial coverage — most models had only 1 axis (VO, or aux, or
bench), scored on different dates with different settings, joined by colliding path-strings. You
can't compare a VO champion to a benchmark champion when neither was run on the other's axis.

**V2 rule: a model earns a row only after it goes through `eval_all.sh --serve` (aux + benchmarks
+ visualobs) so every row is one clean, same-pipeline, 3-axis measurement.**

## The 3 axes
1. **aux** — multimodal_reduced_testset (video / text / image MCQA + dense OKS + task4); headline
   `aux_acc_weighted_3mod`.
2. **benchmarks** — MMMU-val, Video-MME, VSI-Bench (parsable-only accuracy; runaway non-responses
   excluded from the denominator). See `feedback_eval_gotchas` §5.
3. **visual-obs** — 1181-rep stage-2 error detection; headline `vo_error_f1` (+ sample-F1,
   severity acc).

**Goal of the era:** keep improving **VO error-detection F1** while **holding** aux + benchmark
scores. A model that wins VO but tanks benchmarks is not a win.

## Files
- Live board: `/mnt/data/sgsilva/results/master/eval_master.csv` (combined) +
  `eval_master_4b.csv`, `eval_master_27b.csv` (only these two families — `ERA_FAMILIES`).
- Allowlist: `/home/sgsilva/utilities/eval/master_models.txt` — case-insensitive **substring**
  patterns matched against a row's served model PATH or DISPLAY. A row is kept iff it matches any
  line. Edit + re-run the compiler to add a model. If the file is ABSENT the compiler keeps ALL
  rows (legacy fallback).
- Compiler: `/home/sgsilva/utilities/eval/compile_eval_results.py` (read-only union/join of the
  per-stage results; the per-stage CSVs remain source of truth).
- V1 freeze: `/mnt/data/sgsilva/results/master/v1/2026-06-18/` (the 4 pre-curation files).

## Columns (logical order)
identity (`display, model, model_created, owner, is_baseline, eval_thinking`) → when
(`last_eval_ts`) → headline scores (benchmarks → visual-obs → `aux_acc_weighted_3mod`) → aux
detail → training provenance → source bookkeeping. Two clarified flags:
- **`eval_thinking`** = how the server was run at eval time (`_thinkon`/`_thinkoff`).
- **`train_reasoning`** = whether the model was TRAINED on reasoning data. (The old redundant
  `reasoning` benchmark-source column was dropped — it only mirrored `eval_thinking`.)
- **`is_baseline`** = yes/no, explicit (baselines also sort to the top of each file).

## The V2 board roster (curated 2026-06-18)
Each line is in `master_models.txt`. ✅ = has fresh new-pipeline data; ⏳ = allowlisted, awaiting
its `eval_all.sh --serve` run.

| Model | thinking | role | VO err-F1 (V1) | status |
| --- | --- | --- | --- | --- |
| `Qwen3.5-4B` baseline | both | floor | ~0.31/0.28 | ⏳ re-run |
| `Qwen3.5-27B` baseline | both | floor | ~0.42/0.42 | ⏳ re-run |
| `Qwen3.5-397B-A17B` | off | **upper-bound / distillation target** (oracle setting) | 0.788 (oracle) | ⏳ heavy serve |
| `oracle-obs-cat-union5-step339` | off | **BEST DEPLOYABLE** VO | 0.504 | ⏳ |
| `oracle-obs-cat-plus-llm-fms-step1785` | off | +LLM-FMS branch | 0.485 | ⏳ |
| `oracle-obs-merged-1805-step2558` | off | next data scale-up (86k), UNTESTED | — | ⏳ |
| `oracle-obs-cat-1105-step357` (4b) | off | small-model contender | 0.473 | ⏳ |
| `oracle-obs-cat-reasoning-step330` | on | reasoning CONTRAST line | 0.428 | ⏳ |
| `sft_step2812` (pmartins A) | on | new-pipeline SFT (fe_comparison mix) | — | running (job 93603) |
| `grpo_step492` (pmartins B) | on | new-pipeline GRPO (same mix + aux12k) | — | ✅ (Video-MME pending) |

### Findings that shaped the roster (from the 1105 VO results CSV)
- **union5 categorical is the deployable champion** (F1 0.504) — the line to beat.
- **+LLM-FMS is additive** without hurting (0.485); piling on **mix12k did NOT help F1** beyond
  union5/LLM-FMS (diminishing returns from more data).
- **Reasoning consistently UNDER-performs answer-only on VO F1** (every thinkon sweep variant
  ≤0.45 vs union5's 0.50). Keep ONE reasoning ref for contrast; do NOT spend the new line on
  reasoning VO. Matches `[[project_visual_obs_sft]]` (answer-only > reasoning).
- **4b ≈ 27b on VO F1** (0.473 vs 0.504) — small-model angle worth tracking.
- **merged_1805 (86k)** is the untested next scale-up past union5/LLM-FMS — first thing to measure.

## How to add a model to the V2 board
1. Run it through the pipeline (node must be free — `hostname`/`sinfo -t idle` first):
   ```bash
   export MODEL=/mnt/data/sgsilva/models/<ckpt>
   export BASE_MODEL=qwen3.5-27b STAGES=aux,benchmarks,visualobs THINKING=off
   export TRAIN_GROUP_ID=<group> RUN_ID=<ckpt>_full_thinkoff
   export BENCH_MAX_TOKENS=16384            # cap thinkon runaways
   # pmartins/TokenizersBackend ckpts only: export SERVE_VENV=/home/sgsilva/vlm-post-training-home-venv
   sbatch --export=ALL --job-name=eval-<ckpt> /home/sgsilva/utilities/eval/eval_all.sbatch
   ```
2. Add a substring line to `master_models.txt` that matches its served path or run_id.
3. Re-run the compiler:
   `/home/sgsilva/vlm-post-training-home-venv/bin/python /home/sgsilva/utilities/eval/compile_eval_results.py`

## Cross-refs
- Infra + gotchas: `EVAL_README.md`, memory `feedback_eval_gotchas` §5, `feedback_serving_vllm`.
- VO project state: memory `project_visual_obs_sft`, `project_grpo_visual_obs_degrades`.
