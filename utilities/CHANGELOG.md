# Utilities Changelog

## 2026-06-24: `md_to_html.py` — Markdown → self-contained HTML

### `md_to_html.py` (new)
- Converts Markdown to a single self-contained, GitHub-styled `.html` (inline CSS) — for sharing a doc rendered (as the md preview looks) with someone who has no Markdown viewer; openable in any browser / email-attachable.
- Sibling of `md_to_docx.py`, same CLI: `md_to_html.py file.md` (→ sibling `.html`), `-o out.html` (single), `-o dir/` (batch).
- **Stdlib-only** (no pandoc / python-markdown) so it runs in ANY interpreter — unlike `md_to_docx.py`, which needs the `python-docx` venv.
- Supports headings, bold/italic/`code`, links, fenced code blocks, tables, blockquotes, lists, `---` rules, and raw `<details>`/`<summary>` passthrough. Not full CommonMark.

## 2026-06-24: IFBench 4th benchmark integrated into eval pipeline

### `eval/eval_all.sh`
- IFBench now runs by default in `--stages benchmarks`; stage banner, preflight output manifest, header doc, and auto-cap comment updated to name it.
- `--max-samples` now warns when applied to benchmarks (was silently applied).
- `LOG_CMD` records `--bench-extra` so skip flags appear in the log.

### `eval/eval_status.sh`
- IFBench-only runs no longer mislabel STAGE as "Video-MME" with a bogus ~52min ETA.
- `_substep` recognises IFBench; ETA per-benchmark with skip-detection; `Total datasets: 1` + IFBench override (~5min).

---

## 2026-06-23: Eval pipeline hardening, serving tooling, reasoner anti-runaway

### `eval/eval_all.sh` + `eval/EVAL_README.md`
- `--full-rebuild` flag: forces full-scan board rebuild after code changes (previously auto-rebuild only triggered on new data).
- EVAL_README updated to document automatic board rebuild + `--full-rebuild`.

### `serve/` (new)
- `start_vllm_server.sh`: one-shot vLLM server launcher with thinking-mode handling and `showmodels` diagnostic.
- `query_server.py`: helper to probe a live server.
- `README.md`: usage + vlm-evaluation migration notes.

### `eval/` — reasoner thinkON anti-runaway
- `run_eval.py`: anti-runaway stop added; accept ≥1170 tokens (token-collapse tail guard).

### `sam3d/` (new scripts)
- 3D-extras audit + remediation scripts for post-batch extra-file triage.

### `eval/EVAL_MASTER_METRICS.md` (new)
- Board metric dictionary: all 36 columns, VO-focused definitions.

### `eval/compile_eval_results.py`
- Full VO detail block with oracle-ceiling row, Avg-Dist, dense-OKS split.
- VO label clarity pass; dense-OKS fix.
- `rebuild_board`: `EXPORTER` now points at git-tracked `aux_tasks/scripts/export_eval_matrix.py` (previously reached only via untracked results tree).
- Safe rebuild automation added.

---

## 2026-06-19: Eval pipeline V2 relaunch + audit fixes

### `eval/eval_all.sh` + `eval/eval_all.sbatch`
- Fixed stale VO script paths after repo reorg; dev nodes excluded from sbatch.
- sbatch wrapper: `--gres`/`--job-name` overrides now apply correctly.
- Inherited `PORT` under SLURM treated as a leak; job-id port wins.
- `BENCH_EXTRA` (e.g. `--skip-videomme`) forwarded in sbatch.
- Audit-found defects across 3-axis pipeline fixed (same day as audit).

### `eval/compile_eval_results.py`
- Added `aux_video_3d` / `aux_video_non3d` split columns + combined label clarification.
- `aux_acc_weighted_3mod` computed in fallback JSON path.
- Curated display order, blank-row groups, readable headers on eval board.

### `logs-utils/log_run.sh`
- Footer now written even when run aborts before `log_start` exports state.

---

## 2026-06-18: Eval V2 era launch + logging infrastructure

### `eval/eval_all.sh`
- Full `--serve` mode: serves a checkpoint, runs all 3 axes (aux → visualobs → benchmarks), tears down.
- `--bench-max-tokens` caps token budget for ALL benchmarks (VSI config + MMMU/Video-MME).
- `--keep-server`, `--serve-venv` (override venv for pmartins transformers-5.x models).
- Benchmarks run LAST (aux → visualobs → benchmarks order).
- Long external paths via short `models/_ext` symlink.
- sbatch wrapper with dated slurm output subdir.
- Serve log routed to `logs/eval/serve/` (was straying to logs root).

### `eval/compile_eval_results.py` (new era)
- Curated allowlist drives clean display; V1 frozen at `results/master/v1/2026-06-18/`.
- `is_baseline` col; canonical-baseline alias + dedup; baseline reuse.
- `bench_method` column (per-cell benchmark scoring method).
- VO join by filename (two-stage champion F1), not `metadata.model`.
- `aux_video_3d` / `aux_video_non3d` split columns.
- Model columns: `model_created`, `last_eval_ts`; `eval_thinking` + `train_reasoning` (renamed from ambiguous thinking/reasoning).
- Compiler enforces V2 aux = testset_1506 only (excludes stale-testset aux rows).
- Benchmark accuracy over PARSABLE answers only (excludes non-responses).
- dedup fallback-JSON aux rows by `run_id`.

### `logs-utils/log_run.sh` + `logs-utils/logs.sh` (new)
- Centralised run logging to `/mnt/data/sgsilva/logs/`; `log_start`/`log_end`/`clog` wrappers.
- `logs.sh`: query/filter the log store.
- Wired into eval, chain, and cleanup scripts.

### `apps/vibe_test.py` (new)
- Free-form VLM inference playground on port 7874; cluster scan, dataset loader, VO scorer.

### `serve_only.sbatch` (new)
- Standalone long-lived vLLM server for eval reuse across multiple runs.

---

## 2026-02-12: Full Training Run Inventory Update

### Changes to `cleanup_checkpoints.py`

#### 1. Added all training runs (21 total)
Updated `FALLBACK_BEST_CHECKPOINTS` with all current runs organized by task:
- Task 1: 6 runs (original, 1b_cropped, 1c_cropped, cropped_v2, mixed_balanced_v1, task1_original)
- Task 2: 3 runs (v2, v4, v5)
- Task 3: 5 runs (3a_high, 3b_low_missing, 3c_background, 3c_small, 3d_mixed)
- Task 4: 7 runs (v1, v3, v5, v5.1, v5.3, v6.1.2, v6.2)

Runs with `None` best checkpoint = eval pending, keeps all checkpoints.

#### 2. Fixed glob pattern
Script now scans both `sft_vlm_megatron_4b_4epochs*` (old naming) and `sft_vlm_4b_4epochs*` (new naming). Previously missed all Task 4 MCQA and task1_cropped_v2 runs.

#### 3. Updated task_to_run mapping
Added 7 new entries: task1_cropped_v2, task2_v5, task3d_v1_mixed, task4_mcqa_v5.3, task4_mcqa_v6.1.2, task4_mcqa_v6.2.

#### 4. Added eval-pending handling
Runs with `best_step=None` now print "Eval pending" instead of a generic warning, and keep all checkpoints until evaluation determines the best.

---

## 2026-02-02: Checkpoint Cleanup Updates

### Changes to `cleanup_checkpoints.py`

#### 1. Updated Best Checkpoints (Based on Evaluation Results)
Updated the `BEST_CHECKPOINTS` dictionary based on latest evaluation reports from `/mnt/data/sgsilva/vlm-evaluation/results/evaluations` while preserving the current utilities paths for checkpoints at `/mnt/data/sgsilva/checkpoints` and exports at `/mnt/data/sgsilva/models`:

- **task3b_low_missing**: Changed from `step_338` → `step_1131` (Epoch 3, F1=77.0%)
- **task3c_small_displacement**: Changed from `step_338` → `step_1352` (Epoch 4, F1=73.7%)

Other checkpoints remain unchanged:
- task1_original: `step_648` (Epoch 2)
- task2_v2: `step_969` (Epoch 3)
- task2_v4: `step_1328` (Epoch 4)
- task3a_high: `step_646` (Epoch 2, F1=19.3%)
- task3c_background_displacement: `step_338` (Epoch 1)

#### 2. Added Fallback Logic
When the best checkpoint is not found, the script now:
1. Shows a warning message
2. Automatically selects the most recent checkpoint (highest step number) as a fallback
3. Keeps the fallback checkpoint instead of keeping ALL checkpoints
4. Marks the kept checkpoint as "(FALLBACK)" instead of "(BEST)"

**Example output:**
```
⚠️  WARNING: Best checkpoint step_1131 not found!
Available: step_338
📌 Using fallback checkpoint: step_338
✓ Keeping (FALLBACK): step_338
```

This ensures:
- At least one checkpoint is always preserved (no loss of training work)
- Disk space is still saved by deleting other intermediate checkpoints
- Clear indication when fallback logic is used

### Files Modified
- `/mnt/data/sgsilva/utilities/cleanup_checkpoints.py`

### Files Not Changed
- `cleanup_qwen_checkpoints.py` - Already has fallback logic (keeps final checkpoint)
- `cleanup_all.py` - Doesn't reference BEST_CHECKPOINTS dictionary

### Testing
Verified with dry-run mode:
```bash
python cleanup_checkpoints.py --dry-run
```

Results:
- ✅ Fallback logic correctly activates for task3b and task3c
- ✅ Existing checkpoints are preserved when best isn't available
- ✅ Script outputs clear warnings about fallback usage
