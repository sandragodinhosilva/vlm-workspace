# Eval toolkit — `/home/sgsilva/utilities/eval/`

One place for everything about evaluating a served VLM checkpoint: the three eval pipelines,
where each saves, how each is collected, the one driver that runs them, and the unified master
CSV. Recipes/skills: `/eval-vlm` (eval), `/serve-vllm` (serving).

## Results tree (unified 2026-06-17 — all under `/mnt/data/sgsilva/results/`)
```
results/
  aux/         eval_matrix.csv + eval_matrix_<base>.csv (RICH aux master) · evals/ (per-run JSON tree)
  benchmarks/  summary*.csv · vsibench/ mmmu_val/ video_mme/ [_judged]/
  visual_obs/  visual_obs_sft_results_1105[_formatted].csv · single_stage_results_1105.csv
               runs/ · evaluations/ · agreements/
  master/      eval_master.csv + eval_master_{4b,9b,27b}.csv (cross-stage join; baselines pinned top)
```
**Back-compat symlinks** (old hardcoded paths still resolve, so un-migrated scripts don't break):
`/home/sgsilva/benchmarks/results`→`results/benchmarks`;
`results/evaluations`→`visual_obs/evaluations`; `results/agreements`→`visual_obs/agreements`;
⚠ `results/visual_obs_runs` symlink REMOVED 2026-07-10 (Sandra: one dir only) — the canonical
path is `results/visual_obs/runs/`; all live code was migrated, anything stale now fails loudly;
repo `aux_tasks/evals`→`results/aux/evals`. The aux master CSV is no longer in the repo — it's at
`results/aux/`. The VO master CSVs moved out of the repo (`visual-obs-sft/`) into `results/visual_obs/`
(only the Google-sheet exports remain in the repo as reference inputs).

## TL;DR — run everything on an already-served model
```bash
# serve first (see /serve-vllm); thinking mode MUST match the SFT target.
/home/sgsilva/utilities/eval/eval_all.sh \
  --model /mnt/data/sgsilva/models/<exported-ckpt> \
  --base-model qwen3.5-4b \
  --stages aux,benchmarks,visualobs \
  --base-url http://localhost:8000/v1 \
  --train-group-id <group> --run-id <run>      # required for the aux stage
# thinking is autodetected from the endpoint; override with --thinking on|off
```
`--stages` is any subset. step1299 (multimodal) → `aux,benchmarks`. A visual-obs/oracle 27B
ckpt → `visualobs` (+ `aux`). By default `eval_all.sh` orchestrates only — the server must already be up.

## `--serve` — serve + eval + teardown in one shot (the sbatch pattern)
Add `--serve` and `eval_all.sh` launches its OWN vLLM, waits for health, runs the stages,
compiles the master CSV, then **kills only the server it started** on exit (trap on EXIT/INT/TERM
— never `pkill`-by-pattern, honoring the never-stop-others'-processes rule). Wrap it in one
`sbatch` script: allocate node → serve → eval all → teardown → node frees when the job ends.
Fully unattended.
```bash
# inside an sbatch job (hostname/node already allocated):
/home/sgsilva/utilities/eval/eval_all.sh \
  --model /mnt/data/sgsilva/models/<exported-ckpt> --base-model qwen3.5-4b \
  --stages aux,benchmarks,visualobs --serve --thinking off \
  --base-url http://localhost:8000/v1 \
  --train-group-id <group> --run-id <run>
```
- `--thinking on|off` is **REQUIRED** with `--serve` (can't probe a server that isn't up yet; it
  sets `ENABLE_THINKING`). MUST match the SFT target (non-reasoning→off, else degenerate loop).
- **TP auto-derives from the GPUs allocated to THIS job** (`CUDA_VISIBLE_DEVICES` count): a
  gpu:4 alloc → TP 4, a gpu:8 alloc → TP 8. Explicit `--tp` still wins. (A hardcoded TP 8 on a
  4-GPU alloc fails to start the server — that's why it follows the alloc now.) `max-len`
  defaults by base-model (`4b/9b`→32768, `27b`→65536); override with `--max-len`.
- **PORT auto-derives from `SLURM_JOB_ID`** (`8000 + JOB_ID%100`) so two packed jobs on one node
  get DISTINCT ports — collision-free with no manual bookkeeping. Explicit `PORT=` overrides;
  interactive (no SLURM_JOB_ID) falls back to 8000.
  ⚠ A shared-port collision makes the eval query the WRONG model's server → garbage scores
  (war story: a 27B baseline scored video 2% because it hit a grpo492 server on port 8000 —
  see `feedback_eval_gotchas` §6). The OLD GPU-bank heuristic did NOT prevent this (SLURM 0-bases
  CUDA_VISIBLE_DEVICES per job → both packed jobs saw GPUs 0-3 → both picked 8000); the job-id
  scheme (2026-06-19) fixes it. A served-id mismatch is now a HARD preflight FAIL, not a WARN.
- `--serve-wait <secs>` health-wait budget (default 1800). Serve log → DATED subdir
  `/mnt/data/sgsilva/logs/eval/serve/<YYYY-MM-DD>/eval_all_serve_<run_id>[_think<mode>]__<jobid>_<HHMMSS>.log`
  (jobid+timestamp suffix so reruns of one config don't overwrite each other).
- `--serve` and `--preflight` are mutually exclusive (preflight launches nothing).
- `--keep-server` leaves the vLLM up after a NORMAL eval exit (reuse within the job's walltime;
  INT/TERM still tear down). NOTE: in sbatch the node frees at job end regardless — for a server
  that outlives the job (reuse across evals), use `serve_only.sbatch` instead.

### Submit via the wrapper — `sbatch_eval_all.sh` (dated SLURM logs + half-node packing)
**Submit through `sbatch_eval_all.sh`, not `sbatch eval_all.sbatch` directly** — the wrapper
pre-creates the dated SLURM dir and passes `--output`/`--error` on the CLI (a static `#SBATCH
--output` can't expand `$(date)`). Direct submits land flat in `slurm/` (un-dated).
```bash
export MODEL=... BASE_MODEL=qwen3.5-27b STAGES=aux THINKING=off
export TRAIN_GROUP_ID=baseline RUN_ID=baseline_27b_newpipeline_thinkoff
export SERVE_VENV=/home/sgsilva/vlm-post-training-home-venv   # pmartins/27B-merged only; UNSET otherwise
/home/sgsilva/utilities/eval/sbatch_eval_all.sh --job-name=eval-27b-base-aux-off
```
- **Default = HALF NODE** (`gpu:4`, 96 CPU, 1200G) so TWO gpu:4 evals PACK onto one
  8-GPU/192-CPU/2489G node (partition `OverSubscribe=YES:4`). The old full-node default
  (`--cpus-per-task=192`, no `--mem`) grabbed ALL CPUs+RAM → nothing else could land even with
  4 idle GPUs (that's why gpu:4 jobs sat PENDING on "Resources" next to a half-free node).
- **Right-size the GPUs** — TP auto-derives from the alloc, so request only what the model needs:
  4B → `--gres=gpu:2 --cpus-per-task=48 --mem=600G` (TP2); 27B → the gpu:4 default (TP4);
  397B / TP8 → full node (below). Any `--gres`/`--cpus-per-task`/`--mem`/`--job-name` you pass to
  the wrapper are forwarded to `sbatch` as OPTIONS (they override the `#SBATCH` defaults) — the
  wrapper places them BEFORE the script path so SLURM reads them as options, not script args.
  (Bug fixed 2026-06-19: when `"$@"` went AFTER the script path, `--gres=gpu:2` was silently
  dropped and the job fell back to the gpu:4 default — a 4B mis-sized to 4 GPUs.) Verify after
  submit: `scontrol show job <id> | grep -E 'JobName|AllocTRES'`.
- **Full-node jobs (TP 8 / 397B-A17B MoE)** need the whole node — pass the override:
  `sbatch_eval_all.sh --gres=gpu:8 --cpus-per-task=192 --mem=2400G --job-name=...`.
- **`unset SERVE_VENV`** between launches — it's an exported env var and leaks into the next
  job if you only re-export the model. Only the pmartins `TokenizersBackend` / `merged_rep_2603`
  family needs it; bare Qwen3.5 (4B/27B/397B, `Qwen2Tokenizer`) uses the default serving venv.

## External / long-path checkpoints (e.g. pmartins) + thinkon-27B runaways
- **Long paths (>120 chars) auto-resolve to a short symlink** `models/_ext/<run_id>` for
  serve+eval, so no script hits the 255-char filename limit (`Errno 36`). Transparent; compiler
  resolves it back. No action needed beyond `--serve`.
- **pmartins `TokenizersBackend` tokenizer** → `--serve-venv /home/sgsilva/vlm-post-training-home-venv`
  (the default serving venv's transformers is too old). See `/serve-vllm`.
- **`--bench-max-tokens 16384`** for thinkon-27B: caps ALL 3 benchmarks (VSI config + MMMU/Video-MME)
  so runaway reasoning fails in ~2-4min instead of ~30min/sample (else Video-MME ≈ days, ~27%
  non-responses). Real answers untouched (well under 16384). The compiler scores benchmarks over
  PARSABLE answers only — non-responses excluded from the denominator, drop count shown in
  `bench_source` (e.g. `mmmu_val:parsable(980,-70)`).

## serve_only.sbatch — long-lived server for attach-reuse
A per-job `--serve` server dies at job end (SLURM-killed) → can't be reused across evals.
`serve_only.sbatch` serves ONE model for its full walltime (12h) so many eval drivers attach via
`--base-url` (NO `--serve`) without re-reserving. Submit with `MODEL/BASE_MODEL/THINKING/SERVE_VENV`
env, find the node (`squeue`), then `eval_all.sh ... --base-url http://<node>:8000/v1 --stages ...`.

**ALWAYS preflight first** (5s, no eval launched) — add `--preflight` to the exact command you
plan to run. It validates: server reachable + served id matches `--model`; `max_model_len` >
eval max-tokens; thinking mode; each stage's venv/driver/test-data; aux run-id/group present;
benchmark symlink. A real run auto-runs the same preflight and ABORTS on any `[FAIL]`.
```bash
/home/sgsilva/utilities/eval/eval_all.sh --model <ckpt> --base-model qwen3.5-4b \
  --stages aux,benchmarks --base-url http://localhost:8000/v1 \
  --train-group-id <group> --run-id <run> --preflight    # validate, then drop --preflight to run
```

## The three pipelines (each independent: own repo, own venv, own results root)

### 1. aux — multimodal aux-tasks (the domain test set)
- **Driver:** `vlm-post-training/aux_tasks/sft/eval_multimodal_post_sft.sh`
- **venv:** `/home/sgsilva/vlm-post-training-home-venv`
- **Test set:** `--testset-1506` = live `merged_aux_datasets/multimodal_reduced_testset_1506`
  (text + image + BOTH video MCQA sources, combined into one run → `by_source` + `by_template`
  + combined `overall`).
- **Per-run output:** `vlm-post-training/aux_tasks/evals/<base_model>/{multimodal,video,text,
  image}/<train_group>/<eval_family>/<run_id>/<ts>/` — each leg writes `results/*.json` +
  `SUMMARY_*.txt` + `RUN_METADATA.json`. NOTE: a NEW timestamped dir per launch (no skip — a
  re-run re-does inference).
- **Collector:** `aux_tasks/sft/aggregate_multimodal_eval.py` — AUTO-RUN at the end of the
  driver. Writes the per-run `multimodal_*.json` aggregate (`modalities.{video,text,image,
  image_dense,image_task4}.metric_value_pct`). There is NO cross-run master CSV on this side —
  that gap is filled by the unified master below.
- **Required:** `--train-group-id` + `--run-id` (driver hard-errors otherwise for non-baseline).

### 2. benchmarks — general public benchmarks (VSI-Bench / MMMU-val / Video-MME / IFBench)
- **Driver:** `/home/sgsilva/benchmarks/scripts/run_eval.py` (SIBench-VSR + VLMEvalKit)
- **IFBench (4th benchmark, added 2026-06-24)** — `allenai/IFBench_test` (300 prompts, instruction
  following / general non-vision capability; the OOD generalization set). **Text-only +
  RULE-SCORED (NO judge)**: deterministic checkers vendored verbatim from upstream into
  `VLMEvalKit/vlmeval/dataset/utils/ifbench/` (parity-verified ±0 vs upstream run_eval.py by
  `benchmarks/scripts/test_ifbench_parity.py`). Headline = **prompt-level LOOSE accuracy**. Runs
  via `--data IFBench` like MMMU/Video-MME; skip with `--skip-ifbench`. Two load-bearing rules:
  (a) NO `--custom-prompt qwen3` / `\boxed{}` wrapper (corrupts format constraints); (b) thinking
  models have their `<think>…</think>` stripped before scoring (in `IFBenchDataset.evaluate`).
  Data → `LMUData/IFBench/IFBench_test.jsonl`; deps (emoji, syllapy + NLTK corpora) in the
  VLMEvalKit venv (see `requirements-ifbench.txt`). NON-responses count as legit fails (no
  `_valid()` drop in the collector). The interpretation caveat: numbers are LOW (4B lands ~low)
  and n=300 is small — don't read single-digit deltas as regressions without the per-category
  breakdown in `summary_expanded.csv`. Use IFBench as the "did visual SFT/GRPO regress general
  instruction-following" sentinel.
- **venv:** `/home/sgsilva/benchmarks/SIBench-VSR/.venv` (+ VLMEvalKit/.venv)
- **Config:** a JSON with `reasoning: true|false` + `model` = served path + `display_name`.
  The harness HARD-FAILS if the served thinking mode ≠ config `reasoning`. `eval_all.sh`
  auto-generates a TEMP config from `--model` + thinking and deletes it after — no committed
  config per checkpoint is needed. (The pre-written `configs/qwen35-*.json` files are the older
  manual workflow; you don't author them when driving via `eval_all.sh`.)
- **Per-run output:** `/mnt/data/sgsilva/results/benchmarks/{vsibench,mmmu_val,video_mme,ifbench}[_judged]/<model>/`
  (ifbench has NO `_judged` tree — rule-scored)
  (moved here 2026-06-17 from `benchmarks/results/`; old path back-compat-symlinked so the
  collectors/rescorers that still hardcode `/home/sgsilva/benchmarks/results` keep resolving)
- **Skip/resume:** per benchmark — complete result file → SKIP; partial → RESUME (`--reuse` /
  VLMEvalKit checkpoint.pkl). Safe to re-invoke (resumes from `*_checkpoint.pkl`, loses nothing).
- **⚡ Concurrency — Video-MME is CLIENT-bound, not GPU-bound (learned 2026-06-17).** At the low
  VLMEvalKit video default the vLLM server sits at `num_requests_running=0, waiting=0` (STARVED)
  while the client serially decodes video frames → the GPU idles and the full ~2700-Q Video-MME
  takes many hours. Fix: `run_eval.py` now passes `--api-nproc` (env `API_NPROC`, default 32) to
  the MMMU + Video-MME calls, so ~32 requests dispatch in parallel and keep the GPU fed.
  - Diagnose starvation on the serving node: `curl -s http://<node>:8000/metrics | grep
    num_requests_running` — if it stays `0.0` while a benchmark "runs", it's client-bound; bump
    `API_NPROC`. (The login node CAN reach `http://worker-NN:8000` by hostname.)
  - Override per run: `API_NPROC=64 ... run_eval.py ...`. Restart-safe: kill the driver
    (`pkill -u sgsilva -f run_eval.py` — server stays up), relaunch with `--skip-*` for the
    done benchmarks; Video-MME resumes from its checkpoint.
- **Collector:** `benchmarks/scripts/collect_results.py` + `collect_results_expanded.py`
  (AUTO-RUN inside run_eval.py) → `/mnt/data/sgsilva/results/benchmarks/summary.csv`, `summary_expanded.csv`,
  `summary_judge.csv`, `summary_expanded_judge.csv` (cols: Model, Reasoning, MMMU-val,
  Video-MME, VSI-Bench, IF-Bench, Test set Acc). IF-Bench flows into the master CSV as `IF_Bench`
  under the "General benchmarks" band, `bench_method` cell = `IFB=rule` (never judged/parsable).
- **Judge rescore (parsing-rescue, OPTIONAL but recommended):** `--judge-base-url` +
  `--judge-model` → `rescore_{mmmu,videomme,vsibench}.py`. It re-scores right-but-unparsed
  answers (model wrote `\boxed{X}` or prose the regex missed) → writes `*_judged/` +
  `summary_judge.csv` (RAW results untouched). Raw scores can UNDERSTATE the model; the judged
  numbers are the honest ones. **Needs a SEPARATE judge server** (e.g. Qwen3.5-4B on its own
  slot — NOT the model-under-test's endpoint). The unified master CSV PREFERS `summary_judge.csv`
  when present. Run via `eval_all.sh --judge-base-url http://worker-NN:8000/v1 --judge-model
  Qwen/Qwen3.5-4B`, or re-run run_eval.py with the judge flags later (skip/resume makes it cheap).
- **Batch judge ALL models at once (the end-of-campaign pattern — recommended):** rather than
  judging per-run, after the whole eval roster finishes, judge every completed model in one pass
  with `benchmarks/scripts/run_judge_all.py`. It runs all 3 rescorers on every model that has
  finished predictions, then auto-refreshes the summary CSVs (`collect_results.py` →
  `summary[_judge].csv`, `add_test_set_acc.py`, `fill_judge_blanks.py` = fall back to non-judged
  where the judge didn't run). Serve a SEPARATE small judge first (e.g.
  `ENABLE_THINKING=0 QWEN35_VENV=/home/sgsilva/qwen3.5-serving-home-venv start_vllm_server.sh
  /mnt/data/shared/models/Qwen3.5-27B 4 262144 8000` — **judge ALWAYS thinkoff**; the thinking mode
  belongs to the model-under-test, never the judge), confirm the served id
  (`curl -s http://<node>:8000/v1/models`), then:
  ```
  cd /home/sgsilva/benchmarks
  source ~/utilities/logs-utils/log_run.sh && clog eval run_judge_all -- \
    SIBench-VSR/.venv/bin/python scripts/run_judge_all.py \
      --judge-base-url http://<node>:8000/v1 --judge-model <SERVED_ID_FROM_CURL>
  ```
  then recompile the board (`cd ~/utilities/eval && vlm-post-training-home-venv/bin/python
  compile_eval_results.py`). `--judge-model` MUST exactly match the curl'd id or requests fail.
  **clog category MUST be `eval`** (not `claude` — a mis-parsed category silently falls back to
  `claude`; lead with the category: `clog eval <name> -- <cmd>`).
  ⚠ **Stale-checkpoint skip gotcha:** `run_judge_all.py` treats ANY lingering
  `results/benchmarks/*/<model>/**/*_checkpoint.pkl` as an in-progress run and SILENTLY SKIPS that
  model from judging — even though VLMEvalKit leaves these `.pkl` files behind after a COMPLETED
  run. Before a batch judge, confirm nothing is actually running (`squeue -u $USER`), and if a
  completed model is missing from `summary_judge.csv`, a leftover `.pkl` is why (clear only the
  dead-run `.pkl`, never one owned by a live job). The judge only RESCUES (recovers a real answer
  the regex missed) — it can't fabricate: a degenerate model that emits pure `!!!!` stays ~chance
  (e.g. small25 Video-MME 1.5%), which is the correct honest score, not a parse failure.
- **⚠ `/home/sgsilva/benchmarks` is a SYMLINK → `/mnt/data/sgsilva/benchmarks`** (run_eval.py
  hardcodes the `/home` path). If the home-cleanup job removes it, benchmarks break until
  recreated: `ln -s /mnt/data/sgsilva/benchmarks /home/sgsilva/benchmarks`.

### 3. visualobs — visual-obs SFT severity eval (1181-rep test)
- **Driver:** `vlm-post-training/data_preparation/evaluate.py` (single-stage here)
- **venv:** `/home/sgsilva/vlm-post-training-home-venv`
- **For visual-obs/oracle checkpoints ONLY** — NOT multimodal models like step1299. `eval_all`
  WARNS if the model name doesn't look like a visual-obs/oracle ckpt.
- **Per-run output:** `/mnt/data/sgsilva/results/visual_obs/runs/<stem>_singlestage_think<on|off>.json`
  (metrics: `error_detection_f1`, `sample_error_detection_f1`, `overall_severity_accuracy`).
- **Full multi-stage recipe** (stage-1 obs → agreement → two-stage → single-stage) +
  registration: memory `reference_visual_obs_eval_commands` + skill `/eval-vlm`.
- **Collectors (MANUAL — numbers never typed):** `data_preparation/build_results_csv.py`
  (→ `visual-obs-sft/visual_obs_sft_results_1105.csv`), `build_single_stage_csv.py`,
  `build_formatted_csv.py`. Field contract: `data_preparation/canonical_csv_columns.py`.

## Unified master CSV (additive — does NOT touch the per-stage collectors/CSVs)
- **Compiler:** `/home/sgsilva/utilities/eval/compile_eval_results.py` (read-only; re-run anytime).
- **Routing preflight (2026-07-10, stabilization step 2):** `compile_eval_results.py --route
  <file-or-PLANNED-name>…` simulates the compiler's row-routing (row key, cohort/arm, family,
  floor, admission, matching allowlist entry) WITHOUT writing anything; exit 1 = something would
  be invisible. `eval_all.sh`'s preflight runs it automatically over the planned VO filenames.
  All routing rules live in ONE function (`resolve_vo()`); regression tests =
  `tests/test_routing.py` (run via `python -m unittest discover -s tests`).
- **Run cards (step 4):** `eval_all.sh` writes a `<result>.card.json` sidecar (checkpoint, axis,
  cohort, thinking, test set) next to every singlestage/agreement artifact at generation time;
  the compiler routes card-first — carded files need NO `vo_tokens` and cannot mis-route on
  naming. Legacy files keep the filename fallback. (reasoner_sweep stage2 cards: pending.)
- **Filename namer (step 3):** `eval_name.py build --ckpt … --axis … --thinking … [--cohort …
  --arm …]` prints the canonical stem; `eval_name.py check <name>…` validates grammar (doubled
  think tags, buried cohort, unwired arm/cohort). NEVER hand-template stems in campaign scripts.
  Grammar registered in the /nomenclature skill.
- **Rebuild is AUTOMATIC** — `eval_all.sh` runs `rebuild_board.sh` at the end of every run (and
  `eval_all.sbatch` inherits it): backup key CSVs → `results/_backups/<ts>/` → regen the COMBINED
  matrix AND each per-base `eval_matrix_{4b,27b}.csv` (a multi-base export writes ONLY the combined
  file; the compiler reads the per-base file FIRST/PRIMARY, so a stale per-base SHADOWS the board) →
  staleness guard → compile → BEFORE/AFTER diff. Default `--incremental` (only the new run's rows).
  After an **exporter/compiler CODE change** (incremental reuses the cache), force a full rebuild:
  `eval_all.sh --full-rebuild` (or `FULL_REBUILD=1` for the sbatch), or run `rebuild_board.sh`
  standalone (full-scan by default). You normally never call the compiler/exporter by hand.
- **Output:** `/mnt/data/sgsilva/results/master/eval_master.csv` (all families) PLUS one
  **per-base-model split** `eval_master_{4b,27b,other}.csv` — the combined file mixes sizes
  and gets unreadable, so each split holds only that family. Same columns/join in every file.
  (Path NOTE: renamed `eval_master/` → `master/` in the 2026-06-17 reorg; the V1 dump is frozen
  under `master/v1/2026-06-18/`.) ONE row per `(served-checkpoint-path, thinking)`, JOINED on the
  served path across all three stages (aux RUN_METADATA.model · benchmark configs display→path ·
  VO metadata.model / curated filename map). 36 columns: identity → benchmarks (`MMMU_val,
  Video_MME, VSI_Bench, bench_method`) → **visual-obs (see the metric guide below)** → aux
  (`aux_acc_weighted_3mod` + per-modality incl. `aux_video_3d`/`aux_video_non3d`) → provenance.
- **Row order = `master_models.json` group order** (allowlist file), with a BLANK separator row
  between groups — baselines first, then VO contenders, then new-pipeline. (The old alpha-sort
  with baselines-pinned-top is the legacy fallback when no allowlist file is present.)
- **Copies:** `/mnt/data/sgsilva/results/master/runs/<run_id>__<ts>/` — each aux run's
  aggregate JSON + SUMMARY (browse all outcomes in one place). `--no-copy` to skip.
- A model joins across stages only where it was actually run on each; single-stage models stay
  single-stage rows. `thinking=unknown` = a source file whose name lacked `_thinkon/_thinkoff`.

### Visual-obs (VO) metric columns — what each one means
The board carries **THREE separate VO blocks** that are NOT interchangeable (a 2026-06-22 reorg
split them so single-stage and two-stage can never silently mix in one column):

| Board column | Source-JSON field | What it measures | Pipeline |
|---|---|---|---|
| `VO Error-F1 (single-stage)` `vo_s1_error_f1` | `metrics.error_detection_f1` | One of two headline metrics (see below — NOT the sole ranker). Binary F1 of "is this error-slot present (severity>1)?", **micro-averaged over a pooled confusion matrix** of every (rep × error-type) slot; pred matched to GT BY ERROR NAME. | **single-stage** — model emits severity DIRECTLY in one call, no obs step (eval_all.sh visualobs); from `*_singlestage_*.json`. |
| `VO Sample-F1 (single-stage)` `vo_s1_sample_f1` | `metrics.sample_error_detection_f1` | Coarser sibling: binary F1 of "does this **rep** have *any* error?", one decision/rep, micro-pooled. | single-stage. |
| `VO Severity Acc (single-stage)` `vo_s1_severity_acc` | `metrics.overall_severity_accuracy` | **Exact** ordinal match of the predicted severity integer (1–6) vs GT, micro over all slots incl. the sev==1 "no error" slots. **Adjacent miss = full miss.** | single-stage. |
| `VO Error-F1/Sample-F1/Severity Acc (two-stage)` `vo_s2_*` | same 3 fields, from a `stage2_*.json` | **Identical metrics, TWO-STAGE pipeline:** a stage-2 reasoner (always thinkoff) consumes the model's stage-1 observations → severity. NOT comparable to the single-stage columns. | **two-stage** — `stage2_*.json`, but ONLY from the sft2812 reasoner (base-27B reasoner files are filtered out as historical). Currently only the 397B row fills (s2 err-F1 **47.52** = sft2812 reasoner over 397B's PLAIN obs). NOTE: this is NOT the historical 0.788 "oracle ceiling" — that came from the base-27B reasoner over 397B ORACLE obs and is excluded by the sft2812-only policy. Other rows BLANK pending their reasoner run. |
| `VO Agree-F1/Acc/Prec/Rec (vs GT)` `vo_agree_*` | `agreement_*.json` → `error_relevant.vs_gt.a.overall.{micro_f1,accuracy,precision,recall}` | Model-under-test (side **a**) vs **human GT**, from the rules-derived error firing of the stage-1 obs (sev≥2 = error). `micro_f1` = pooled-count micro F1. The comparable no-reasoner clinical signal. | single-stage obs. Only exists for runs that emit stage-1 obs (`agreement_*.json`). |

Severity is a per-slot **integer 1–6** (1 = no error; ≥2 = present); "positive" everywhere = sev>1.

**TWO headline metrics — report BOTH, NOT one ranker** (decided 2026-06-22; the audit found a single
ranker is misleading because the two measure different things and name DIFFERENT champions):
- **`vo_s1_error_f1`** (single-stage): the model emits a severity dict directly; pred is matched to GT
  **by error NAME**. This rewards detection AND output-format/name alignment, so format-trained
  reasoning models score high (e.g. GRPO492 55.27 > SFT2812 54.92 > 397B 45.21 > union5 42.51). It does
  NOT match the formatted CSV's VO ordering.
- **`vo_agree_errf1`** (agreement vs human GT): the model only answers categorical obs questions; fixed
  clinical RULES turn answers→errors→severity, scored vs the human annotator. Confound-free (no naming
  credit). This ordering MATCHES the formatted CSV's "agreement with human" band: LLM-FMS 52.04 >
  union5 49.50 > 4B-oracle 48.05 > … SFT2812 41.43 > GRPO492 40.06.

When citing "the VO winner" SAY WHICH metric — they disagree. `cat` (categorical) is the canonical
visual-obs variant — when a baseline has both a `_cat_` and an angle/other single-stage file, the board
takes `_cat_`.

**Mapping to the historical `visual_obs_sft_results_1105_formatted.csv`** (the reference Google-sheet
export): that CSV is organized in BANDS. The *"Single stage baseline"* band → the `vo_s1_*` columns;
the *"Two stages"* band → `vo_s2_*`; the *"agreement with human annotations"* band (row ~148) →
`vo_agree_*`. Within a band, col `F1 Score` of the 1st metric block = `error_f1`, col `F1 Score` of
the 2nd block = `sample_f1`, the severity block's `Acc` = `severity_acc`; the agreement band's
`Error F1 score` = `vo_agree_errf1`. (Verified byte-exact on the 4B baseline single-stage row.)

## Naming — uniformize going forward
The three pipelines historically named the same model differently (aux: base_model/run_id;
benchmark: display_name; VO: served path), which is why the master JOINS on the served PATH (the
one shared key). The path is recoverable from each pipeline WITHOUT config files: aux from
`RUN_METADATA.model`, benchmarks by decoding the result-tree `model_slug`
(`results/<bench>/<display>/--mnt--…/` → `/mnt/…`), VO from `metadata.model`. Driving evals
through `eval_all.sh` (which always serves the canonical `--model` path) keeps the join clean —
no per-checkpoint config or naming upkeep required.

## Hard rules honored
home-venv for aux/VO; benchmark venvs for benchmarks; outputs to canonical roots under
`/mnt/data/sgsilva/results` (benchmarks now under `results/benchmarks`) + `aux_tasks/evals`; literal paths; no node
assumptions (server is yours to start); thinking-mode must match the SFT target.

## Troubleshooting — check these BEFORE trusting numbers
- **Garbage scores from a PORT collision (data-corruption, not a crash).** If a `--serve` eval's
  numbers look wrong-but-plausible (e.g. video 2% for a model that should get ~50%), it likely
  queried the WRONG model's server on a shared port. **Grep the run/slurm log for**
  `served id != expected` **or** `differs from registered server model`. If present, the run is
  poisoned → quarantine it (`mv` to `aux_tasks/evals/_poisoned_*`, don't delete) and recompile.
  Prevented by auto-PORT (GPU-bank derived); never hardcode `PORT=8000` on two co-located jobs.
- **A stage aborts on `No module named X` / `can't open file …`.** A script was moved into a
  subdir and a caller still points at the old path. Sweep `${SCRIPT_DIR}/…` refs (shell) and
  `sys.path`/`parents[N]` math (python). Known-fixed: `suggest_eval_names.py` (→`inspect/`),
  `eval_layout` import in `eval_text_datasets.py` (`parents[2]`→`parents[1]`, it's in
  `aux_tasks/shared`).
- **Aux landed VIDEO-ONLY (text/image blank).** Symptom of the text stage crashing early (it runs
  before image) — historically the `eval_layout` import bug. Re-run aux after the fix; video
  re-runs harmlessly.
- **`thinking=unknown` on the board.** The source file's name lacked `_thinkon/_thinkoff`. Tag the
  `--output-file`/`RUN_ID` with the mode (same ckpt → different numbers per mode).
- **Run shows no `==== RUN END ====` footer / `/log` can't tell pass from fail.** Was the
  `exec > >(tee…)` proc-sub bug (fixed: direct `exec >> $LOG` + EXIT trap). If you re-introduce a
  `tee` in a driver, the footer/trap breaks again — redirect directly, like `clog` does.

## Logs — all dated (2026-06-19)
- **Run log** (eval_all.sh stdout+stderr): `logs/eval/<YYYY-MM-DD>/eval_all_<run>_think<mode>_<stages>__<jobid>_<ts>.log`
  via `log_start`; carries START header (real `cmd`) + END footer (status/exit/duration).
- **Serve log:** `logs/eval/serve/<YYYY-MM-DD>/eval_all_serve_<run>[_think<mode>]__<jobid>_<ts>.log`.
- **SLURM .out/.err:** `logs/eval/slurm/<YYYY-MM-DD>/eval_all_slurm-<jobid>.{out,err}` — ONLY when
  submitted via `sbatch_eval_all.sh` (the wrapper passes dated `--output`/`--error`; a direct
  `sbatch eval_all.sbatch` lands flat in `slurm/`).

## Changelog
- **2026-06-19** — half-node packing default (gpu:4/96C/1200G) + full-node override; PORT auto by
  GPU bank, TP auto by `CUDA_VISIBLE_DEVICES`; stage order aux→visualobs→benchmarks; all logs
  dated + de-collided; run-log footer fixed (silent-fail); allowlist is now `master_models.json`
  (`{pattern,display,train_reasoning,group,note}`, group-ordered + blank separators) — supersedes
  `master_models.txt` and the old "baselines alpha-pinned to top" behavior; fixed moved-script
  path breaks (`suggest_eval_names.py`, `eval_layout`). See `feedback_eval_gotchas` §6,
  `project_eval_v2_era`.
