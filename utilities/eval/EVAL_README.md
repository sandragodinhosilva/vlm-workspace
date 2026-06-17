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
`/home/sgsilva/benchmarks/results`→`results/benchmarks`; `results/visual_obs_runs`→`visual_obs/runs`;
`results/evaluations`→`visual_obs/evaluations`; `results/agreements`→`visual_obs/agreements`;
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
ckpt → `visualobs` (+ `aux`). `eval_all.sh` orchestrates only — the server must already be up.

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

### 2. benchmarks — general public benchmarks (VSI-Bench / MMMU-val / Video-MME)
- **Driver:** `/home/sgsilva/benchmarks/scripts/run_eval.py` (SIBench-VSR + VLMEvalKit)
- **venv:** `/home/sgsilva/benchmarks/SIBench-VSR/.venv` (+ VLMEvalKit/.venv)
- **Config:** a JSON with `reasoning: true|false` + `model` = served path + `display_name`.
  The harness HARD-FAILS if the served thinking mode ≠ config `reasoning`. `eval_all.sh`
  auto-generates a TEMP config from `--model` + thinking and deletes it after — no committed
  config per checkpoint is needed. (The pre-written `configs/qwen35-*.json` files are the older
  manual workflow; you don't author them when driving via `eval_all.sh`.)
- **Per-run output:** `/mnt/data/sgsilva/results/benchmarks/{vsibench,mmmu_val,video_mme}[_judged]/<model>/`
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
  Video-MME, VSI-Bench, Test set Acc).
- **Judge rescore (parsing-rescue, OPTIONAL but recommended):** `--judge-base-url` +
  `--judge-model` → `rescore_{mmmu,videomme,vsibench}.py`. It re-scores right-but-unparsed
  answers (model wrote `\boxed{X}` or prose the regex missed) → writes `*_judged/` +
  `summary_judge.csv` (RAW results untouched). Raw scores can UNDERSTATE the model; the judged
  numbers are the honest ones. **Needs a SEPARATE judge server** (e.g. Qwen3.5-4B on its own
  slot — NOT the model-under-test's endpoint). The unified master CSV PREFERS `summary_judge.csv`
  when present. Run via `eval_all.sh --judge-base-url http://worker-NN:8000/v1 --judge-model
  Qwen/Qwen3.5-4B`, or re-run run_eval.py with the judge flags later (skip/resume makes it cheap).
- **⚠ `/home/sgsilva/benchmarks` is a SYMLINK → `/mnt/data/sgsilva/benchmarks`** (run_eval.py
  hardcodes the `/home` path). If the home-cleanup job removes it, benchmarks break until
  recreated: `ln -s /mnt/data/sgsilva/benchmarks /home/sgsilva/benchmarks`.

### 3. visualobs — visual-obs SFT severity eval (1181-rep test)
- **Driver:** `vlm-post-training/data_preparation/evaluate.py` (single-stage here)
- **venv:** `/home/sgsilva/vlm-post-training-home-venv`
- **For visual-obs/oracle checkpoints ONLY** — NOT multimodal models like step1299. `eval_all`
  WARNS if the model name doesn't look like a visual-obs/oracle ckpt.
- **Per-run output:** `/mnt/data/sgsilva/results/visual_obs_runs/<stem>_singlestage_think<on|off>.json`
  (metrics: `error_detection_f1`, `sample_error_detection_f1`, `overall_severity_accuracy`).
- **Full multi-stage recipe** (stage-1 obs → agreement → two-stage → single-stage) +
  registration: memory `reference_visual_obs_eval_commands` + skill `/eval-vlm`.
- **Collectors (MANUAL — numbers never typed):** `data_preparation/build_results_csv.py`
  (→ `visual-obs-sft/visual_obs_sft_results_1105.csv`), `build_single_stage_csv.py`,
  `build_formatted_csv.py`. Field contract: `data_preparation/canonical_csv_columns.py`.

## Unified master CSV (additive — does NOT touch the per-stage collectors/CSVs)
- **Compiler:** `/home/sgsilva/utilities/eval/compile_eval_results.py` (read-only; re-run anytime;
  auto-run at the end of `eval_all.sh`).
- **Output:** `/mnt/data/sgsilva/results/eval_master/eval_master.csv` (all families) PLUS one
  **per-base-model split** `eval_master_{4b,9b,27b,other}.csv` — the combined file mixes sizes
  and gets unreadable, so each split holds only that family. Same columns/join in every file.
  ONE row per `(served-checkpoint-path, thinking)`, JOINED on the served path across all three
  stages (aux RUN_METADATA.model · benchmark configs display→path · VO metadata.model). Columns:
  `model, display, thinking, MMMU_val, Video_MME, VSI_Bench, aux_video_acc, aux_text_acc,
  aux_image_composite, aux_image_dense_oks, aux_image_task4_acc, vo_error_f1, vo_sample_f1,
  vo_severity_acc, + provenance`.
- **Baselines pinned to top:** in every file the raw un-SFT'd Qwen3.5 reference rows (shared-
  models path / bare `Qwen3.5-NB` id / `baseline`-named runs) sort to the top, then the rest
  alphabetically — so the reference line is always the first thing you see.
- **Copies:** `/mnt/data/sgsilva/results/eval_master/runs/<run_id>__<ts>/` — each aux run's
  aggregate JSON + SUMMARY (browse all outcomes in one place). `--no-copy` to skip.
- A model joins across stages only where it was actually run on each; single-stage models stay
  single-stage rows. `thinking=unknown` = a source file whose name lacked `_thinkon/_thinkoff`.

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
