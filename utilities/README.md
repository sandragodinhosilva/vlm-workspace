# `~/utilities` — personal helper scripts

Standalone helper scripts kept on the small **/home** volume. These are the
day-to-day glue around training / eval / cleanup. The big sibling utilities (the
exporter, checkpoint-cleanup Python, model-naming doc) live in
**`/mnt/data/sgsilva/utilities/`** — see its own `README.md`.

> Conventions (from `~/.claude/CLAUDE.md`): run `hostname` before anything that
> serves or launches; outputs go to `/mnt/data/sgsilva/results/`, logs to
> `/mnt/data/sgsilva/logs/`; never write durable output to `/tmp`. Each script's
> own header block is the source of truth — this table is just the index.

---

## Cluster / status

| Script | What it does | Run on | Interpreter / venv |
| --- | --- | --- | --- |
| [`status.sh`](status.sh) | One-shot cluster state: vLLM servers (SSHes into all running job nodes, shows model + max_len), SLURM jobs coloured by state. `--full` adds vLLM concurrency + KV-cache metrics. | login node or any worker | bash |

---

## Export & training chains (`chains/`)

| Script | What it does | Run on | Interpreter / venv |
| --- | --- | --- | --- |
| [`chains/export_and_cleanup_nvidia_rl.sh`](chains/export_and_cleanup_nvidia_rl.sh) | **Main checkpoint exporter.** Exports all `step_*` dirs from common training roots to `/mnt/data/sgsilva/models` as HF format. `--dry-run` shows plan; `--delete` removes raw Megatron dirs only after verified export. Handles `_thinkon` suffix renaming per naming convention. See `MODEL_NAMING_CONVENTION.md`. | any | bash; export uses `nemo-rl-vlm/.venv` (has megatron.bridge) |
| [`chains/chain_thinkoff_export_eval.sh`](chains/chain_thinkoff_export_eval.sh) | Unattended chain for GRPO runs: wait for job to leave queue → wait for `tmp_step_*` to finalize → export every `step_*` to HF → **verified-delete** each raw megatron ckpt → eval sweep on a fresh idle node. | eval node | bash; GRPO export venv set inside |
| [`chains/chain_sft_export_1805_merged_reasoning.sh`](chains/chain_sft_export_1805_merged_reasoning.sh) | Same wait→export→verified-delete chain, hardcoded for the 1805 merged-reasoning SFT job. Usage: `bash chains/chain_sft_export_1805_merged_reasoning.sh <slurm_job_id>`. | any | bash; uses `nemo-rl-vlm/.venv` via `export_all_checkpoints.sh` |

---

## Eval (`eval/`)

| Script | What it does | Run on | Interpreter / venv |
| --- | --- | --- | --- |
| [`eval/eval_all.sh`](eval/eval_all.sh) | **Preferred eval entry point.** Modular driver for an already-served model: `--stages aux,benchmarks,visualobs` (any subset). Autodetects thinking mode. Each stage uses its own venv + canonical results root; compiles a unified master CSV at the end. Always `--preflight` first. | eval node | bash; dispatches to per-stage venvs |
| [`eval/compile_eval_results.py`](eval/compile_eval_results.py) | Compile results from all three pipelines (aux, benchmarks, visualobs) into one master CSV at `/mnt/data/sgsilva/results/master/eval_master.csv` (renamed from `eval_master/`; V1 frozen under `master/v1/`). Additive — originals untouched. Re-run anytime; rebuilds from scratch. | any | `vlm-post-training-home-venv/bin/python` |
| [`eval/eval_grpo_steps.sh`](eval/eval_grpo_steps.sh) | **Stage-1** sweep: sequentially eval all exported GRPO steps — serve each (thinkoff/thinkon) → stage-1 visual-obs → agreement-vs-human → stop server → per-step trajectory table. One model at a time. | eval node | `vlm-post-training-home-venv` (serving via `start_vllm_server.sh`) |
| [`eval/eval_grpo_stage2.sh`](eval/eval_grpo_stage2.sh) | **Stage-2** (two-stage) for selected GRPO steps: serve ONE thinkOFF 27B reasoner, loop each model's existing stage-1 obs JSON through `evaluate.py --two-stage` → err-detection F1 trajectory. | eval node | `vlm-post-training-home-venv` |
| [`eval/watch_benchmarks_step1299.sh`](eval/watch_benchmarks_step1299.sh) | Poll until the step1299 Video-MME rating JSON appears, then log completion. Read-only watcher; exits after 12h safety cap. Specific to step1299 — kept for reference. | any | bash |

---

## Apps (`apps/`)

| Script | What it does | Run on | Interpreter / venv |
| --- | --- | --- | --- |
| [`apps/launch_app.sh`](apps/launch_app.sh) | Launch any Gradio inspection app by name. Kills the old process on the port, launches with the correct venv + env vars, prints the browser URL. Registry in [`apps/apps_registry.yaml`](apps/apps_registry.yaml). | login node | bash + system python3 |
| [`apps/make_app_video_dataset.py`](apps/make_app_video_dataset.py) | Convert any HF dataset to a `*_browse.jsonl` for the video-sft-vlm browse app. Writes to `app_video_datasets/`. See its `--help`. | any | `vlm-post-training-home-venv/bin/python` |
| [`apps/claude-tracker.py`](apps/claude-tracker.py) | Local dashboard for Claude Code token usage and cost estimation. Reads `~/.claude/` transcript JSONLs. | any | system python3 |

---

## Cleanup (`cleanup/`)

| Script | What it does | Run on | Interpreter / venv |
| --- | --- | --- | --- |
| [`cleanup/prune_sweep_checkpoints.sh`](cleanup/prune_sweep_checkpoints.sh) | Delete superseded raw Megatron checkpoints for the visual-obs reasoning sweep (~357 GB/step). Never removes the latest step or anything written in the last `--min-age-min` (default 15). **Dry-run by default**; `--run` to delete. | any | bash |
| [`cleanup/cleanup_home.sh`](cleanup/cleanup_home.sh) | Reclaim `/home/sgsilva`: delete regenerable caches + move heavy dirs to `/mnt`, leaving symlinks so paths keep working. **Dry-run by default**; `--run`, `--run --yes`, `VLLM_KEEP_DAYS=N`. | any | bash |
| [`cleanup/cleanup_caches.py`](cleanup/cleanup_caches.py) | Interactive scan of cache + venv dirs: shows sizes, asks per-directory whether to delete. `--include-venvs`, `--include-models`. | any | `vlm-post-training-home-venv/bin/python` |
| [`cleanup/cleanup_checkpoints.py`](cleanup/cleanup_checkpoints.py) | Keep only the best checkpoint per training run; remove intermediate steps. `--dry-run` to preview. `--cleanup-models` also prunes `/mnt/data/sgsilva/models`. | any | `vlm-post-training-home-venv/bin/python` |
| [`cleanup/cleanup_qwen_checkpoints.py`](cleanup/cleanup_qwen_checkpoints.py) | Keep only the final checkpoint for each Qwen training run. `--dry-run` to preview. | any | `vlm-post-training-home-venv/bin/python` |
| [`cleanup/cleanup_all.py`](cleanup/cleanup_all.py) | Orchestrates all cleanup operations: logs (by age), Qwen checkpoints, cache dirs. `--dry-run` to preview. | any | `vlm-post-training-home-venv/bin/python` |

---

## Misc

| Script | What it does | Run on | Interpreter / venv |
| --- | --- | --- | --- |
| [`md_to_docx.py`](md_to_docx.py) | Convert one/many Markdown files to `.docx` (single or batch, `-o` for output). **Note: appends a `<!-- md_to_docx -->` note to the source `.md`** — don't point it at files you don't want modified. | any | `vlm-post-training-home-venv/bin/python` |
| [`retry_utils.py`](retry_utils.py) | Exponential-backoff retry decorator + utilities for API calls / network ops. Import as a library: `from utilities.retry_utils import retry`. | — | any python |
| [`sync_obsidian.sh`](sync_obsidian.sh) | Rsync cluster docs to a local Obsidian vault. Run from your **laptop**: `CLUSTER=new-login-0 VAULT=/Users/sandragodinhosilva/vlm-research-vault bash sync_obsidian.sh`. Optional `--dry-run`. | laptop | bash |

---

## Reference docs

| Doc | Contents |
| --- | --- |
| [`eval/EVAL_README.md`](eval/EVAL_README.md) | Full eval toolkit doc: `eval_all.sh` flags, stage details, gotchas, output layout. |
| [`slurm_vllm_workflow.md`](slurm_vllm_workflow.md) | SLURM + vLLM serving workflow notes (srun details, serving recipes). |
| [`apps/apps_registry.yaml`](apps/apps_registry.yaml) | Registry of all Gradio inspection apps: name → repo, script, port, venv, env vars. Edit here to add/change an app. |
| [`MODEL_NAMING_CONVENTION.md`](MODEL_NAMING_CONVENTION.md) | Canonical model export naming rules (stem, branch token, runtype, step, thinking flag). |
| [`CHANGELOG.md`](CHANGELOG.md) | History of changes to cleanup scripts and training run inventory. |

---

## Safety notes

- **Destructive scripts are dry-run by default.** `chains/export_and_cleanup_nvidia_rl.sh`,
  `cleanup/cleanup_home.sh`, and all `cleanup/*.py` print a plan and exit until you pass `--run`
  or `--delete`. Read the plan first.
- **Chain scripts delete raw checkpoints only after a FULLY verified HF export** —
  `config.json` + `model.safetensors.index.json` + every shard the index references.
  A partial export keeps the raw ckpt. They also wait for the training job to leave
  `squeue` and for any `tmp_step_*` to finalize first.
- **Verify ownership before stopping anything** (`squeue` / `ps -o user=`); these
  scripts only touch sgsilva's own jobs and dirs.
