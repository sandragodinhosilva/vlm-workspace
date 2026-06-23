# Serving-script migration: `vlm-evaluation` → `utilities/serve/` (IN PROGRESS)

**Status:** 🟡 IN PROGRESS — started 2026-06-23. Serving script migrated; repo archive pending.
**Owner:** sgsilva. **For:** any Claude agent (or human) touching VLM serving or the eval drivers.

## TL;DR for agents
- **The canonical vLLM serving script is now `/home/sgsilva/utilities/serve/start_vllm_server.sh`.**
  Use THIS path in any new command, doc, or script. Do NOT use
  `/home/sgsilva/vlm-evaluation/start_vllm_server.sh` (deprecated; being archived).
- **The serving client `query_server.py` now lives at `/home/sgsilva/utilities/serve/query_server.py`**
  (the canonical May-27 copy from `vlm-post-training/inference/` — see "query_server.py" note below).
- `/home/sgsilva/vlm-evaluation` is **being de-promoted / archived.** It is a legacy eval toolkit;
  the only things still live in it were the serving script + client, which have now moved here.

## Why
`vlm-evaluation` is a legacy repo (its `evaluate.py` was long superseded by
`vlm-post-training/eval/evaluate.py`; its `docs/` are stale — see "Stale docs" below). The single
load-bearing asset was `start_vllm_server.sh`. Sandra decided (2026-06-23) to move serving into her
own tools dir (`utilities/serve/`) and archive the rest.

**Important — the two `start_vllm_server.sh` copies were NOT equivalent.** The `vlm-evaluation` copy
(Jun 18, ~13 KB) is the GOOD one: GPU/port/local-path preflights, `ENABLE_THINKING` +
`--reasoning-parser qwen3`, `SERVED_MODEL_NAME` long-path fix, startup heartbeat, venv resolver,
GLM-4.7 + Kimi branches. The `vlm-post-training/inference/start_vllm_server.sh` copy (Apr 21, ~5 KB)
is an OLDER pmartins-flavored version (it `cd`s into `/home/pmartins/...` and uses `uv`, no
preflights). **Do not "consolidate" by pointing things at the vlm-post-training copy — that's a
regression.** The canonical copy is the one now in `utilities/serve/` (a verbatim copy of the Jun-18
vlm-evaluation script).

## What's DONE (2026-06-23)
- ✅ Copied the Jun-18 canonical script → `/home/sgsilva/utilities/serve/start_vllm_server.sh`
  (byte-identical to the vlm-evaluation original; `bash -n` clean).
- ✅ Repointed OPERATIVE scripts to the new path:
  `utilities/eval/eval_all.sh` (preflight + invocation), `eval_grpo_stage2.sh`, `eval_grpo_steps.sh`,
  `serve_only.sbatch`, `reasoner_sweep.sh` (comment), `benchmarks/scripts/run_full_eval.sh`.
- ✅ Repointed RUNTIME CONFIG + SKILL: `~/.claude/settings.json`, `settings.local.json`,
  `~/.claude/skills/serve-vllm/SKILL.md`.
- ✅ Repointed UTILITIES DOCS: `utilities/slurm_vllm_workflow.md`, `utilities/commands.txt`.
- ✅ Created a forwarding shim at `/home/sgsilva/vlm-evaluation-shim/start_vllm_server.sh`
  (prints a deprecation notice, `exec`s the new path). Destined to occupy the old path slot when the
  repo is archived, so `~/.bash_history` replays (~285 hits) and un-swept doc examples keep working.

## What's PENDING
- ⏳ **Sweep `vlm-post-training` docs/runbooks** still citing the old path (~18 files, mostly
  historical command examples): `docs/SINGLE_STAGE_EVAL_PIPELINE.md`, `SFT_EXPORT_EVAL_CHECKLIST.md`,
  `TASKS_OVERVIEW.md`, `aux_tasks/.../VIDEO_SFT_RUNBOOK.md`, `aux_tasks/docs/visual_obs/*`,
  `aux_tasks/sft/commands/*.txt`, etc. (Low urgency — the shim keeps them working.)
- ⏳ **Archive the `vlm-evaluation` repo itself.** Plan: install the shim at the old path
  (`/home/sgsilva/vlm-evaluation/start_vllm_server.sh` → forwarder) BEFORE moving the bulk, OR move
  the repo to an archive location and drop a standalone shim dir at the old path. NOTE: the repo is a
  personal fork committed to `sandragodinhosilva/vlm-evaluation`; it has ~1023 uncommitted deletions
  (do NOT stage/commit those). `query_server.py` also lives there (`utils/query_server.py`) — decide
  whether it migrates too — DONE: see note below.

## query_server.py (DONE 2026-06-23)
Copied the **canonical** copy → `/home/sgsilva/utilities/serve/query_server.py`. There were TWO
diverging copies (neither a superset):
- `vlm-post-training/inference/query_server.py` (May 27) — **CANONICAL, the one copied here.** Has
  Gemini/Vertex AI (`query_gemini_with_genai_sdk`, google-genai SDK) + `query_with_vllm_direct` +
  `encode_images_to_video(..., mirror=)`. The original at `vlm-post-training/inference/` STAYS (not
  removed) — this is a second canonical copy next to the launcher.
- `vlm-evaluation/utils/query_server.py` (Apr 6) — older; has the async-litellm path
  (`async_query_with_litellm`, `acompletion`) + `resolve_model_name`, NO Gemini. Goes away with the
  archive. If you ever need the async path or `resolve_model_name`, pull them from git history.
The copy is self-contained (no relative imports / `__file__` assumptions) — safe at the new path.
- ⏳ **Stale benchmark docs** in `vlm-evaluation/docs/` flagged for archive/correction (see below).

## Stale docs found in `vlm-evaluation/docs/` (de-promote with the repo)
- `BENCHMARKS_TESTED.md` (2026-02-02): dead paths (`/mnt/data/sgsilva/SIBench-VSR`,
  `/mnt/data/sgsilva/outputs/sibench/` — both gone; canonical is `/mnt/data/shared/vlm/data/
  benchmarks/SIBench/` + `/mnt/data/sgsilva/results/benchmarks/`); claims SIBench = "12 tasks" (the
  harness runs 5); stale "🔴 URGENT eval 18 remaining" TODO; baseline 38.86% (the wrong number the
  monitoring-app already corrected to 56.6%). No STALE banner (3 sibling docs already carry one).

## Authoritative serving docs (keep — these are correct)
- `~/.claude/skills/serve-vllm/SKILL.md` — the serving recipe (now points here).
- `/home/sgsilva/benchmarks/README.md` — accurate benchmark run recipes (correctly distinguishes
  SIBench-VSR from the NYU VSI-Bench paper).
- `~/utilities/eval/EVAL_README.md` + `EVAL_MASTER_METRICS.md` — the eval board + metric dictionary.
