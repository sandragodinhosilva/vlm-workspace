# utilities/cleanup — disk housekeeping scripts

All scripts are **dry-run by default**. Nothing is deleted until you explicitly pass `--run` or `--yes`.
See also: `/clean` skill (Claude-side recipe that fronts these scripts).

---

## Scripts

### `cleanup_checkpoints.py` — MAIN checkpoint pruner ✅
**Use this for all checkpoint/model cleanup.**

Prunes non-best training steps from `/mnt/data/sgsilva/checkpoints` and optionally
`/mnt/data/sgsilva/models`. Keeper detection reads the **live eval board**
(`eval_master*.csv` + `master_models.json`) — never keeps-latest or uses a hardcoded list.

Safety guards (all ON by default):
- Export-verified: only prunes a run if its keeper step has an HF export under `models/`
- Unknown runs: kept entirely (`--unknown-policy keep-all`)
- Always dry-run first; live run requires `--yes`

```bash
PY=/home/sgsilva/vlm-post-training-home-venv/bin/python

# Dry run — preview keepers + reclaimable space
$PY cleanup_checkpoints.py --dry-run

# Live prune (checkpoints only)
$PY cleanup_checkpoints.py --yes

# Also prune HF model exports
$PY cleanup_checkpoints.py --dry-run --cleanup-models
$PY cleanup_checkpoints.py --yes --cleanup-models

# Useful flags
#   --exclude <pattern>           skip runs whose name contains pattern
#   --interactive / -i            confirm per run
#   --unknown-policy keep-latest  prune unknown runs by highest step (NOT by eval — use carefully)
#   --skip-export-check           DANGER: bypass export guard (only on explicit request)
```

Tests: `test_cleanup_checkpoints.py` (pytest-covered detector + export guard).

---

### `cleanup_home.sh` — home directory cache cleanup ✅
**Use this to reclaim space in `/home/sgsilva`.**

Deletes regenerable caches, prunes old vLLM compile cache entries, and moves heavy
data dirs to `/mnt/data/sgsilva` (leaving symlinks so all paths keep working).

What it targets (sizes as of 2026-06-25):
- `~/.cache/vllm/torch_compile_cache` — 188 GB, entries older than `VLLM_KEEP_DAYS` days
- `~/.cache/uv` — 26 GB
- `~/.triton` — 2.5 GB
- `~/.nv` — 1 GB
- `~/.vlm_video_cache`, `~/.gradio_temp`, `~/.cache/flashinfer`, `~/.cache/torch`
- `/mnt/data/sgsilva/logs/_archive/` — 25 GB cold-archived pre-reorganization logs
- Heavy dirs moved to /mnt: `vlm-post-training/aux_tasks`, `vlm-post-training/archive`,
  `vlm-evaluation/results`, `benchmarks`

```bash
# Dry run (default — nothing touched)
VLLM_KEEP_DAYS=7 /home/sgsilva/utilities/cleanup/cleanup_home.sh

# Live run (prompts per section)
VLLM_KEEP_DAYS=7 /home/sgsilva/utilities/cleanup/cleanup_home.sh --run

# Live run, skip prompts
VLLM_KEEP_DAYS=7 /home/sgsilva/utilities/cleanup/cleanup_home.sh --run --yes
```

`VLLM_KEEP_DAYS` defaults to 21; lower to 7 when the vllm cache has grown large.

---

### `cleanup_all.py` — log date-dir pruner ✅
**Use this to prune old logs from `/mnt/data/sgsilva/logs`.**

Understands the current layout: `logs/<category>/YYYY-MM-DD/` (with nested subdirs like
`eval/serve/` and `eval/slurm/`). Reports `logs/_archive/` size with a manual delete hint
but never auto-deletes it.

```bash
PY=/home/sgsilva/vlm-post-training-home-venv/bin/python

# Dry run (default: keep last 30 days)
$PY cleanup_all.py --cleanup-logs --keep-days 30 --dry-run

# Live run
$PY cleanup_all.py --cleanup-logs --keep-days 30
```

---

### `cleanup_caches.py` — interactive per-dir cache scan ✅
**Use when you want to pick-and-choose individual cache dirs interactively.**

Scans `~/.cache` dirs and (optionally) venvs and eval artifact dirs, shows sizes,
and asks per-directory whether to delete.

```bash
PY=/home/sgsilva/vlm-post-training-home-venv/bin/python

# Show sizes only (dry run)
$PY cleanup_caches.py

# Interactive deletion
$PY cleanup_caches.py --delete

# Also include venvs (VLMEvalKit only — others removed)
$PY cleanup_caches.py --delete --include-venvs

# Also include archived eval artifacts
$PY cleanup_caches.py --delete --include-models
```

---

### `prune_sweep_checkpoints.sh` — campaign script (oracle_obs_cat sweep) ⚠️
**Narrow-purpose: only targets the `sft_qwen35_27b_oracle_obs_cat_reasoning_{A,B,C,D,baseline_rerun}` sweep.**

Deletes superseded step dirs within each variant dir (keeps the latest step + optionally ep1).
The sweep it was written for is CLOSED. Keep for reference; do not repurpose for other runs —
use `cleanup_checkpoints.py` instead.

```bash
# Dry run
./prune_sweep_checkpoints.sh

# Live run, keep ep1 checkpoint
./prune_sweep_checkpoints.sh --run --keep-ep1

# Single variant
./prune_sweep_checkpoints.sh --run --variant A
```

---

### `test_cleanup_checkpoints.py` — test suite
pytest tests for `cleanup_checkpoints.py` (board detector, export guard, run-key matching).

```bash
/home/sgsilva/vlm-post-training-home-venv/bin/python -m pytest test_cleanup_checkpoints.py -v
```

Run after any change to `cleanup_checkpoints.py` or when a run starts showing `(FALLBACK)`.

---

## Decision tree

```
Want to free disk on /mnt/data?
  → Checkpoints/models:  cleanup_checkpoints.py --dry-run
  → Old logs:            cleanup_all.py --cleanup-logs --keep-days 30 --dry-run

Want to free disk on /home?
  → Caches + move dirs:  cleanup_home.sh  (dry-run by default)
  → Interactive cache:   cleanup_caches.py

Tidy memory notes?
  → /clean skill (mode B)
```
