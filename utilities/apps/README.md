# `~/utilities/apps` — Gradio inspection app tooling

Everything needed to launch, browse, and build input datasets for Sandra's
data-inspection Gradio apps.

## Quick start

```bash
# On the login node:
~/utilities/apps/launch_app.sh --list              # see all 16 apps + ports
~/utilities/apps/launch_app.sh video-sft           # launch; prints browser URL

# On your local Mac/PC (tunnel — remote port + 10000):
ssh -N -L 17862:localhost:7862 new-login-1
# Then open: http://localhost:17862/
```

## Layout

Apps come in two flavors. **Only the 4 standalone apps physically live here**; the other
12 stay inside their data/working repos (they import sibling code from there, e.g.
`video-sft` → `video-sft-vlm/app.py`, `vo-compare` → `../data_preparation/canonical_csv_columns`,
and several are part of the `original-vlm-post-training` upstream-sync set). The **registry +
`launch_app.sh` are the unification layer** — they reach into each repo. The directory is *not*
meant to hold all 16 app sources, and moving the embedded ones here would break their imports.

| Path | What |
|------|------|
| `*.py` (root) | The 3 self-contained app entrypoints: `vibe_test.py`, `grpo_dashboard.py`, `claude-tracker.py` |
| `monitoring-app/` | The 4th standalone app (`monitoring`, port 7861) — absorbed from its old `vlm-monitoring-app.git` repo; now plain-tracked inside `utilities` |
| `scripts/` | Non-launched tooling: the `make_app_video_dataset.py` browse-dataset builder + the subprocess helpers `vibe_test.py` shells out to (`_vertex_call.py`, `_severity_metrics.py`) |

## Files

| File | Purpose |
|------|---------|
| [`launch_app.sh`](launch_app.sh) | Launcher: `launch_app.sh <name> [KEY=VAL...] [--extra-arg...]` — kills existing process, launches correct venv + env, prints browser URL |
| [`apps_registry.yaml`](apps_registry.yaml) | Registry of all 16 apps: name → repo, script, port, venv, env vars. **Edit here** to add or change an app. |
| [`scripts/make_app_video_dataset.py`](scripts/make_app_video_dataset.py) | Convert an HF dataset to a `*_browse.jsonl` for the video-sft-vlm app. Output → `/mnt/data/sgsilva/datasets/app_video_datasets/`. |
| [`claude-tracker.py`](claude-tracker.py) | Local HTTP dashboard for Claude Code token usage and cost (port 8080). |
| [`scripts/`](scripts/) | Non-launched tooling + private support modules; not registry entrypoints. |

## Port table (all unique — all apps can run concurrently)

| Port | App name | What it does |
|:----:|----------|-------------|
| 7860 | `reasoning-prompt` | Reasoning-trace prompt editor with live VLM calls |
| 7861 | `monitoring` | Pipeline health dashboard (datasets / training / eval) |
| 7862 | `video-sft` | **Main video browse app** — MCQA samples, exercise explorer |
| 7863 | `browser` | Multi-tab dataset browser (image / video / text) |
| 7864 | `vo-severity` | VO severity comparator — single-stage A vs B |
| 7865 | `vo-compare` | VO stage-1 comparator — side-by-side observations |
| 7866 | `sft-data` | Text SFT data browser (5 MCQA families) |
| 7867 | `llm-fms` | LLM-FMS image+text SFT row viewer |
| 7868 | `multimodal-compare` | Multimodal eval run comparer (keypoint overlay) |
| 7869 | `reas-inspector` | Reas-mix inspector (reas2 merged mix) |
| 7870 | `prejudge-viewer` | LLM prejudge smoke verdicts vs post-hoc labels |
| 7871 | `mesh-viewer` | SAM-3D overlay videos — mesh + 3D skeleton |
| 7872 | `sword-viewer` | SWORD SAM-3D pipeline output browser |
| 7873 | `grpo-dashboard` | GRPO training run dashboard |
| 7874 | `vibe-test` | Free-form VLM inference playground (text / image / video) |
| 8080 | `claude-tracker` | Claude Code token/cost tracker |

Local tunnel URL = `http://localhost:1<PORT>/`  (e.g. port 7862 → `http://localhost:17862/`)

## Creating browse datasets (for video-sft app)

Browse datasets live in `/mnt/data/sgsilva/datasets/app_video_datasets/` as
`<name>_browse.jsonl`. The video-sft app's **Quick-load dataset** dropdown
picks them up automatically on startup.

```bash
# Convert an HF dataset:
/home/sgsilva/vlm-post-training-home-venv/bin/python \
  ~/utilities/apps/scripts/make_app_video_dataset.py \
  --source /mnt/data/sgsilva/datasets/<hf_dataset> \
  --name   <name>
# → writes app_video_datasets/<name>_browse.jsonl

# Optional flags:
#   --old-reas-from <HF>        join source reasoning_trace onto metadata.old_reas_trace
#   --echo-to-metadata <cols>   put provenance columns under metadata
#   --stratify --per-stratum N  small representative subset
#   --max-samples N             cap rows for a quick look
```

## Adding a new app to the registry

Edit [`apps_registry.yaml`](apps_registry.yaml) — add a block under `apps:`:

```yaml
  my-new-app:
    label: "My new app (short description)"
    repo: /home/sgsilva/<repo>
    script: <script.py>
    port: <unique port 7860-7873>
    venv: /home/sgsilva/<venv>/bin/python
    env:                      # optional
      MY_VAR: some_value
    args:                     # optional fixed CLI args
      - --flag
      - value
```

Then `launch_app.sh --list` will show it immediately.

## Full audit report

[`GRADIO_APPS_REPORT.md`](GRADIO_APPS_REPORT.md) — per-app details,
git history, input schemas, dataset creation index, all implemented changes,
and remaining improvement proposals (§6b): tmux survival, login-node pinning,
restart-on-crash, `--stop`/`--stop-all`, `--logs` flag.
