# `~/utilities/apps` — Gradio inspection app tooling

Everything needed to launch, browse, and build input datasets for Sandra's
data-inspection Gradio apps. For **which app to use for a given task**, see the
`/apps` skill's "I want to…" routing table — this file is the mechanical reference
(ports, launch, schemas, registry).

## Quick start

```bash
# On the login node:
~/utilities/apps/launch_app.sh --list              # see all 19 apps + ports
~/utilities/apps/launch_app.sh video-sft           # launch; prints browser URL

# On your local Mac/PC (tunnel — remote port + 10000):
ssh -N -L 17862:localhost:7862 new-login-1
# Then open: http://localhost:17862/
```

## Layout

Apps come in two flavors. **Only a handful of standalone apps physically live here**
(`~/utilities/apps/`); the rest stay inside their data/working repos (they import
sibling code from there, e.g. `browser` → `vlm-post-training/web/browser-app/`,
`vo-compare` → `../data_preparation/canonical_csv_columns`, and several are part of the
`original-vlm-post-training` upstream-sync set). The **registry + `launch_app.sh` are
the unification layer** — they reach into each repo. This directory is *not* meant to
hold all app sources; moving the embedded ones here would break their imports.

| Path | What |
|------|------|
| `*.py` (root) | Self-contained entrypoints: `vibe_test.py`, `grpo_dashboard.py`, `sft_dashboard.py`, `sft_text_data_browser.py`, `vobs_schema_inspector.py`, `claude-tracker.py` |
| `video_sft/` | `video-sft` app + its siblings `prejudge_viewer.py`, `mesh_viewer.py` — relocated 2026-06-30 out of the archived `video-sft-vlm` repo |
| `monitoring-app/` | The `monitoring` app (port 7861) — absorbed from its old `vlm-monitoring-app.git` repo; plain-tracked inside `utilities` |
| `scripts/` | Non-launched tooling: the `make_app_video_dataset.py` browse-dataset builder + subprocess helpers `vibe_test.py` shells out to (`_vertex_call.py`, `_severity_metrics.py`) |

## Files

| File | Purpose |
|------|---------|
| [`launch_app.sh`](launch_app.sh) | Launcher: `launch_app.sh <name> [KEY=VAL...] [--extra-arg...]` — kills existing process, launches correct venv + env, prints browser URL. Also `--list`, `--status`. |
| [`apps_registry.yaml`](apps_registry.yaml) | Registry of all 19 apps: name → workstream, repo, script, port, venv, env vars, args. **Edit here** to add or change an app. |
| [`scripts/make_app_video_dataset.py`](scripts/make_app_video_dataset.py) | Convert an HF dataset to a `*_browse.jsonl` for the video-sft app. Output → `/mnt/data/sgsilva/datasets/app_video_datasets/`. |
| [`claude-tracker.py`](claude-tracker.py) | Local HTTP dashboard for Claude Code token usage and cost (port 8080). |
| [`GRADIO_APPS_REPORT.md`](GRADIO_APPS_REPORT.md) | Dated audit report: per-app history, staleness fixes already applied, and remaining (unimplemented) infra proposals (§6b). |

## Port table (all unique — all apps can run concurrently)

Grouped by **workstream** (matches the registry's `workstream:` field and the `/apps`
skill's routing table).

### General
| Port | App name | What it does |
|:----:|----------|-------------|
| 7862 | `video-sft` | **Main video browse app** — MCQA samples, Quick-load dropdown |
| 7863 | `browser` | Multi-tab dataset browser (image / video / text / reasoning) |
| 7874 | `vibe-test` | Free-form VLM inference playground (text / image / video / Vertex-gemini) |

### Visual Observations
| Port | App name | What it does |
|:----:|----------|-------------|
| 7860 | `reasoning-prompt` | Reasoning-trace **generation**-prompt editor (live teacher calls) |
| 7864 | `vo-severity` | VO severity comparator — single/two-stage A vs B |
| 7865 | `vo-compare` | VO stage-1 comparator — side-by-side vs human GT (+ `--raw-cohort` mode) |
| 7876 | `vobs-schema` | VObs schema inspector (angle vs categorical, `--version`) |
| 7877 | `rft-harvest` | RFT harvest browser — blind Pass@K samples, hallucination spot-check |
| 7879 | `multiturn-tools` | Multi-turn tool-call chat probe (`query_vlm`/`query_obs`/keypoints) |

### Aux Tasks
| Port | App name | What it does |
|:----:|----------|-------------|
| 7866 | `sft-data` | Text SFT data browser (TEXT-ONLY, 5 MCQA families) |
| 7867 | `image-viewer` | Image-row viewer (image + prompt + target + metadata; judge-review) |
| 7868 | `multimodal-compare` | Multimodal eval run comparer (COCO keypoint overlay) |
| 7869 | `reas-inspector` | Reas-mix inspector (reas2 merged mix) |
| 7870 | `prejudge-viewer` | LLM prejudge smoke verdicts vs post-hoc labels |

### 3D
| Port | App name | What it does |
|:----:|----------|-------------|
| 7871 | `mesh-viewer` | SAM-3D overlay videos — mesh + 3D skeleton (⚠ known-broken at source, see registry) |
| 7872 | `sword-viewer` | SWORD SAM-3D pipeline output browser |

### Training
| Port | App name | What it does |
|:----:|----------|-------------|
| 7873 | `grpo-dashboard` | GRPO training run dashboard |
| 7875 | `sft-dashboard` | SFT training dashboard (loss/grad-norm/lr, config-intelligence panel) |

### Utilities
| Port | App name | What it does |
|:----:|----------|-------------|
| 7861 | `monitoring` | Pipeline health dashboard (datasets / training / eval) |
| 8080 | `claude-tracker` | Claude Code token/cost tracker |

Local tunnel URL = `http://localhost:1<PORT>/`  (e.g. port 7862 → `http://localhost:17862/`)

## Venv discipline

| Venv | Has plotly | Apps |
|------|:---:|------|
| `vlm-post-training-home-venv` | ✓ | Most apps: video-sft, mesh-viewer, prejudge-viewer, sword-viewer, vibe-test, all `web/` compare apps, sft-data, grpo-dashboard, reasoning-prompt, image-viewer, vobs-schema, rft-harvest, multiturn-tools (`video-sft-vlm-home-venv` was merged in + deleted 2026-06-30) |
| `browser-app-home-venv` | ? | monitoring, browser, reas-inspector |
| `nemo-rl-vlm/.venv` | — | sft-dashboard ONLY (needs `tensorboard`; no gradio in the eval/cu12 venvs) |
| system `python3` | — | claude-tracker |

Don't cross venvs — `nemo-rl-vlm/.venv` has no gradio for any app except sft-dashboard,
which needs it specifically for the tensorboard event-file reader.

## Creating browse datasets (for video-sft app)

Browse datasets live in `/mnt/data/sgsilva/datasets/app_video_datasets/` as
`<name>_browse.jsonl`. The video-sft app's **Quick-load dataset** dropdown
picks them up automatically (rescans on restart).

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

### JSONL schema (video-sft app)

**Format A — MCQA:** `messages` (list, user question + assistant answer letter) +
`metadata` (`video_id`, `exercise_code`, `question_template`, `difficulty_tier`,
`correct_index`, `choices`, `fps`, `need_to_flip`, …). Minimum viable row: `messages`
+ `metadata.video_id`.

**Format B — Visual-observations:** same shape, but assistant content contains
`[VISUAL OBSERVATIONS]` — detected automatically, no MCQA fields needed.

Key rules: `messages` must be a real list (not a JSON string); `video_id` must resolve
under `/mnt/data/shared/vlm/data/10k/all/<video_id>/` or via `metadata.video_path`;
always supply true per-rep `fps` (never hardcode 30); set `need_to_flip` explicitly.
Full schema detail is in the `/apps` skill.

## Adding a new app to the registry

Edit [`apps_registry.yaml`](apps_registry.yaml) — add a block under `apps:`, inside the
matching workstream banner (General / Visual Observations / Aux Tasks / 3D / Training /
Utilities):

```yaml
  my-new-app:
    label: "My new app (short description)"
    workstream: <General|Visual Observations|Aux Tasks|3D|Training|Utilities>
    goal: "One sentence: what it's for and when to use it vs neighboring apps."
    repo: /home/sgsilva/<repo>
    script: <script.py>
    port: <unique port, next free after 7879, or 8080 range>
    venv: /home/sgsilva/<venv>/bin/python
    env:                      # optional
      MY_VAR: some_value
    args:                     # optional fixed CLI args
      - --flag
      - value
```

Then `launch_app.sh --list` shows it immediately, and update the registry's header
port-table comment (top of the YAML file) to match.

## Common features new browse/inspect apps should include

prev/next/random nav + jump-to-index; a live `pos / N (scope) · row i of total`
counter; a filter dropdown that scopes nav (hidden when the field is absent);
`gr.themes.Soft()` + `gr.Group()` panels + `gr.Accordion(open=False)` for secondary
detail; a color-coded status banner when rows carry a verdict; and metadata rendering
that walks the dict rather than hard-coding an allowlist. Reference implementation:
`web/image_row_viewer.py` (`image-viewer`, port 7867) — fork it for new HF image-row /
judge-review viewers. Full detail in the `/apps` skill.

## Full audit report

[`GRADIO_APPS_REPORT.md`](GRADIO_APPS_REPORT.md) — per-app git history, dataset
creation index, staleness fixes already applied (2026-06-17), and remaining
unimplemented infra proposals (§6b: tmux survival, login-node pinning,
restart-on-crash, `--stop`/`--stop-all`, `--logs` flag).
