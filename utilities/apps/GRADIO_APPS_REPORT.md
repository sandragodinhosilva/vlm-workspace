# Gradio Data-Inspection Apps — Full Audit Report

**Date:** 2026-06-17 (original audit); **facts refreshed:** 2026-07-08 — see §0.
**Scope:** All Gradio apps across Sandra's home repos; current workflow; staleness audit; speedup proposals; implemented changes.

---

## 0. 2026-07-08 Refresh Note

The app count grew from 16 (this report's original scope) to **19** — three apps added
since: `sft-dashboard` (7875), `vobs-schema` (7876), `rft-harvest` (7877), plus
`multiturn-tools` (7879). Two apps were renamed: `text-sft`→`sft-data`,
`llm-fms`→`image-viewer`. `video-sft`'s sibling viewers (prejudge, mesh) relocated
06-30 out of the archived `video-sft-vlm` repo into `~/utilities/apps/video_sft/`.
Registry now carries a `workstream:` field (General · Visual Observations · Aux Tasks ·
3D · Training · Utilities) — §1 below reflects it. Sections 2–6 (workflow history,
per-app original detail, stale-entry fixes, port reallocation, speedup proposals) are
kept as a historical record and are NOT re-verified line-by-line; §1 and §7–9 are the
refreshed current-state sections. For task routing ("which app do I want"), see the
`/apps` skill — it is now the canonical routing table, kept in sync going forward.

---

## 1. Quick-Reference Table (refreshed 2026-07-08, 19 apps)

| App | Workstream | Path | Port | Venv | Goal |
|-----|-----------|------|:----:|------|------|
| `reasoning-prompt` | Visual Obs | `vlm-post-training/web/reasoning_trace_prompt_app.py` | 7860 | `vlm-post-training-home-venv` | Tune dataset-GENERATION teacher prompts (4 task formatters) |
| `monitoring` | Utilities | `apps/monitoring-app/app.py` | 7861 | `browser-app-home-venv` | Pipeline health dashboard — datasets, training, eval |
| `video-sft` | General | `utilities/apps/video_sft/app.py` | 7862 | `vlm-post-training-home-venv` | **Main video browse app** — MCQA samples, Quick-load dropdown |
| `browser` | General | `vlm-post-training/web/browser-app/app.py` | 7863 | `browser-app-home-venv` | Multi-tab browser: image/video/text/reasoning |
| `vo-severity` | Visual Obs | `vlm-post-training/web/single_stage_compare_app.py` | 7864 | `vlm-post-training-home-venv` | Compare single/two-stage eval runs side-by-side |
| `vo-compare` | Visual Obs | `vlm-post-training/web/human_visual_obs_compare_app.py` | 7865 | `vlm-post-training-home-venv` | Stage-1 observations vs human GT (+ `--raw-cohort`) |
| `sft-data` | Aux Tasks | `utilities/apps/sft_text_data_browser.py` | 7866 | `vlm-post-training-home-venv` | Text SFT QC (5 MCQA families), TEXT-ONLY |
| `image-viewer` | Aux Tasks | `vlm-post-training/web/image_row_viewer.py` | 7867 | `vlm-post-training-home-venv` | Image row + judge-review viewer (was llm-fms) |
| `multimodal-compare` | Aux Tasks | `vlm-post-training/web/multimodal_compare_app.py` | 7868 | `vlm-post-training-home-venv` | Multimodal eval comparer, COCO keypoint overlay |
| `reas-inspector` | Aux Tasks | `vlm-post-training/aux_tasks/sft/inspect/inspect_reas2_mix_gradio.py` | 7869 | `browser-app-home-venv` | Reas2 merged mix inspector |
| `prejudge-viewer` | Aux Tasks | `utilities/apps/video_sft/prejudge_viewer.py` | 7870 | `vlm-post-training-home-venv` | LLM prejudge smoke verdicts vs post-hoc labels |
| `mesh-viewer` | 3D | `utilities/apps/video_sft/mesh_viewer.py` | 7871 | `vlm-post-training-home-venv` | SAM-3D mesh overlay + 3D skeleton ⚠ known-broken at source |
| `sword-viewer` | 3D | `vlm-post-training/…/sam3d_pilot/sword_viewer.py` | 7872 | `vlm-post-training-home-venv` | SWORD SAM-3D pipeline outputs |
| `grpo-dashboard` | Training | `utilities/apps/grpo_dashboard.py` | 7873 | `vlm-post-training-home-venv` | GRPO reward curves + Live Status tab |
| `vibe-test` | General | `utilities/apps/vibe_test.py` | 7874 | `vlm-post-training-home-venv` | Free-form inference playground; vLLM + Vertex/gemini |
| `sft-dashboard` | Training | `utilities/apps/sft_dashboard.py` | 7875 | `nemo-rl-vlm/.venv` | SFT loss/grad-norm/lr, config-intelligence panel |
| `vobs-schema` | Visual Obs | `utilities/apps/vobs_schema_inspector.py` | 7876 | `vlm-post-training-home-venv` | VObs schema browser (angle vs categorical) |
| `rft-harvest` | Visual Obs | `vlm-post-training/data_preparation/reasoning/inspect/rft_harvest_app.py` | 7877 | `vlm-post-training-home-venv` | RFT harvest browser — hallucination spot-check |
| `multiturn-tools` | Visual Obs | `vlm-post-training/web/gradio_app_multiturn.py` | 7879 | `vlm-post-training-home-venv` | Multi-turn tool-call chat probe |
| `claude-tracker` | Utilities | `utilities/apps/claude-tracker.py` | 8080 | system python3 | Claude Code token/cost dashboard |

Source of truth for this table is [`apps_registry.yaml`](apps_registry.yaml) — if they
ever diverge, the registry wins.

---

## 2. Current SSH Workflow (as-was before improvements)

```
On your local Mac/PC, open a tmux window with 2 panes:

  Pane 1 (remote shell):
    ssh new-login-1
    cd /home/sgsilva/video-sft-vlm
    source /home/sgsilva/vlm-post-training-home-venv/bin/activate
    lsof -ti:7863 | xargs -r kill -9
    DEFAULT_JSONL=/mnt/data/sgsilva/datasets/app_video_datasets/1805_merged_reasoning_v2_browse.jsonl \
      python app.py --port 7863

  Pane 2 (port tunnel):
    ssh -N -L 17863:localhost:7863 new-login-1

  Browser:
    http://0.0.0.0:17863/
```

**Pain points (now resolved by section 6):**
- 5 manual steps per app (kill → cd → activate → set env → launch)
- Separate pane just for the SSH tunnel
- Port-to-URL mapping non-obvious (remote 7863 → local 17863)
- Port 7863 was the "default" for 5+ different apps — only one at a time
- commands.txt had stale entries (all fixed — see section 4)

---

## 3. Per-App Detail

### 3.1 Monitoring Dashboard

**Path:** `/home/sgsilva/utilities/apps/monitoring-app/app.py`  
**Port:** 7861  
**Venv:** `/home/sgsilva/browser-app-home-venv/bin/python` (has gradio ✓)  
**Input:**
- Multi-source, all overridable via env vars:
  - `MONITOR_DATASETS_PATH` → `/mnt/data/shared/vlm/data/image_aux_datasets`
  - `MONITOR_RESULTS_PATH` → eval results dir
  - `MONITOR_MODELS_PATH` → `/mnt/data/sgsilva/models/`
  - `MONITOR_EXPERIMENTS_CSV` → `vlm-evaluation/experiments-final.csv`
- Read-only aggregation; no conversion needed  

**Launch:**
```bash
~/utilities/apps/launch_app.sh monitoring
# or manual:
cd /home/sgsilva/utilities/apps/monitoring-app
lsof -ti:7861 | xargs -r kill -9
/home/sgsilva/browser-app-home-venv/bin/python app.py --port 7861
```

---

### 3.2 Video SFT Browser (main browse app)

**Path:** `/home/sgsilva/utilities/apps/video_sft/app.py`  
**Port:** 7862  
**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python` (gradio ✓ plotly ✓ cv2 ✓ pyvis ✓)  
**Input:**
- `DEFAULT_JSONL` env var (fallback: first file in `app_video_datasets/`)
- **Quick-load dropdown** (added 2026-06-17): auto-scans `app_video_datasets/`, sorted by mtime. Select from dropdown to reload without restarting.

**Input schema:** `app_video_datasets/*.jsonl` — rows with `video`, `messages`, `metadata` fields.  
**How inputs are created:** `~/utilities/apps/scripts/make_app_video_dataset.py --source <hf> --name <n>`

**Launch:**
```bash
~/utilities/apps/launch_app.sh video-sft
# Switch dataset via dropdown inside the app — no restart needed
# Override dataset at launch:
~/utilities/apps/launch_app.sh video-sft DEFAULT_JSONL=/mnt/data/sgsilva/datasets/app_video_datasets/mcqa_1405_skel_partition_inspect.jsonl
```

---

### 3.3 Dataset Browser

**Path:** `/home/sgsilva/vlm-post-training/web/browser-app/app.py`  
**Port:** 7863  
**Venv:** `/home/sgsilva/browser-app-home-venv/bin/python`  
**Input:**
- `BROWSER_MIXED_ROOT` env var → HF dataset directory  
  Default: `reas2_mix_image10k_text5k_video10k_0804` (exists ✓ without `_audit` suffix)

**Launch:**
```bash
~/utilities/apps/launch_app.sh browser
```

---

### 3.4 VO Severity Comparator

**Path:** `/home/sgsilva/vlm-post-training/web/single_stage_compare_app.py`  
**Port:** 7864  
**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python`  
**Input:** CLI `--results "label=/path/to/eval.json"` (repeatable). Reads single-stage AND two-stage `stage2_*` JSONs.

**Launch:**
```bash
~/utilities/apps/launch_app.sh vo-severity --results "SFT ep3=/mnt/data/sgsilva/results/visual_obs_runs/sft27b_oracleobs_cat_ep3_stage1_thinkoff.json"
```

---

### 3.5 VO Stage-1 Comparator

**Path:** `/home/sgsilva/vlm-post-training/web/human_visual_obs_compare_app.py`  
**Port:** 7865  
**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python`  
**Input:** CLI `--results "label=/path/to/stage1_obs.json"` (repeatable).

**Launch:**
```bash
~/utilities/apps/launch_app.sh vo-compare \
  --results "SFT ep3=/mnt/data/sgsilva/results/visual_obs_runs/sft27b_oracleobs_cat_ep3_stage1_thinkoff.json" \
  --results "397B oracle=/mnt/data/sgsilva/results/visual_obs_runs/oracle_397b_1105_categorical.json"
```

---

### 3.6 SFT Data Browser

**Path:** `/home/sgsilva/utilities/apps/sft_text_data_browser.py`  
**Port:** 7866  
**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python`  
**Input:** Loads text MCQA families from an HF dataset root (configurable via `--dataset`).

**Launch:**
```bash
~/utilities/apps/launch_app.sh text-sft
```

---

### 3.7 LLM-FMS Viewer

**Path:** `/home/sgsilva/vlm-post-training/web/llm_fms_viewer.py`  
**Port:** 7867  
**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python`  
**Input:** CLI `--dataset /path/to/hf` required.

**Launch:**
```bash
~/utilities/apps/launch_app.sh llm-fms --dataset /mnt/data/sgsilva/datasets/<hf_dataset>
```

---

### 3.8 Multimodal Eval Comparer

**Path:** `/home/sgsilva/vlm-post-training/web/multimodal_compare_app.py`  
**Port:** 7868  
**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python`  
**Input:** Eval run directories with COCO keypoint overlays.

**Launch:**
```bash
~/utilities/apps/launch_app.sh multimodal-compare
```

---

### 3.9 Reas-Mix Inspector

**Path:** `/home/sgsilva/vlm-post-training/aux_tasks/sft/inspect/inspect_reas2_mix_gradio.py`  
**Port:** 7869  
**Venv:** `/home/sgsilva/browser-app-home-venv/bin/python`  
**Input:** `--root <merged_mix_dir>` CLI arg, passed via registry `args:` block.

**Launch:**
```bash
~/utilities/apps/launch_app.sh reas-inspector
```

---

### 3.10 Prejudge Viewer

**Path:** `/home/sgsilva/utilities/apps/video_sft/prejudge_viewer.py`  
**Port:** 7870  
**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python`  
**Input:** Prejudge smoke verdicts JSON + video dataset.

**Launch:**
```bash
~/utilities/apps/launch_app.sh prejudge-viewer
```

---

### 3.11 Mesh Viewer

**Path:** `/home/sgsilva/utilities/apps/video_sft/mesh_viewer.py`  
**Port:** 7871  
**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python`  
**Input:** SAM-3D overlay videos from the 3D batch pipeline.

**Launch:**
```bash
~/utilities/apps/launch_app.sh mesh-viewer
```

---

### 3.12 SAM3D Sword Viewer

**Path:** `/home/sgsilva/vlm-post-training/aux_tasks/video_tasks/video_mcqa/sam3d_pilot/sword_viewer.py`  
**Port:** 7872  
**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python` (fixed from dead Poetry venv)  
**Input:** SAM-3D pipeline output directories.

**Launch:**
```bash
~/utilities/apps/launch_app.sh sword-viewer
```

---

### 3.13 GRPO Dashboard

**Path:** `/home/sgsilva/utilities/apps/grpo_dashboard.py` (moved out of `nemo-rl-vlm/tools/` 2026-06-26 — self-contained, no `nemo_rl` imports; moved so SWORD-repo resets can't wipe the local **Live Status** tab)  
**Port:** 7873  
**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python` (fixed from `.venv` which had no gradio)  
**Input:** `--logs-dir` → default `/mnt/data/sgsilva/logs/grpo_logs/` (EXISTS ✓). Reads per-exp `train_data_step*.jsonl` + `val_data_step0.jsonl` + node logs.  
**Live Status tab (2026-06-26):** lists every run the moment it launches — phase, val@start accuracy (bar to beat), train-steps-done, live log tail — BEFORE step 1 makes it appear in *Compare Runs*.

**Launch:**
```bash
~/utilities/apps/launch_app.sh grpo-dashboard
```

---

### 3.14 Reasoning-Trace Prompt Editor

**Path:** `/home/sgsilva/vlm-post-training/web/reasoning_trace_prompt_app.py`  
**Port:** 7860  
**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python`  
**Input:** Hardcoded dataset path (one missing ⚠ — tool prompts for paths at runtime). Needs a served teacher VLM (default `http://worker-3:8000`).

**Launch:**
```bash
~/utilities/apps/launch_app.sh reasoning-prompt
```

---

### 3.15 Claude Usage Tracker

**Path:** `/home/sgsilva/utilities/apps/claude-tracker.py`  
**Port:** 8080  
**Venv:** system `python3`  
**Input:** Reads `~/.claude/projects/` (JSON session stats); shows token usage and cost estimates.

**Launch:**
```bash
~/utilities/apps/launch_app.sh claude-tracker
```

---

### 3.16 VLM Vibe Tester

**Path:** `/home/sgsilva/utilities/apps/vibe_test.py`  
**Port:** 7874  
**Venv:** `vlm-post-training-home-venv` (was video-sft-vlm-home-venv; merged + deleted 2026-06-30)  
**Goal:** Free-form inference playground — send any text / image / video to any served model and
inspect the answer + thinking trace. Used heavily for the EXP-B stage-2 template + teacher comparison.

**Inputs / features:**
- **Model source:** *Scan cluster* probes worker-0…31 on **ports 8000–8003** (not just 8000) and
  lists every live vLLM server with the **port-specific owner** (so two servers on one node are
  attributed correctly). Pick + *Use selected*, OR type a model name manually.
- **Vertex / gemini:** type a gemini model (`gemini-3-flash-preview`, `gemini-3.1-pro-preview`) and
  the call routes through the eval venv (`_vertex_call.py`) — the app venv lacks the GCloud SDK.
  Server URL is ignored for gemini; video is sent as a single mp4 (genai SDK).
- **Dataset dropdown:** lists HF Arrow datasets under `app_video_datasets/` (EXP-B sets pinned on
  top), fills the path box, builds the rep video, and loads system/user/assistant. A **stored
  `<think>` trace** (generated dataset) is split into the Thinking box; the bracket answer stays in GT.
- **Scoring:** auto-detects the GT format — stage-2 `[ERRORS]` → severity scoring; stage-1 numbered
  `[VISUAL OBSERVATIONS]` → per-line match. A **Canonical board metrics** box accumulates (gt,pred)
  across runs and computes the eval's own `compute_severity_metrics` (Error-F1/P/R/Acc, severity
  acc exact/within-1/non-1, eff/injury) via `_severity_metrics.py` in the eval venv. Reset button clears.

**Helper scripts (same dir):** `_vertex_call.py` (Vertex calls in the eval venv),
`_severity_metrics.py` (canonical metrics in the eval venv).

**Launch:**
```bash
~/utilities/apps/launch_app.sh vibe-test
```

**History:** improvements 2026-06-24 (this session) — multi-port scan, port-specific owner, dataset
dropdown, Vertex/gemini routing, stage-2 severity scoring + canonical board metrics, stored-trace split.

---

## 4. Stale Entries in commands.txt — Audit (All Fixed)

| Issue | Fix applied |
|-------|-------------|
| Monitoring app `cd /mnt/data/sgsilva/monitoring-app` | Changed to `cd /home/sgsilva/utilities/apps/monitoring-app` |
| SAM3D sword viewer used dead Poetry venv | Replaced with `/home/sgsilva/video-sft-vlm-home-venv/bin/python` |
| Data-tuning app block (4 lines) — `/home/sgsilva/data-tuning` doesn't exist | Removed |
| GRPO dashboard used `.venv/bin/python` (no gradio) | Replaced with `/home/sgsilva/vlm-post-training-home-venv/bin/python` |
| Dataset browser `BROWSER_MIXED_ROOT` had `_audit` suffix that doesn't exist | Removed `_audit` suffix |
| Claude tracker `cd /mnt/data/sgsilva/` | Changed to `cd /home/sgsilva/utilities/apps` |
| All manual fallback commands had wrong ports (collisions) | Updated to unique ports from the clean allocation table |

---

## 5. Port Allocation

### Before (chaotic — 7863 had 7 claimants)

| Port | Apps |
|:----:|------|
| 7862 | Video SFT browser, Dataset browser, SAM3D sword viewer, Multimodal eval comparer |
| 7863 | Mesh viewer, SFT data browser, VO stage-1, GRPO dashboard, Reas-mix inspector, Dataset browser |
| 7864 | VO severity comparator, LLM-FMS viewer |

### After (all unique — implemented 2026-06-17)

| Port | App | Code default updated? |
|:----:|-----|-----------------------|
| 7860 | Reasoning-trace prompt editor | — (already unique) |
| 7861 | Monitoring dashboard | — (already unique) |
| 7862 | Video SFT browser | — (already unique) |
| 7863 | Dataset browser | ✓ `browser-app/config.py` |
| 7864 | VO severity comparator | — (already unique) |
| 7865 | VO stage-1 comparator | ✓ `human_visual_obs_compare_app.py` |
| 7866 | SFT data browser | ✓ `utilities/apps/sft_text_data_browser.py` |
| 7867 | LLM-FMS viewer | ✓ `llm_fms_viewer.py` |
| 7868 | Multimodal eval comparer | ✓ `multimodal_compare_app.py` |
| 7869 | Reas-mix inspector | ✓ `inspect_reas2_mix_gradio.py` |
| 7870 | Prejudge viewer | — (already unique) |
| 7871 | Mesh viewer | ✓ `mesh_viewer.py` |
| 7872 | SAM3D sword viewer | ✓ `sword_viewer.py` |
| 7873 | GRPO dashboard | ✓ `grpo_dashboard.py` |
| 8080 | Claude usage tracker | — (already unique) |

All 16 apps can now run concurrently without port management overhead.

---

## 6. Speedup Proposals & Implementation Status

### 6.1 SSH Config with Pre-Wired Tunnels — **IMPLEMENTED** (local Mac `~/.ssh/config`)

```ssh
Host new-login-1
    HostName 89.124.37.171
    User sgsilva
    LocalForward 17860 localhost:7860
    LocalForward 17861 localhost:7861
    LocalForward 17862 localhost:7862
    LocalForward 17863 localhost:7863
    LocalForward 17864 localhost:7864
    LocalForward 17865 localhost:7865
    LocalForward 17866 localhost:7866
    LocalForward 17867 localhost:7867
    LocalForward 17868 localhost:7868
    LocalForward 17869 localhost:7869
    LocalForward 17870 localhost:7870
    LocalForward 17871 localhost:7871
    LocalForward 17872 localhost:7872
    LocalForward 17873 localhost:7873
    LocalForward 18080 localhost:8080
```

`ssh new-login-1` opens the interactive shell AND all tunnels simultaneously. **Pane 2 eliminated.**

### 6.2 Launcher Script — **IMPLEMENTED** (`~/utilities/apps/launch_app.sh`)

Registry-based script. Features as of 2026-06-17:
- Single command from anywhere — worker nodes auto-redirect to login-1 via SSH
- Kills the old process on the port before starting
- Launches in named tmux window inside persistent `app` session on login-1
- Prints full banner: label, goal, script, venv, port, browser URL, logs command
- `--list` to see all 16 apps + ports
- `--status` to see which apps are currently running (SSHes to login-1 from worker)
- Health-check after launch: polls the app for up to 30s and reports ready/failed

```bash
~/utilities/apps/launch_app.sh --list
~/utilities/apps/launch_app.sh --status
~/utilities/apps/launch_app.sh video-sft
~/utilities/apps/launch_app.sh video-sft DEFAULT_JSONL=/mnt/data/.../mcqa_1405.jsonl
~/utilities/apps/launch_app.sh vo-compare --results "SFT=/mnt/data/sgsilva/results/..."
```

### 6.3 Dataset Dropdown in Video-SFT App — **IMPLEMENTED** (`video-sft-vlm/app.py`)

- `_scan_browse_datasets()` scans `app_video_datasets/` sorted by mtime on startup
- `dataset_picker` Dropdown in the sidebar — selecting reloads immediately without restart
- Overridable via `APP_VIDEO_DATASETS_DIR` env var

### 6.4 Fixed-Port Discipline — **IMPLEMENTED** (9 files updated)

Code defaults updated so all 16 apps have unique ports and can run concurrently:
- `vlm-post-training/web/browser-app/config.py`: `7862` → `7863`
- `vlm-post-training/web/human_visual_obs_compare_app.py`: `7863` → `7865`
- `sft-data-vlm/app.py`: `7863` → `7866`
- `vlm-post-training/web/llm_fms_viewer.py`: `7864` → `7867`
- `vlm-post-training/web/multimodal_compare_app.py`: `7862` → `7868`
- `vlm-post-training/aux_tasks/sft/inspect/inspect_reas2_mix_gradio.py`: `7863` → `7869`
- `video-sft-vlm/mesh_viewer.py`: `7863` → `7871`
- `vlm-post-training/…/sam3d_pilot/sword_viewer.py`: `7862` → `7872`
- `nemo-rl-vlm/tools/grpo_dashboard.py`: `7863` → `7873`

---

## 6b. Remaining Improvements (not yet implemented)

### 6b.1 tmux Session Survival Across login-1 Reboots

**Problem:** The `app` tmux session lives in memory on login-1. If login-1 reboots (cluster maintenance, OOM kill), the session is gone and all running apps die silently. You only notice when the browser stops responding.

**Fix options:**
- Add to `~/.bashrc` on login-1: `tmux has-session -t app 2>/dev/null || tmux new-session -d -s app` — recreates the session automatically on every login, so the next `launch_app.sh` call just works.
- Use `tmux-resurrect` plugin to persist and restore window state across reboots (more complex, requires plugin install).
- The simpler approach (`.bashrc` snippet) is enough for this cluster — add it once on login-1.

**Effort:** 5 minutes. **Impact:** eliminates silent "why is my app gone?" confusion after cluster events.

### 6b.2 Login Node Pinning

**Problem:** The launcher hardcodes `login-1` as the target node. If you're ever allocated on login-0's side, or login-1 goes down, apps launched via the redirector land on a node your `~/.ssh/config` tunnels don't point at.

**Fix:** Make the login node configurable via an env var with a sensible default:
```bash
LAUNCH_LOGIN_NODE="${LAUNCH_LOGIN_NODE:-login-1}"
```
Then `LAUNCH_LOGIN_NODE=login-0 launch_app.sh video-sft` works without editing the script.
The `~/.ssh/config` would need a matching `Host new-login-0` block to be added locally.

**Effort:** 10 minutes. **Impact:** resilience when login-1 is unavailable or you're working from login-0.

### 6b.3 Restart-on-Crash

**Problem:** If an app crashes (OOM, missing dataset, uncaught exception), the tmux window stays open showing the traceback but nothing restarts the app. The browser just hangs.

**Fix:** Wrap the launch command in a simple retry loop inside the tmux window:
```bash
while true; do
    launch_app.sh video-sft
    echo "[crashed — restarting in 5s]"
    sleep 5
done
```
Could be added as a `--auto-restart` flag to `launch_app.sh`.

**Effort:** 30 minutes. **Impact:** useful for long-running sessions where you don't want to babysit apps.

### 6b.4 `--stop` and `--stop-all` Flags

**Problem:** No clean way to stop an app from the worker shell. Currently requires SSHing to login-1 and killing the tmux window or the process manually.

**Fix:** Add `--stop <app-name>` (kill the port process + close the tmux window on login-1) and `--stop-all` (stop every running app). Mirrors the existing `--status` SSH pattern.

**Effort:** 30 minutes. **Impact:** completes the lifecycle management — launch, status, stop all from one command.

### 6b.5 `--logs <app-name>` Flag

**Problem:** "Check logs: `ssh login-1 -t 'tmux attach -t app'`" is shown in the banner but requires knowing which window to navigate to once attached.

**Fix:** Add `--logs <app-name>` that SSHes directly into the correct tmux window:
```bash
launch_app.sh --logs video-sft
# → ssh login-1 -t 'tmux select-window -t app:video-sft && tmux attach -t app'
```
One command from the worker to tail the right app's output.

**Effort:** 15 minutes. **Impact:** removes the last manual step in the debug workflow.

---

## 7. App Tooling (~/utilities/apps/) — refreshed 2026-07-08

All app tooling consolidated under `~/utilities/apps/`:

| File | Purpose |
|------|---------|
| `launch_app.sh` | Launcher: single command to start any app; `--list`, `--status` |
| `apps_registry.yaml` | Registry of all 19 apps: name → workstream, repo, script, port, venv, env vars, args |
| `scripts/make_app_video_dataset.py` | Convert HF dataset to `*_browse.jsonl` for the video-sft app |
| `claude-tracker.py` | Claude Code token/cost tracker dashboard |
| `video_sft/` | `video-sft` app + siblings `prejudge_viewer.py`, `mesh_viewer.py` |
| `sft_dashboard.py`, `vobs_schema_inspector.py`, `sft_text_data_browser.py`, `grpo_dashboard.py`, `vibe_test.py` | Other self-contained standalone apps |
| `README.md` | Quick-start guide for the apps/ subdir (current-state source of truth) |
| `GRADIO_APPS_REPORT.md` | This file — historical audit + refreshed quick-reference |

---

## 8. Dataset Creation Index (refreshed 2026-07-08)

| App | Input data | How it's created |
|-----|-----------|-----------------|
| video-sft | `app_video_datasets/*.jsonl` | `~/utilities/apps/scripts/make_app_video_dataset.py --source <hf> --name <name>` |
| monitoring | Results CSVs + experiments CSV | Written by results pipeline (`build_results_csv.py`) |
| sft-data | HF text MCQA datasets | `aux_tasks/text_tasks/generation/generate_text_sft_datasets.py` |
| browser | HF aux datasets (image/video/text) | Various dataset-builder scripts in `vlm-post-training/aux_tasks/` |
| vo-severity / vo-compare | Eval JSONs (or `--raw-cohort` for vo-compare) | `evaluate.py` — see Skill `/eval-vlm` |
| reasoning-prompt | HF annotation datasets + live VLM | Static datasets; server started via `/serve-vllm` |
| image-viewer | HF image+text dataset | Various SFT dataset builders |
| multimodal-compare | Eval run directories | `evaluate.py` multimodal eval runs |
| reas-inspector | HF merged reas2 mix | Reas2-mix generation pipeline |
| prejudge-viewer | Prejudge smoke verdicts JSON | Prejudge pipeline |
| mesh-viewer / sword-viewer | SAM-3D pipeline outputs | `/run-3d` skill batch inference pipeline |
| grpo-dashboard | GRPO training logs (`/mnt/data/sgsilva/logs/grpo_logs/`) | Written automatically by GRPO training loop |
| sft-dashboard | Tensorboard event files (`nemo-rl-vlm/logs/<run>/exp_NNN/tensorboard`) | Written automatically by SFT training loop |
| vobs-schema | `visual_observations_{angle,categorical}_<version>.json` | VObs schema authoring pipeline; `--version` selects |
| rft-harvest | `.ckpt.jsonl` checkpoint sidecar | `harvest_stage2_rft_traces.py` (pass `--input` at launch) |
| multiturn-tools | Cached stage-1 GT obs bank (2906/1806) | `visual_obs/query_obs_tool_executor.py` |

---

## 9. Venv Summary (refreshed 2026-07-08)

| Venv | Has gradio | Has plotly | Has cv2 | Used by |
|------|:---:|:---:|:---:|---------|
| `vlm-post-training-home-venv` | ✓ | ✓ | ✓ | Most apps: video-sft, mesh/prejudge/sword viewer, vibe-test, all `web/` compare apps, sft-data, grpo-dashboard, reasoning-prompt, image-viewer, vobs-schema, rft-harvest, multiturn-tools (`video-sft-vlm-home-venv` merged in + deleted 2026-06-30) |
| `browser-app-home-venv` | ✓ | ? | ? | monitoring, browser, reas-inspector |
| `nemo-rl-vlm/.venv` | ✗ (has tensorboard) | — | — | sft-dashboard ONLY — needs the tensorboard event-file reader; ❌ DO NOT USE for any other Gradio app (no gradio) |
| `nemo-rl-vlm-grpo-home-venv` | ✗ | ✗ | ✗ | GRPO training only |
| `sam3d-home-venv` | ✗ | ✗ | ✗ | SAM-3D batch inference only |
| `qwen3.5-serving-home-venv` | ✗ | ✗ | ✗ | vLLM serving only |

---

## 10. Legacy / Auxiliary Apps (Not Actively Used)

- `/home/sgsilva/vlm-post-training/web/gradio_app.py` — older exercise analysis UI, superseded
- `/home/sgsilva/vlm-post-training/web/gradio_app_multiturn.py` — multi-turn chat viewer, occasional use
- `/home/sgsilva/vlm-post-training/web/comparison_app.py` — older same-member video comparison, superseded by multimodal_compare_app.py
- `/home/sgsilva/4D-Humans/gradio_app.py` — upstream 4D-Humans repo demo, not Sandra's code
- `/home/sgsilva/vlm-post-training/aux_tasks/transcripts/scripts/app.py` — narrow transcript inspection tool
- `/home/sgsilva/vlm-post-training/aux_tasks/sft/inspect/inspect_reas_judged_mix_gradio.py` — variant of the reas-mix inspector for judged data

---

*End of report.*
