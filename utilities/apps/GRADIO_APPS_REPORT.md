# Gradio Data-Inspection Apps — Full Audit Report

**Date:** 2026-06-17  
**Scope:** All Gradio apps across Sandra's home repos; current workflow; staleness audit; speedup proposals; implemented changes.

---

## 1. Quick-Reference Table

| App | Path | Port | Venv | Goal | Created | Last commit | Commits | Status |
|-----|------|:----:|------|------|:-------:|:-----------:|:-------:|--------|
| Monitoring dashboard | `apps/monitoring-app/app.py` | 7861 | `browser-app-home-venv` | Pipeline health dashboard — datasets, training metrics, eval results in one view | 2026-02-05 | 2026-04-09 | 24 | ✓ |
| Video SFT browser | `utilities/apps/video_sft/app.py` | 7862 | `vlm-post-training-home-venv` | Browse generated MCQA video samples; inspect exercise types, tier, messages | 2026-02-19 | 2026-06-30 | 27 | ✓ active (relocated from video-sft-vlm 06-30) |
| Dataset browser | `vlm-post-training/web/browser-app/app.py` | 7863 | `browser-app-home-venv` | Multi-tab browser for all post-training datasets: image, video, text, reasoning tabs | 2026-03-06 | 2026-06-16 | 8 | ⚠ BROWSER_MIXED_ROOT dataset not on disk |
| VO severity comparator | `vlm-post-training/web/single_stage_compare_app.py` | 7864 | `vlm-post-training-home-venv` | Compare two single/two-stage eval runs side-by-side: per-rep error detection + video | 2026-05-22 | 2026-05-29 | 5 | ✓ |
| VO stage-1 comparator | `vlm-post-training/web/human_visual_obs_compare_app.py` | 7865 | `vlm-post-training-home-venv` | Side-by-side visual observations across multiple VLMs vs human ground-truth | 2026-05-04 | 2026-05-28 | 6 | ✓ |
| SFT data browser | `utilities/apps/sft_text_data_browser.py` | 7866 | `vlm-post-training-home-venv` | Browse and QC text-based SFT datasets (5 MCQA families) from Thrive VLM database | 2026-02-27 | 2026-04-09 | 6 | ✓ |
| LLM-FMS viewer | `vlm-post-training/web/llm_fms_viewer.py` | 7867 | `vlm-post-training-home-venv` | Inspect image-based LLM-FMS SFT rows: image + prompt + target + metadata | 2026-05-22 | 2026-05-22 | 2 | ✓ |
| Multimodal eval comparer | `vlm-post-training/web/multimodal_compare_app.py` | 7868 | `vlm-post-training-home-venv` | Side-by-side multimodal eval run comparison with COCO keypoint overlay | 2026-04-21 | 2026-04-21 | 2 | ⚠ hardcoded default dirs stale |
| Reas-mix inspector | `vlm-post-training/aux_tasks/sft/inspect/inspect_reas2_mix_gradio.py` | 7869 | `browser-app-home-venv` | Quick inspector for the reas2 merged reasoning mix: filter by modality/judge verdict | 2026-04-21 | 2026-04-21 | 1 | ✓ |
| Prejudge viewer | `utilities/apps/video_sft/prejudge_viewer.py` | 7870 | `vlm-post-training-home-venv` | Inspect LLM prejudge smoke verdicts alongside video frames and post-hoc labels | 2026-06-17 | 2026-06-30 | 1 | ✓ (relocated 06-30) |
| Mesh viewer | `utilities/apps/video_sft/mesh_viewer.py` | 7871 | `vlm-post-training-home-venv` | Browse SAM-3D-Body overlay videos: mesh render + interactive 3D skeleton + angle signals | 2026-06-17 | 2026-06-30 | 1 | ✗ BROKEN at source (imports `_sam3d_output_dir` never merged into app.py); relocated 06-30 |
| SAM3D sword viewer | `vlm-post-training/…/sam3d_pilot/sword_viewer.py` | 7872 | `vlm-post-training-home-venv` | Browse SWORD SAM-3D pipeline outputs: 2D overlay + interactive 3D skeleton + mesh | 2026-05-13 | 2026-06-30 | 1 | ✓ (venv → vlm-post-training-home-venv 06-30) |
| GRPO dashboard | `utilities/apps/grpo_dashboard.py` (moved 06-26) | 7873 | `vlm-post-training-home-venv` | Explore and compare GRPO training runs: reward curves, task breakdowns | 2026-06-01 | 2026-06-03 | 4 | ✓ fixed (was broken venv) |
| Reasoning-trace prompt editor | `vlm-post-training/web/reasoning_trace_prompt_app.py` | 7860 | `vlm-post-training-home-venv` | Iterate on reasoning-trace prompts with inline editing + live VLM calls for testing | 2026-05-22 | 2026-05-28 | 3 | ⚠ one default dataset path missing |
| Claude usage tracker | `utilities/apps/claude-tracker.py` | 8080 | system python3 | Local dashboard for Claude Code token usage and cost estimates | — | — | — | ✓ |
| VLM vibe tester | `utilities/apps/vibe_test.py` | 7874 | `vlm-post-training-home-venv` | Free-form inference playground: text/image/video → any served vLLM **or Vertex/gemini** model; cluster scan, dataset dropdown, stage-2 canonical metrics vs GT | 2026-06 | 2026-06-30 | — | ✓ active (venv merged 06-30) |

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

## 7. App Tooling (~/utilities/apps/)

All app tooling consolidated under `~/utilities/apps/`:

| File | Purpose |
|------|---------|
| `launch_app.sh` | Launcher: single command to start any app |
| `apps_registry.yaml` | Registry of all 16 apps: name → repo, script, port, venv, env vars |
| `make_app_video_dataset.py` | Convert HF dataset to `*_browse.jsonl` for the video-sft app |
| `claude-tracker.py` | Claude Code token/cost tracker dashboard |
| `README.md` | Quick-start guide for the apps/ subdir |
| `GRADIO_APPS_REPORT.md` | This file |

---

## 8. Dataset Creation Index

| App | Input data | How it's created |
|-----|-----------|-----------------|
| Video SFT browser | `app_video_datasets/*.jsonl` | `~/utilities/apps/scripts/make_app_video_dataset.py --source <hf> --name <name>` |
| Monitoring dashboard | Results CSVs + experiments CSV | Written by results pipeline (`build_results_csv.py`) |
| SFT data browser | HF text MCQA datasets | `aux_tasks/text_tasks/generation/generate_text_sft_datasets.py` |
| Dataset browser | HF aux datasets (image/video/text) | Various dataset-builder scripts in `vlm-post-training/aux_tasks/` |
| VO severity / stage-1 comparators | Eval JSONs | `evaluate.py` — see Skill `/eval-vlm` |
| Reasoning-trace prompt editor | HF annotation datasets + live VLM | Static datasets; server started via `/serve-vllm` |
| LLM-FMS viewer | HF image+text dataset | Various SFT dataset builders |
| Multimodal eval comparer | Eval run directories | `evaluate.py` multimodal eval runs |
| Reas-mix inspector | HF merged reas2 mix | Reas2-mix generation pipeline |
| SAM3D sword viewer | SAM-3D pipeline outputs in `tmp/` | SAM-3D batch inference pipeline |
| GRPO dashboard | GRPO training logs | Written automatically by GRPO training loop |

---

## 9. Venv Summary

| Venv | Has gradio | Has plotly | Has cv2 | Used by |
|------|:---:|:---:|:---:|---------|
| `video-sft-vlm-home-venv` | ✓ | ✓ | ✓ | video SFT apps + mesh + prejudge + sword viewer |
| `vlm-post-training-home-venv` | ✓ | ✗ | ✓ | all web/compare apps + GRPO dashboard |
| `browser-app-home-venv` | ✓ | ? | ? | monitoring-app, dataset browser, reas-mix inspector |
| `nemo-rl-vlm/.venv` | ✗ | ✗ | ✗ | ❌ DO NOT USE for Gradio apps |
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
