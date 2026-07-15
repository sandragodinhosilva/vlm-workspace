# `scripts/` — app tooling & helpers

Code used *by* or *alongside* the apps in this directory, but never launched as a Gradio app
via `launch_app.sh` (so it doesn't belong at the root next to the registry entrypoints).

## Standalone tools (run by hand)

| File | What |
|------|------|
| `make_app_video_dataset.py` | Convert an HF dataset → a `*_browse.jsonl` for the `video-sft` browse app. Writes to `/mnt/data/sgsilva/datasets/app_video_datasets/`. Run with `vlm-post-training-home-venv/bin/python`; see `--help`. |
| `snapshot_sft_config.py` | Snapshot an SFT config into `nemo-rl-vlm/logs/<run>/config_snapshot.yaml` at launch, so the `sft-dashboard` (port 7875) config panel shows config-as-run. Called by the `/launch-sft` workflow right after config verification. Run with the nemo `.venv`; see `--help`. |

## Private subprocess helpers (invoked by an app)

| File | Used by | What |
|------|---------|------|
| `_vertex_call.py` | `vibe_test.py` | Vertex/Gemini call helper (run in the eval venv). |
| `_severity_metrics.py` | `vibe_test.py` | Runs `eval.compute_severity_metrics` (needs sklearn from the eval venv). |

Both helpers are invoked as subprocesses by `vibe_test.py` via `EVAL_VENV_PY <helper>` (paths set
near the top of `vibe_test.py`), and they resolve `vlm-post-training` by absolute path — so they
are location-independent. If you move them, update the `VERTEX_HELPER` / `METRICS_HELPER` paths in
`vibe_test.py`.

## Shared library, imported directly (not subprocessed)

| File | Used by | What |
|------|---------|------|
| `row_video.py` | `vobs_tool_pipeline/app.py` (pipeline-inspector) | Row-own-frames → playable MP4 for SELF-DESCRIBING datasets (row carries `video_frames`/`fps`/`need_to_flip`, e.g. run_tool_sft_4k.py output). `encode_video()` lifted from `video_sft/app.py` (fps-correct `-r` container rate + hflip mirror); `build_row_video(row, cache_dir)` returns `(mp4_path|None, loud_status)` — a missing field yields a distinct pipeline-gap message, never a guessed path or defaulted fps ([[feedback_no_silent_fail]]). If you fix a bug in the encoder here, mirror it in video_sft's copy. |
| `nav_widgets.py` | `image_row_viewer.py` (image-viewer), `inspect_reas2_mix_gradio.py` (reas-inspector), `vobs_tool_pipeline/app.py` (pipeline-inspector) | Prev/next/random/jump-to-index nav — component factories (`make_nav_row`, `make_jump_row`) + pure nav logic (filtered-index resolution, step/random-within-filter, counter formatting). 2026-07-08 homogenization pass (§ Priority 1, `GRADIO_APPS_REPORT.md`) — extracted from `browser-app/shared/components.py` + the inline nav in `image_row_viewer.py`, merging both into one module rather than a third copy. Import directly (same venv as gradio, no subprocess needed): `sys.path.insert(0, "/home/sgsilva/utilities/apps/scripts"); import nav_widgets`. `format_scoped_counter()` is the house convention for the position counter — prefer it over inventing a new string format in a new app. |
| `live_inference.py` | `vibe_test.py` (vibe-test, canonical caller), `gradio_app_multiturn.py` (multiturn-tools, via `vibe_test` re-export) | Cluster-scan for live vLLM servers (`scan_cluster`, `apply_scan_selection`) — SSH-probes worker-0..31 on ports 8000-8003, resolves per-port owner (sentinel `"unknown"` on permission wall, never a guessed owner). 2026-07-08 homogenization pass — extracted out of `vibe_test.py`, which now imports it back (`from live_inference import scan_cluster, apply_scan_selection`) rather than defining it inline. `gradio_app_multiturn.py` continues to import from `vibe_test.py` (unchanged import line — it transparently re-exports), so it picks this up without a direct dependency on `scripts/`. `GEMINI_MODELS` stays in `vibe_test.py` (not moved) since it's paired with vibe-test's own Vertex-subprocess routing, not this module. Model-CALLING (the actual vLLM/Vertex request) is a separate, larger shared surface already centralized in `vlm-post-training/inference/query_server.py` — not duplicated here.
