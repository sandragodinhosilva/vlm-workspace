# `scripts/` — app tooling & helpers

Code used *by* or *alongside* the apps in this directory, but never launched as a Gradio app
via `launch_app.sh` (so it doesn't belong at the root next to the registry entrypoints).

## Standalone tools (run by hand)

| File | What |
|------|------|
| `make_app_video_dataset.py` | Convert an HF dataset → a `*_browse.jsonl` for the `video-sft` browse app. Writes to `/mnt/data/sgsilva/datasets/app_video_datasets/`. Run with `vlm-post-training-home-venv/bin/python`; see `--help`. |

## Private subprocess helpers (invoked by an app)

| File | Used by | What |
|------|---------|------|
| `_vertex_call.py` | `vibe_test.py` | Vertex/Gemini call helper (run in the eval venv). |
| `_severity_metrics.py` | `vibe_test.py` | Runs `eval.compute_severity_metrics` (needs sklearn from the eval venv). |

Both helpers are invoked as subprocesses by `vibe_test.py` via `EVAL_VENV_PY <helper>` (paths set
near the top of `vibe_test.py`), and they resolve `vlm-post-training` by absolute path — so they
are location-independent. If you move them, update the `VERTEX_HELPER` / `METRICS_HELPER` paths in
`vibe_test.py`.
