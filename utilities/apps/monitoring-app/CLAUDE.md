# monitoring-app (vlm-monitoring-app)
**What this repo does:** Gradio monitoring dashboard covering dataset creation progress, SFT training metrics, and eval results across checkpoints.
**Active branch / thread:** `main` — live dashboard; image_id match fix and benchmark-tab baseline parsing are shipped.
**Entry points:**
- `app.py` — main Gradio dashboard
- `start_app.sh` — convenience launch script
**Which venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python` (full table: ~/.claude/CLAUDE.md).
**Knowledge (read first):** ~/.claude/projects/-home-sgsilva/memory/MEMORY.md
  Active threads: [[project_monitoring_app_image_id_match_fix_decision]] · [[project_monitoring_app_benchmark_tab_baseline_fixes_decision]] · [[project_results_csv_manifest]]
**Connects to:** reads results CSVs produced by `vlm-post-training/data_preparation/`; gallery images come from eval output dirs.
**Ownership:** personal fork → committed to sandragodinhosilva/vlm-monitoring-app
