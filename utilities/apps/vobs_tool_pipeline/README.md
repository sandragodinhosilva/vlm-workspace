# vobs-tool-pipeline — VObs-tool-SFT pipeline inspector

Gradio app (port **7880**) for reviewing the VObs-tool-SFT synthetic pipeline row by row:
the rep video built from the row's OWN `video_frames` (fps + `need_to_flip` honored), the
verbatim prompt/raw/parsed triple per stage (generation best-of-K → rewrite → 3-judge
cascade → regen), the per-step `step_metrics` table, and the final shipped messages.

Pipeline diagram: [`vlm-post-training/visual_obs/workflow_tool_use.mmd`](/home/sgsilva/vlm-post-training/visual_obs/workflow_tool_use.mmd)
(rendered live in the app's **App Guidance** tab; PNG/SVG in
`/mnt/data/sgsilva/results/visual_obs/diagrams/`).

---

## Launch

**Launch detached.** The app must outlive the shell that starts it:

```bash
cd /home/sgsilva/utilities/apps/vobs_tool_pipeline
setsid nohup /home/sgsilva/vlm-post-training-home-venv/bin/python app.py \
  > /mnt/data/sgsilva/logs/misc/$(date +%F)/app_pipeline_inspector_7880.log 2>&1 < /dev/null &
```

`launch_app.sh pipeline-inspector` also works and reads the registry entry, but **if the
launcher runs as a child of a short-lived shell (e.g. an agent tool call), the app dies
when that parent is torn down.** Observed 2026-07-28: the app bound 7880, served HTTP 200,
then vanished with its parent. `setsid` reparents to PPID 1 and survives.

Verify it came up (Gradio buffers stdout — an empty log is normal, not a failure):

```bash
ss -ltnp | grep 7880
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7880/
```

**Tunnel** (from the Mac):

```bash
ssh -N -L 17880:localhost:7880 login-1
```

Then open <http://localhost:17880>.

**Stop:** `kill <pid>` — take the PID from `ss -ltnp | grep 7880`. Verify ownership first
(`ps -o user= -p <pid>`); never kill another user's process.

**Venv:** `/home/sgsilva/vlm-post-training-home-venv/bin/python` (the eval/dataset venv —
the app reuses the producer's own `step_metrics`/preview code). Do **not** use the SFT venv.

---

## Which run it loads

Resolution order for the startup run:

1. `DEFAULT_JSONL` env var (set by `apps_registry.yaml`, or exported manually)
2. the in-code default in `app.py` (currently `smoke_review_mix_0717/smoke.jsonl`)
3. fallback: newest clean run under `DATASET_ROOT` — self-healing, so a removed run
   can't leave a stale pin that 500s the auto-load

The sibling `.dropped.jsonl` auto-loads alongside the kept rows; both are tagged with
`_disposition` so you can filter kept vs dropped.

**Canonical review run — `smoke_review_mix_0717`** (100 rows: 59 kept + 41 dropped). The
curated, all-flavor colleague-review mix; buggy drops excluded, so every drop shown is the
*designed* quality gate. It is **deliberately kept-leaning (59%) and is NOT a yield
benchmark** — for the true ratio use `smoke_allflavors_fixed_0717` (untrimmed 100-rep run,
~35% yield). See that run's own `README.md`.

**⚠️ Registry drift.** `apps_registry.yaml`'s `DEFAULT_JSONL` pointed at
`smoke_stage4_0716c/`, a run since deleted from disk (fixed 2026-07-28 → repointed to
`smoke_review_mix_0717`). Because the env var *overrides* the in-code default, a stale
registry entry silently beats a correct default. **When a run is renamed or deleted, update
the registry entry too** — or unset `DEFAULT_JSONL` and let the in-code default win.

Override for a one-off:

```bash
DEFAULT_JSONL=/mnt/data/sgsilva/datasets/1806/vobs_tool_sft_4k/<run>/smoke.jsonl \
  setsid nohup .../python app.py > <log> 2>&1 < /dev/null &
```

The dropdown is intentionally pinned to the curated run (colleague-review mode). Use the
collapsed **"Advanced: load a run by path"** box to open any other run for debugging.

---

## Env vars

| Var | Default | What |
| --- | --- | --- |
| `DEFAULT_JSONL` | `smoke_review_mix_0717/smoke.jsonl` | startup run (kept-rows jsonl) |
| `DATASET_ROOT` | `/mnt/data/sgsilva/datasets/1806/vobs_tool_sft_4k` | scanned for the ↻ Runs fallback |
| `VIDEO_CACHE_DIR` | `/mnt/data/sgsilva/tmp/vobs_tool_pipeline_videos` | rendered rep videos |

CLI: `--port 7880` · `--host 0.0.0.0` · `--share` · `--no-ssr`.

---

## Gotchas

- **Empty log ≠ failure.** Gradio buffers stdout; the log stays quiet until a request
  generates output. Check the port and an HTTP 200 instead. Logs also lag ~10s — use
  `tail -F`.
- **Startup is slow.** The curated run is ~108 MB of JSONL (kept + dropped); binding 7880
  takes tens of seconds. Wait on the port, don't assume it failed.
- **The app only MIRRORS the data.** A row missing its self-describing fields shows a loud
  pipeline-gap state — it never reconstructs a path. If a row looks wrong, suspect the
  producer (`run_tool_sft_4k.py`), not the app.
- **Gradio 6.0** moved `theme`/`css`/`js` off the `Blocks` constructor onto `launch()` —
  they're applied in `main()`, not at Blocks build time.

---

## Related

- Producer: `vlm-post-training/visual_obs/run_tool_sft_4k.py` (gen → rewrite → judge → regen)
- Text-dump exporter: `vlm-post-training/visual_obs/preview_tool_sft_pipeline.py`
- Registry: `utilities/apps/apps_registry.yaml` · launcher `utilities/apps/launch_app.sh`
- Skill: `/vlm-apps`
