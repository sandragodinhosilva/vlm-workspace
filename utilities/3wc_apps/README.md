# `~/utilities/3wc_apps` — 3WC app tooling

Launcher for the **3WC (3-Way Chat / Thrive) trace viewer**. Deliberately separate from
`~/utilities/apps/` (the VLM Gradio launcher + registry): 3WC is a different workstream
whose app lives in the SWORDHealth `dawn-research` repo, and `CLAUDE.local.md`'s membrane
rule keeps VLM mechanism out of 3WC. Nothing here imports `launch_app.sh` or
`apps_registry.yaml`, and nothing there knows about this.

> **Why the launcher lives here and not next to the app.** `dawn-research` is a SWORDHealth
> repo I never push to. Keeping the launcher in `~/utilities/` means no untracked file sits
> in that repo waiting to be swept into a commit.

## Quick start

```bash
# On the login node:
~/utilities/3wc_apps/run_app.sh                    # full merged corpus, port 7860
~/utilities/3wc_apps/run_app.sh --status -p 7860   # is it up?
~/utilities/3wc_apps/run_app.sh --stop   -p 7860   # stop it

# On your local Mac (tunnel — SAME port both sides; middle host = the node it runs on):
ssh -N -L 7860:login-1:7860 new-login-0
# Then open: http://localhost:7860
```

`run_app.sh` prints the exact tunnel line for the host and port it actually started on —
copy that rather than reconstructing it. The middle hostname matters: if the app is on
`login-1` and you land on a different login node, `localhost` there won't find it.

## The app

| | |
|---|---|
| Source | `/home/sgsilva/dawn-research/3wc/app.py` (Gradio; `core/` holds the shared logic) |
| What it shows | Langfuse production traces: Member ↔ **Phoenix** ↔ Clinical Specialist, plus per-turn tool calls, escalations, task updates |
| Views | "By patient (merged)" — all of a member's turns unioned into one chronological conversation with a 🧠 card per turn; "By single trace" — one raw turn |
| Corpus | `/mnt/data/shared/3wc/` (raw exports under `raw/`) — **read-only, never write there** |

## Usage

| Command | What |
|---|---|
| `run_app.sh` | Launch in background on port 7860, merged corpus (`.index_files`) |
| `run_app.sh -p 7862` | Pick a port |
| `run_app.sh --era 0907` | Index **only** the 0907 export (skip 2206/2606) — the era the eval/test sets are built from |
| `run_app.sh --fg` | Foreground (Ctrl-C to stop) |
| `run_app.sh --status [-p N]` | Running? PID, CPU, RSS, log path |
| `run_app.sh --stop [-p N]` | Stop **your** instance on that port (ownership-checked) |
| `run_app.sh --help` | Usage |

Overridable env: `APP_DIR`, `DATA_DIR`, `PYTHON`, `LOG_DIR`, `TMP_3WC`, `ONLY_UNIT` (default `thrive`).

## The index cache — why the first launch is slow

The corpus is ~34 GB across 4 JSONLs (~60k traces). The app never loads it into memory:
it builds a **byte-offset index** once, caches it to `3wc/.trace_index/`, and thereafter
seeks to each trace on demand.

- **Cold start** ≈ 4 min (full merged corpus). **Warm start** ≈ 9 s. Measured 2026-07-27.
- Cache is **per source file**, keyed on `(size, mtime, INDEX_VERSION)` — adding or growing
  one export re-scans only that file.
- Force a rebuild by deleting `3wc/.trace_index/`.
- That directory is inside the SWORDHealth repo and is **not** in the tracked `.gitignore`.
  It is excluded via `dawn-research/.git/info/exclude`. ⚠️ `.git/info/exclude` does **not**
  survive a re-clone — re-add `3wc/.trace_index/` if the repo is ever re-cloned, or send a
  `.gitignore` line upstream as the durable fix.

## Where things get written

Nothing is written under `/mnt/data/pmartins/` — the app opens the JSONLs read-only and seeks.

| What | Where |
|---|---|
| Trace index cache | `dawn-research/3wc/.trace_index/` (git-excluded) |
| App logs | `/mnt/data/sgsilva/logs/3wc/app_<port>_<timestamp>.log` |
| Gradio temp | `/home/sgsilva/tmp/gradio_3wc` (override with `TMP_3WC`) |

`GRADIO_TEMP_DIR` is set **unconditionally**, not with a `:-` default. An ambient
`GRADIO_TEMP_DIR=/tmp/gradio_sgsilva` in the shell would otherwise win and silently put
temp files back on `/tmp`, which is a shared mount.

## Gotchas

- **`pgrep -f` matches wrapper shells.** A `bash -c '… python3 app.py --port 7860 …'`
  wrapper (nohup, this launcher, a Claude Bash call) has the whole command in its *own*
  cmdline, so `pgrep -f` returns it too — reporting ~0% CPU and looking like a stalled app.
  `pid_on_port()` discriminates on **argv[0]** (must be the interpreter) and argv[1] (must
  be `app.py`). If you check by hand, verify with `/proc/<pid>/cmdline`.
- **Login node, not a worker.** Indexing is I/O-heavy. Fine when the node is quiet; if it
  gets busy, run it under `srun` on a worker or scope with `--era 0907`.
- **`anthropic` / `google.genai` are optional.** Lazy imports on the Vertex judge paths only
  — the viewer runs fine without them. They *are* needed by the eval/judge scripts.
- **Port hygiene.** The launcher refuses an occupied port rather than killing what's there
  (unlike VLM's `launch_app.sh`, which kills first) — a shared node may host someone else's
  process. Use `--stop` deliberately; it checks ownership.

## Related

- Status reference for the whole project: `~/.claude/reports/3wc/2026-07-27_3wc_status_reference.md`
- My personal 3WC hub: `~/.claude/3WC_HOME.md` · index: `~/.claude/3WC_DOC_INDEX.md`
- Team-authoritative context: `dawn-research/AGENTS.md` + `.knowledge/`
- The pipeline scripts this app's corpus feeds: `dawn-research/3wc/scripts/`
