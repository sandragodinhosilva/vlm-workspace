# utilities/serve — canonical VLM serving tools

Home of Sandra's vLLM serving launcher + query client (moved here 2026-06-23 from the now-archived
`vlm-evaluation` repo).

- **`start_vllm_server.sh`** — the vLLM server launcher. GPU/port/path preflights, `ENABLE_THINKING`
  + `--reasoning-parser qwen3`, `SERVED_MODEL_NAME` long-path alias, startup heartbeat, Qwen3.5 /
  GLM-4.7 / Kimi branches. Usage:
  ```bash
  ENABLE_THINKING=<0|1> QWEN35_VENV=/home/sgsilva/qwen3.5-serving-home-venv \
    /home/sgsilva/utilities/serve/start_vllm_server.sh <MODEL_PATH> <TP_SIZE> <MAX_MODEL_LEN> <PORT>
  ```
  Recipe + thinking-mode-match rule: `~/.claude/skills/serve-vllm/SKILL.md` (`/serve-vllm`).
- **`query_server.py`** — the serving/eval query client (litellm + Gemini/Vertex AI + direct-vLLM).
  The `MODELS` dict `"model"` alias must equal the exact served path (which is the full path string
  `start_vllm_server.sh` passes, since it does not set `--served-model-name`).
- **`MIGRATION_vlm-evaluation.md`** — why these moved + what's still pending in the archive.

Don't point anything at `/home/sgsilva/vlm-evaluation/...` — that repo is being archived.
