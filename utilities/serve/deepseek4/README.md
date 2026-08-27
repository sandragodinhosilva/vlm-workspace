# Serving DeepSeek-V4-Flash (vLLM + Slurm)

run:
sbatch /home/sgsilva/utilities/serve/deepseek4/deepseek-v4-flash.sh

This is **jmendonca's recipe**, copied from `/home/jmendonca/servers/deepseek-v4-flash.sh` with
two hunks changed. All vLLM flags are his, verbatim. `diff` against his file to see the delta.

| | |
|---|---|
| **model** | `deepseek-v4-flash` |
| **api base** | `http://<worker>:9100/v1` (Slurm picks the worker) |

Find the node once it is running: `squeue --name deepseek-v4-flash -o '%N'`
Follow startup: `tail -f run/slurm-<jobid>.log` — expect **many minutes** (149 GB of weights +
DeepGEMM MoE autotune). Stop with `scancel <jobid>`.

## What was changed from his, and why

1. **`uv run vllm` → `.venv/bin/vllm` (absolute).** `uv run` resolves against `uv.lock` in the
   project dir, and his dir is not writable by us. Same vLLM build either way.
2. **HF repo id → resolved local snapshot path.** The weights are already on shared storage
   (149 GB, 46/46 shards, complete), so this avoids a redundant pull and cannot drift to a
   newer hub revision.
3. Added `--served-model-name` + `--host 0.0.0.0 --port 9100`; without them the server is not
   cluster-reachable and collides with whatever else is on the default 8000. **9100 not 9000** —
   9000 is the long-lived GLM judge and `kimi-k3`.
4. `--output` points at `run/` instead of the submitting CWD.

His `--mem=2488G` header and all 11 model-behaviour flags are untouched.

## The venv is SHARED with kimi3

`.venv` is a **symlink to `../kimi3/.venv`**. That vLLM (`0.27.2rc1.dev2+g34735aced`) already ships
every DeepSeek-V4 component — `tokenizers/deepseek_v4.py`, `renderers/deepseek_v4.py`,
`deepseekv4_engine_tool_parser`, `deepseek_v4_engine_reasoning_parser`, and the `sparse_swa` MLA
backend — so a second 7.9 GB copy would buy nothing.

⚠️ The coupling is real: upgrading kimi3's venv changes what this serves. If the two ever need
different vLLM versions, replace the symlink with a real venv (see `kimi3/README.md`).

## Before quoting any number from this server

- **`--max-model-len` is NOT set**, so it inherits the model max (1048576). The official vLLM
  recipe gives **393216 (384K)** as the floor for Think High / Think Max; 1M clears it. Pinning it
  lower would trade context for KV headroom/concurrency — measure before assuming that is a win.
- **Reasoning effort is a per-REQUEST knob, not a server flag.** Non-think / Think High / Think Max
  are selected via `chat_template_kwargs`, so the server cannot pin it and a caller that omits it
  gets the model default rather than an error:
  ```python
  extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}
  ```
  Record the level next to any number produced at it.
- **Sampling**: DeepSeek publish `temperature=1.0`, `top_p=1.0` (`0.95` agentic). Note this differs
  from the 0.6 our Qwen arms use — freeze it deliberately rather than inheriting.
- **Tool calls**: a misconfigured tool parser still answers `/v1/models` and plain chat fine, then
  returns prose where a harness expects a call. Probe for `finish_reason: "tool_calls"` before
  trusting an agentic eval.

## Refs

- Upstream: `/home/jmendonca/servers/deepseek-v4-flash.sh`
- Official recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- Model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- Sibling stack: `../kimi3/` (Kimi-K3, port 9000, full launcher + helper scripts)
