# Serving Kimi-K3 (vLLM + Slurm)

run:
/home/sgsilva/utilities/serve/kimi3/serve.sh


Serves the local Kimi-K3 checkpoint (`/mnt/data/shared/models/Kimi-K3`) with vLLM on a single
Slurm node using all 8 B300 GPUs (tensor-parallel).

Once running, the model is reachable as:

| | |
|---|---|
| **model** | `hosted_vllm/kimi-k3` |
| **api base** | `http://<worker>:9000/v1` (worker node assigned by Slurm) |

The exact node changes per run; `serve.sh` records it in `run/endpoint.env`, and the helper
scripts resolve it automatically (no need to hardcode the worker).

## One-time setup

vLLM (nightly) and litellm are installed into the local `.venv`. To recreate it:

```bash
uv venv
uv pip install --python .venv/bin/python -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python litellm
```

## Start serving

```bash
./serve.sh
```

Submits the Slurm job (name `kimi-k3`, 1 node, `gpu:8`, `--exclusive`), launches vLLM, and waits
until it is healthy before publishing `run/endpoint.env`. It prints the job id and log path.

- Follow startup:  `tail -f run/slurm-<jobid>.log`  (vLLM's own output is in `run/vllm.log`)
- Startup of the 1M-context model can take **many minutes** (weight load + compile).
- `./serve.sh --print-only` renders the generated sbatch script without submitting.
- Nodes `worker-30` / `worker-31` are always excluded (team policy).

## Check the endpoint health

```bash
./health.sh
```

Resolves the worker node, checks vLLM's `/health`, and confirms `kimi-k3` is listed at
`/v1/models`. Exits 0 when healthy, non-zero otherwise.

## Send an example request ("hi")

```bash
.venv/bin/python check_response.py
```

Sends a single `"hi"` through **litellm** to `hosted_vllm/kimi-k3` and prints the reply.

## Monitor the workload

```bash
./monitor.sh            # one-shot snapshot
./monitor.sh --watch    # refresh every 2s (Ctrl-C to stop); ./monitor.sh --watch 5 for 5s
```

Scrapes vLLM's `/metrics` and shows **running** requests, **waiting (queue)** requests, and
KV-cache usage.

## Stop serving

```bash
scancel <jobid>         # jobid printed by serve.sh; or: scancel --name kimi-k3
```

On cancel, the batch script's trap sends `SIGTERM` to vLLM, waits for it to exit, and removes
`run/endpoint.env`; Slurm's cgroup reaps any stragglers, so the GPUs are freed cleanly.

**You do not need to clean up `run/` after a `scancel`.** `endpoint.env` is removed automatically,
and the helper scripts treat Slurm as the source of truth: `resolve_base_url` only trusts
`endpoint.env` when its `JOBID` is still running, so a stale file left behind by a hard crash is
ignored rather than pointing you at a dead node. The only files that accumulate are the per-job
`run/slurm-<jobid>.log` logs — delete those whenever you like (`rm run/slurm-*.log`), purely for
tidiness.

## Files

| File | Purpose |
|---|---|
| `serve.sh` | Submit the Slurm job that runs `vllm serve` (config vars at the top). |
| `health.sh` | Check endpoint health. |
| `check_response.py` | Minimal litellm "hi" round-trip. |
| `monitor.sh` | Show running/queued request counts from `/metrics`. |
| `lib.sh` | Shared endpoint resolver (sourced by the shell scripts). |
| `run/` | Runtime artifacts: `endpoint.env`, `slurm-<jobid>.log`, `vllm.log`, generated sbatch. |
