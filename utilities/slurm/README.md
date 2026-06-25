# `slurm/` — cluster-view tools

Everything for "what's the cluster doing right now / where can I land". Read-only —
none of these kill or scancel anything (per the never-stop-others'-processes rule).

| Tool | What it does |
| --- | --- |
| [`gpuq`](gpuq) | Pretty slurm queue + per-node GPU table. SSH-probes `nvidia-smi` and flags **rogue** GPUs (`◆` = busy per nvidia-smi but no slurm job claims it → node is NOT really free). Symlinked into `~/.local/bin`, so just type `gpuq`. Stdlib-only python3. |
| [`myjobs.sh`](myjobs.sh) | Your slurm jobs + the last log line of each running job (so you see *progress*, not just "RUNNING"). `-n N` for N lines. |
| [`gpuwho.sh`](gpuwho.sh) | What's eating a node's GPUs: pid → user → gpu-mem, tagged `(you)` / `(someone else — do NOT kill)`. Use when `gpuq` shows a rogue `◆` and you want to know whose it is. |
| [`status.sh`](status.sh) | One-shot cluster state: vLLM servers (model + max_len per running job node), slurm jobs by state, /home space. `--full` adds vLLM concurrency + KV-cache. |
| [`slurm_vllm_workflow.md`](slurm_vllm_workflow.md) | SLURM + vLLM serving workflow notes (srun details, serving recipes). |

## bashrc shortcuts (in `~/.bashrc`)

| Shortcut | Expands to |
| --- | --- |
| `gpuq` | the tool (via `~/.local/bin` symlink) |
| `gq` | `FORCE_COLOR=1 watch -c -n5 gpuq` — live auto-refresh; Ctrl+C to quit |
| `freenode` | `gpuq` filtered to the truly-clean idle nodes (slurm idle AND no rogue `◆`) |
| `rogue_nodes` | comma list of nodes busy off-scheduler right now (fallback `worker-30,worker-31`) |
| `sftvlm` / `nodestart [name]` | `srun` a full node, `--exclude "$(rogue_nodes)"` (live set, not hardcoded) |
| `myjobs` | `bash slurm/myjobs.sh` |
| `gpuwho <node>...` | `bash slurm/gpuwho.sh` |
| `status` | `bash slurm/status.sh` (`--full` for vLLM concurrency) |

Copy-paste recipes: `~/utilities/commands.txt` → **CLUSTER VIEW** section.

## Why "rogue" matters here

Slurm reports a node as `idle` whenever no slurm job is allocated to it — but on this
cluster people run vLLM servers directly (off-scheduler), so an "idle" node can be fully
busy. `gpuq` cross-checks `nvidia-smi` to catch that; `freenode` and the dynamic
`rogue_nodes` exclude keep you from `srun`-ing onto an occupied node and OOMing.
