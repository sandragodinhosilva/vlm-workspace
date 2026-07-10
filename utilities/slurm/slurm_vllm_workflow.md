# Slurm + vLLM serving workflow (this cluster)

Practical guide for allocating GPUs and serving a vLLM model + running an eval client against it, without hogging shared resources or queueing yourself for no reason.

Cluster: 8× NVIDIA B300 SXM6 per worker node, ~2.5 TB RAM, 192 CPUs per node. `DefMemPerCPU` ≈ 53 GB/CPU — this is the root cause of most "why am I queued?" surprises.

---

## TL;DR — golden rules

1. **Always pass `--mem` explicitly.** If you omit it, Slurm assigns `~53 GB × num_cpus`, which can claim the whole node's RAM even when you only asked for 2 GPUs.
2. **Size `--mem` to your workload, not to your CPU count.** For a vLLM server + eval at 2 GPUs, 256–512 GB is plenty.
3. **Don't allocate a second job to talk to your own server — attach instead** with `srun --overlap --jobid=<existing>`.
4. **TP size in the vLLM start script must match GPUs in your allocation**, or the preflight rejects you.
5. **`--model` passed to `evaluate.py` must match what vLLM advertises at `/v1/models`** (usually the full model path).

---

## 1. Allocating GPUs

### Per-GPU fair-share formula (safe default)

This is the formula recommended for the cluster — it gives you exactly your proportional share of a node and never over-claims:

```bash
NUM_GPUS=2
srun --nodes=1 \
     --gres=gpu:${NUM_GPUS} \
     -c $((24*NUM_GPUS)) \
     --mem=$((311*NUM_GPUS))G \
     --job-name=27b \
     --pty bash -i 

NUM_GPUS=2
srun --nodes=1 \
     --gres=gpu:${NUM_GPUS} \
     -c $((24*NUM_GPUS)) \
     --mem=$((311*NUM_GPUS))G \
     --job-name=eval1 \
     --nodelist=worker-5 \
     --pty bash -i 

ENABLE_THINKING=1 /home/sgsilva/utilities/serve/start_vllm_server.sh \
    /mnt/data/sgsilva/models/qwen35-27b-oracle-obs-cat-step339 \
    2 262144 7861
```

Where the multipliers come from (B300 node: 192 CPUs, ~2489 GB RAM, 8 GPUs):
- `192 / 8 = 24` CPUs per GPU
- `2489 / 8 ≈ 311` GB per GPU

Two users each running this formula at 2 GPUs fit cleanly on the same node (4 GPUs / 48 CPUs / 622 GB used, plenty left over).

### Right-sized for vLLM serving (more efficient)

The fair-share formula is safe but reserves more RAM than vLLM actually needs. For serving workloads you can pack more tightly:

```bash
NUM_GPUS=2
srun --nodes=1 \
     --exclude=worker-30,worker-31 \
     --gres=gpu:${NUM_GPUS} \
     -c $((8*NUM_GPUS)) \
     --mem=$((128*NUM_GPUS))G \
     --job-name=sft-vlm-small \
     --pty bash -i
```

Why these smaller numbers:
- `8 CPUs/GPU` is enough for vLLM + data loading; 24 CPUs/GPU is excessive for inference and inflates default memory.
- `128 GB/GPU` (= 256 GB for 2 GPUs) comfortably runs Qwen3.5-27B at 256k context.

**When to use which:**
- **Fair-share (`24` / `311`)** — training, unfamiliar workloads, when in doubt.
- **Right-sized (`8` / `128`)** — vLLM serving, eval clients, anything you've profiled.

### Full-node (8 GPUs) — the `sftvlm` alias

Defined in `~/.bashrc`:
```bash
alias sftvlm='srun --nodes=1 --exclude worker-30,worker-31 --gres=gpu:8 -c 192 --job-name sft-vlm --pty bash -i'
```
Exclusive use of one node is fine because you've taken all 8 GPUs anyway — memory hoarding doesn't block anyone new.

### Pin to a specific node

Add `--nodelist=worker-11`. **Hard requirement** — Slurm will queue you until that node has resources free; it won't fall back.

### Avoid drained nodes

`--exclude=worker-30,worker-31` (current known-bad set; check `sinfo` if jobs misbehave).

---

## 2. The memory trap (why you'd get queued unexpectedly)

If you do this:
```bash
srun --gres=gpu:2 -c 48 --pty bash -i   # NO --mem
```

Slurm silently assigns `~53 GB × 48 CPUs ≈ 2489 GB` — the entire node's RAM. Result: 6 GPUs sit idle on that node but no one else can land a GPU job there because there's no memory left to allocate.

### Verify before you accept the allocation

After your shell lands:
```bash
scontrol show job $SLURM_JOB_ID | grep -E "AllocTRES|ReqTRES"
scontrol show node $SLURMD_NODENAME | grep -E "RealMemory|AllocMem"
```
If `AllocMem` ≈ `RealMemory` and you only have 2 GPUs, you're the hog. `scancel` and resubmit with `--mem`.

### Check the cluster's default

```bash
scontrol show config | grep -i defmem
scontrol show partition main | grep -iE "DefMem|MaxMem"
```

---

## 3. Attaching to an existing allocation (don't waste a queue slot)

If you already have a running job on worker-11 and want a second shell on the same GPUs:

```bash
srun --overlap --jobid=<existing-jobid> --pty bash
```

- `--overlap` = share resources with the existing step (needed for additional shells).
- No new allocation, no queue, same GPUs visible.
- Use this for: running the eval client against a vLLM server you started in the first shell.

---

## 4. Inspecting Slurm state

| Question | Command |
|---|---|
| What's holding my job in queue? | `scontrol show job <jobid> \| grep Reason` |
| Who's on a specific node? | `squeue -w worker-11 -o "%.10i %.9P %.12j %.8u %.2t %.10M %b"` |
| GPU/CPU/memory free on a node? | `scontrol show node worker-11 \| grep -E "CPUAlloc\|AllocMem\|RealMemory\|Gres"` |
| Cluster-wide availability | `sinfo -o "%n %C %G %t"` |
| List your jobs | `squeue -u $USER` |

Common `Reason` codes:
- `Resources` — no eligible node has enough free.
- `Priority` — others ahead of you.
- `Nodes_required_for_job_are_DOWN_DRAINED` — your `--nodelist` target is unavailable.

---

## 5. Serving a vLLM model

Script: `/home/sgsilva/utilities/serve/start_vllm_server.sh`

Signature:
```
start_vllm_server.sh <model> <tensor_parallel_size> <max_model_len> <port>
```

Example — Qwen 3.5 27B with thinking mode on 2 GPUs:
```bash
ENABLE_THINKING=1 /home/sgsilva/utilities/serve/start_vllm_server.sh \
    qwen3.5-27b 2 262144 8000
```

Critical: **`<tensor_parallel_size>` must equal the number of GPUs in your allocation**. If TP > visible GPUs, the GPU preflight rejects the launch with:
```
ERROR: Requested tensor parallel size N, but only K/K GPUs look mostly free.
```

Override switches (use sparingly):
- `ALLOW_BUSY_GPUS=1` — bypass the "GPUs look busy" check.
- `SKIP_GPU_PREFLIGHT=1` — skip the check entirely.

Other env vars:
- `ENABLE_THINKING=1` — enables `--reasoning-parser qwen3` for Qwen 3.5 models.
- `QWEN35_VENV` — override the Qwen serving venv path (default `~/qwen3.5-serving-home-venv`).
- `STARTUP_HEARTBEAT_SECS=30` — print "still initializing" every N seconds.

### Confirm the server is up

```bash
curl -s http://$(hostname):8000/v1/models | python -m json.tool
```

The `id` field is the **served model name** — you need this exact string for the eval client.

---

## 6. Running the eval client

Script: `/home/sgsilva/vlm-post-training/data_preparation/evaluate.py`

Use the dedicated venv (the in-repo `.venv` is missing `sklearn`):
```bash
/home/sgsilva/vlm-post-training-home-venv/bin/python data_preparation/evaluate.py ...
```

Critical args:
- `--model <served-model-id>` — **must match `/v1/models` `id`**, typically the full model path like `/mnt/data/shared/models/Qwen3.5-27B`, not the shortname.
- `--server-url http://<node>:<port>/v1` — `<node>` is the node your server runs on (`hostname` inside that shell).
- `--provider vllm` — **don't skip this**. Default is `vertex_ai`, which tunes retry/concurrency for Gemini (5 workers instead of 10).
- `--resume` — safe to add; skips items already in the output file.

Full example (matching the 2-GPU server above):
```bash
cd /home/sgsilva/vlm-post-training
/home/sgsilva/vlm-post-training-home-venv/bin/python data_preparation/evaluate.py \
    --test-dataset-dir /mnt/data/shared/vlm/data/human_annotation_datasets/1105_not_reviewed/repetitions_test \
    --model /mnt/data/shared/models/Qwen3.5-27B \
    --server-url http://worker-11:8000/v1 \
    --provider vllm \
    --two-stage \
    --precomputed-visual-obs /mnt/data/sgsilva/results/visual_obs/runs/oracle_397b_1105_categorical.json \
    --output-file /mnt/data/sgsilva/results/visual_obs/runs/stage2_oracle_397b_categorical.json \
    --resume
```

---

## 7. End-to-end recipe: serve + eval on same node

Two shells, one allocation.

**Shell 1 — start allocation + server:**
```bash
# From login node
NUM_GPUS=2
srun --nodes=1 --exclude=worker-30,worker-31 \
     --gres=gpu:${NUM_GPUS} -c $((8*NUM_GPUS)) --mem=$((128*NUM_GPUS))G \
     --job-name=sft-vlm --pty bash -i

# Inside the allocation
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
ENABLE_THINKING=1 /home/sgsilva/utilities/serve/start_vllm_server.sh \
    qwen3.5-27b ${NUM_GPUS} 262144 8000
```

**Shell 2 — attach + run eval:**
```bash
# From login node, attach to the same job
srun --overlap --jobid=<JOB_ID_FROM_SHELL_1> --pty bash

# Wait for server to be ready
curl -s http://$SLURMD_NODENAME:8000/v1/models   # should show the model id

# Run eval against the local server
cd /home/sgsilva/vlm-post-training
/home/sgsilva/vlm-post-training-home-venv/bin/python data_preparation/evaluate.py \
    --model <id-from-/v1/models> \
    --server-url http://$SLURMD_NODENAME:8000/v1 \
    --provider vllm \
    ... # remaining args
```

---

## 8. Common mistakes & fixes

| Symptom | Cause | Fix |
|---|---|---|
| New job queues forever on worker-X | Your other job there reserved all RAM | Cancel it, resubmit with `--mem=<sane>` |
| Preflight: "Requested TP=8 but only 2 GPUs free" | TP > GPUs in allocation | Lower TP to match `--gres=gpu:N` |
| Eval errors `model not found` | `--model` doesn't match served-model-name | Use the `id` from `/v1/models` (full path) |
| Eval slow / weird retries | Forgot `--provider vllm` | Add it |
| Server can't bind port 8000 | Port already in use on the node | Use a different `<port>` arg |
| `evaluate.py` import errors | Wrong venv | Use `/home/sgsilva/vlm-post-training-home-venv/bin/python` |

---

## 8.5 Maximizing node utilization for reasoning eval (decode-bound)

The fair-share 2-GPU serving recipe above is for **packing politely** onto shared nodes. When the goal is the opposite — **finish a batch of evals fast by using the big nodes hard** — different rules apply. This matters most for **think-ON reasoning** evals, which are *decode-bound*: each rep generates thousands of `<think>` tokens, so wall-clock is dominated by token-by-token decode, not by how many reps you queue.

**Diagnosis: are you under-utilizing?** Check while a run is going:
```bash
# GPU util per device (run on the serving node)
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader
# server batch occupancy
curl -s http://<node>:<port>/metrics | grep -E "num_requests_running|num_requests_waiting"
```
Red flags:
- **Only 1 of your TP GPUs at ~100%, the rest at 0%** → batch too small to fill tensor-parallel ranks. The 2nd rank only earns its keep once the batch is full.
- **`num_requests_waiting: 0`** → the client is *starving* the server; raise client concurrency.

**Levers (in order of impact for decode-bound 27B reasoning):**
1. **Raise tensor parallelism: serve TP=4, not TP=2.** Splits each forward pass across 4 GPUs → faster per-token decode → long `<think>` sequences finish sooner. Biggest single win. 27B at TP=4 / ctx 65536 fits easily on B300.
2. **Raise client concurrency HARD — the client, not the GPU, is usually the bottleneck.** Stage-1 / eval clients do heavy **CPU preprocessing per request** (e.g. `generate_visual_observations_human.py` re-encodes frames to a temp mp4, loads frames, base64s, builds the prompt) BEFORE the request reaches vLLM. Measured 2026-05-28: `--max-workers 16` against a TP=4 27B kept only **~5 requests in flight** (`num_requests_running=5, num_requests_waiting=0`) — the GPU drained to 0% between bursts (bursty 0↔100 in nvtop) because the CPU couldn't refill the pipe. **`waiting=0` always = server starved = raise workers.** On a 192-core node, push `--max-workers` to **48–64** so CPU preprocessing overlaps deeply and a prepared request is always ready. Tune until `num_requests_waiting` sits at a small positive number (shallow queue = GPU never idle, but not so deep you risk client timeouts). GPU MEM% being high (~86%) is just resident weights — it is NOT compute; watch the compute % line and `num_requests_running` instead.
3. **One full node, two TP=4 servers, two variants in parallel.** With 5 variants, don't cycle one TP=2 server serially. Take a full idle node (`sftvlm`) and run two TP=4 servers side by side:
   ```bash
   sftvlm   # full 8-GPU node
   # server 1 on GPUs 0-3
   ENABLE_THINKING=1 QWEN35_VENV=/home/sgsilva/vlm-post-training-home-venv \
     CUDA_VISIBLE_DEVICES=0,1,2,3 /home/sgsilva/utilities/serve/start_vllm_server.sh \
     <ckpt_A> 4 65536 8011 &
   # server 2 on GPUs 4-7
   ENABLE_THINKING=1 QWEN35_VENV=/home/sgsilva/vlm-post-training-home-venv \
     CUDA_VISIBLE_DEVICES=4,5,6,7 /home/sgsilva/utilities/serve/start_vllm_server.sh \
     <ckpt_B> 4 65536 8012 &
   ```
   Then run two eval clients (one per port) at `--max-workers 16`.

**Pick an idle node, don't pin to a busy one.** Drop `--nodelist=worker-1` (which forces co-location and stacks you behind your own jobs). Grab a fresh node:
```bash
sinfo -N -h -o "%n %t" | grep idle    # nodes with all 8 GPUs free
```
Then either omit `--nodelist` (let Slurm place you) or pin to one of the idle ones.

**Watch out:**
- **Context headroom:** serve `max_model_len` strictly greater than the eval `--max-tokens`, or every request 400s with "maximum input length of 0 tokens". Standard reasoning combo = serve **65536**, eval `--max-tokens 32768`.
- **Distinct `--job-name` per server** (`vllm-A`, not `sft-vlm`) so `squeue` shows what's what and you don't stack duplicate idle allocations.
- **The eval client is often CPU-bound, not GPU-bound.** Before adding GPUs/TP, check `num_requests_running` vs `--max-workers`: if running ≪ max-workers and waiting=0, you're CPU-starved on preprocessing — raise `--max-workers` (48–64 on a 192-core node) FIRST. TP=4 with too few in-flight requests leaves the GPUs bursty/idle.

---

## 9. Resource sizing cheat sheet (B300 nodes)

| GPUs | CPUs (`-c`) | Memory (`--mem`) | Notes |
|---|---|---|---|
| 1 | 8 | 128G | Smoke tests, small models |
| 2 | 16 | 256G | Qwen3.5-27B server + eval client |
| 4 | 32 | 512G | Qwen3.5-122B-A10B at 256k |
| 8 | 192 | (omit, take node) | Full-node; large models, training |

Two rules of thumb:
- **Fair-share** (safe default, never over-claims): `-c $((24*N))`, `--mem=$((311*N))G`. Use for training or unknown workloads.
- **Right-sized for serving**: `--mem ≈ 128 GB × num_gpus` covers almost every vLLM workload. Bump higher only if you actually OOM.
