# B300s GPU Cluster

# Nebius Specifications

## General Information

### nebius Overview

- **Infrastructure**: [Nebius Soperator](https://nebius.com/services/soperator) (Slurm + Kubernetes)
- **General Docs from Nebius:** https://docs.nebius.com/slurm-soperator
- **Total Nodes**: 32 worker nodes (worker-0 to worker-31)
- **Total GPUs**: 256 NVIDIA B300 GPUs (8 GPUs per node)
- **Total CPUs**: 6,144 logical CPUs (192 per node)
- **Total Memory**: ~~80 TB RAM (~~2.5 TB per node)
- **Main Partition**: All nodes in single "main" partition

### Node Specifications

- **CPU**: 192 logical CPUs per node (2 sockets × 48 cores × 2 threads)
- **Memory**: ~2.5 TB RAM per node
- **GPU**: 8x NVIDIA B300 SXM6 AC GPUs per node (~269 GB VRAM each)
- **GPU Configuration**: 2 sockets (S:0-1) with 4 GPUs each
- **Driver Version**: 580.126.09
- **CUDA Version**: 13.0

### Node Status Overview

- **Development Nodes**: worker-30, worker-31 - Reserved for development/testing
- **Production Nodes**: worker-0 to worker-29 - Allocated for production workloads

### Storage Overview

| Mount Point | Size | Purpose |
| --- | --- | --- |
| `/home` | 3.6 TB | User home directories (NFS shared) |
| `/mnt/data` | 190 TB | Shared data storage (models, datasets). **Slow I/O.** |
| `/` (Root) | 9.0 TB | System and local storage |
| `/mnt/memory` | 112 GB | Memory-based tmpfs |

**Storage Guidelines:**

- **Projects and code**: Use `/home/<username>` (shared across all nodes, much faster than `/mnt/data`)
- **Large models and datasets**: Use `/mnt/data/shared` (shared across cluster, but slow). Good for storing and reading model weights and datasets; avoid using it for active training I/O.
- **Temporary processing**: Use `/mnt/memory` for fast in-memory I/O

> ⚠️ `/mnt/data` has high latency compared to `/home`. Keep your working projects, code, and environments in your home directory. Only use `/mnt/data` for large shared assets like model checkpoints and datasets.

## Access & Authentication

### General Login

Connect to the nebius gateway:

```
ssh -i <path_to_private_key> <username>@89.124.37.171
```

### Login Node Access

For persistent sessions (tmux/byobu), connect to specific login nodes:

```
ssh -J <username>@ <username>@login-<number> -i <path_to_private_key>
```

> ⚠️ Login Node Guidelines:
> 
> - Login nodes are shared resources - avoid heavy computation
> - Even CPU-intensive tasks can cause OOM (Out of Memory) issues
> - Always use worker nodes via Slurm for actual work
> - Use login nodes only for file management, job submission, and light tasks

### SSH Configuration Example

Add this to your `~/.ssh/config` file for easier access:

```
# nebius gateway
Host nebius
    HostName 89.124.37.171
    User <your_username>
    IdentityFile ~/.ssh/<your_private_key>
    ForwardAgent yes

# Direct login node access
Host login-0
    HostName login-0
    User <your_username>
    IdentityFile ~/.ssh/<your_private_key>
    ProxyJump nebius

Host login-1
    HostName login-1
    User <your_username>
    IdentityFile ~/.ssh/<your_private_key>
    ProxyJump nebius

Host nebius_worker_30
    HostName worker-30
    User <your_username>
    IdentityFile ~/.ssh/<your_private_key>
    ProxyJump nebius
    LocalForward 8888 localhost:8888

Host nebius_worker_31
    HostName worker-31
    User <your_username>
    IdentityFile ~/.ssh/<your_private_key>
    ProxyJump nebius
    LocalForward 8888 localhost:8888
```

Then connect with:

```
ssh nebius           # Gateway
ssh login-0          # Specific login node
ssh login-1          # Alternative login node
```

### Environment Setup

Add to your `~/.bashrc` for shared cache and environment variables:

```
# Shared Hugging Face cache to avoid redundant downloads
export HF_HOME="/mnt/data/shared/cache"
export HF_AUTH_TOKEN=$HF_TOKEN # for authentication

# Shared directory for models (all models should live here)
export MODEL_DIR="/mnt/data/shared/models"
```

Apply changes:

```
source ~/.bashrc
```

## Development & Testing

### Development Nodes

Workers 30-31 are reserved for development, testing, and small experiments.

### Resource Allocation

### Development Access with Time Limits

```
srun --job-name=my-gpu-dev-job --nodes=1 --nodelist=worker-30,worker-31 --gres=gpu:{NUM_GPUS} --time=2:00:00 --pty bash -i
```

### Best Practices

- **Resource Sharing**: Release resources promptly when finished
- **Time Limits**: Use `--time` parameter for better resource management
- **Name your jobs:** Use `--job-name` parameter to `srun` commands for identification and tracking
- **Clean Up**: Always exit interactive sessions when done

**NOTE:**

If {NUM_GPUS} < 8 you must specify the number of CPUs and memory you need. Otherwise you will allocate all CPUs and memory and no one else can use that node (even if you are not using all the GPUs):

```
srun --nodes=1 --gres=gpu:1 -c 24 --mem 311G --pty bash -i
# If you need more GPUs (e.g. 2) just use multipliers.
srun --nodes=1 --gres=gpu:{NUM_GPUS} -c {24*NUM_GPUS} --mem {311*NUM_GPUS}G --pty bash -i
```

## Production Compute Jobs

### Interactive Production Jobs

For compute jobs on production nodes (worker-0 to worker-29):

```
# Single GPU production job
srun --job-name=my-single-gpu-prod-job --nodes=1 --exclude=worker-30,worker-31 --gres=gpu:1 --time=4:00:00 --pty bash -i

# Multi-node GPU job
srun --job-name=my-multinode-gpu-prod-job --nodes=2 --exclude=worker-30,worker-31 --gres=gpu:8 --time=8:00:00 --pty bash -i

# CPU-only job
srun --job-name=my-cpu-prod-job --nodes=1 --exclude=worker-30,worker-31 --cpus-per-task=32 --time=2:00:00 --pty bash -i
```

### Multi-Node Interactive Job Behavior

**What happens with multi-node interactive jobs:**

1.  Slurm allocates 2 nodes (e.g., worker-5 and worker-12) with 8 GPUs each (16 GPUs total)
2.  You get an interactive bash session on the **first allocated node** only
3.  The second node is reserved and accessible, but you're not directly connected to it
4.  From the first node, you can SSH to other allocated nodes or use `srun` to run commands across all nodes

**Accessing other nodes in your allocation:**

```
# Check which nodes were allocated
echo $SLURM_JOB_NODELIST

# SSH to other allocated nodes
ssh worker-12

# Run commands on all allocated nodes
srun --nodes=2 hostname
srun --nodes=2 nvidia-smi
```

### Resource Allocation with salloc

An alternative to `srun --pty` is using `salloc` for resource allocation:

```
# Allocate resources and get a new shell
salloc --nodes=2 --exclude=worker-30,worker-31 --gres=gpu:8 --time=8:00:00

# Once allocated, you're in a new shell with access to the resources
# Run commands on the allocated nodes
srun hostname
srun nvidia-smi

# Exit the allocation
exit
```

### srun vs salloc

**srun:**

- Runs commands directly on allocated nodes
- `srun --pty bash -i` gives you an interactive session on the first node
- Command execution and resource allocation in one step

**salloc:**

- Only allocates resources, gives you a new shell on the login node
- Use `srun` within the allocation to run commands on worker nodes
- Can SSH directly to allocated nodes for interactive work
- More flexible for running multiple different commands
- Better for complex workflows with multiple job steps

**Example workflow with salloc:**

```
# Allocate resources
salloc --nodes=1 --gres=gpu:4 --time=4:00:00

# Option 1: Run commands via srun
srun python preprocess.py
srun python train.py
srun python evaluate.py

# Option 2: SSH directly to the allocated node
ssh $SLURM_JOB_NODELIST
# Now you're on the worker node with interactive access
nvidia-smi
python train.py
exit  # Exit SSH session

# Exit allocation
exit
```

### Batch Job Submission

Create a batch script (`job.sbatch`):

```
#!/bin/bash
#SBATCH --job-name=my-2-node-training-job # Job name shown in queue
#SBATCH --nodes=2                         # Number of nodes to allocate
#SBATCH --exclude=worker-30,worker-31     # Exclude development nodes
#SBATCH --gres=gpu:8                      # Request 8 GPUs
#SBATCH --time=12:00:00                   # Max runtime (HH:MM:SS)
#SBATCH --output=job_%j.out               # Standard output file (%j = job ID)
#SBATCH --error=job_%j.err                # Standard error file

# Load environment
source ~/miniconda3/bin/activate
conda activate myenv

# Run your job
python train.py --config config.yaml
```

Submit with:

```
sbatch job.sbatch
```

To check the job as if you were in an interactive session, you can run:

```
tail -f job_%j.out
# OR
tail -f job_%j.err
```

### Job Management Commands

```
# Check job status
squeue -u $USER

# Cancel a job
scancel <job_id>

# View job details
scontrol show job <job_id>

# Check node availability
sinfo --exclude=worker-30,worker-31
```

### HF Transfer — super fast downloads 🚀

```
pip install huggingface_hub[hf_transfer]
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-4B --local-dir $MODEL_DIR/Qwen3-4B --repo-type model
```