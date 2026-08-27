#!/bin/bash
#SBATCH --job-name=deepseek-v4-flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192          # 24 CPUs per GPU (full per-GPU entitlement)
#SBATCH --mem=2488G                  # 311G per GPU
#SBATCH --output=/home/sgsilva/utilities/serve/deepseek4/run/slurm-%j.log


/home/sgsilva/utilities/serve/deepseek4/.venv/bin/vllm serve \
  /mnt/data/shared/cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1 \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 --port 9100 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --attention_config.use_fp4_indexer_cache=True \
  --moe-backend deep_gemm_mega_moe \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4
