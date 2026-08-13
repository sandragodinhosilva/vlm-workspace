#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT=/mnt/data/shared/models
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/download.log"   # temporary progress log in servers/kimi3
models=(moonshotai/Kimi-K3)

# Send all output (incl. hf's progress bars on stderr) to console AND the log file.
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== download started: $(date -Is) ==="

# iterate over each model and download it (org is before / and model is after /)
for model_full in "${models[@]}"; do
    org=$(echo "${model_full}" | cut -d'/' -f1)
    model=$(echo "${model_full}" | cut -d'/' -f2)   # -> Kimi-K3
    echo "Downloading model: ${model} from org: ${org}"
    # huggingface_hub >=1.0 downloads via the Xet backend; HF_XET_HIGH_PERFORMANCE=1 is the
    # modern replacement for the now-deprecated HF_HUB_ENABLE_HF_TRANSFER and speeds up large pulls.
    HF_XET_HIGH_PERFORMANCE=1 hf download "${org}/${model}" \
        --local-dir "${DEST_ROOT}/${model}" \
        --token="$HF_TOKEN" \
        --repo-type model
    echo "Downloaded model: ${model} -> ${DEST_ROOT}/${model}"
done
echo "=== download finished: $(date -Is) ==="
