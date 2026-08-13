#!/usr/bin/env bash
#
# Check the health of the running Kimi-K3 endpoint.
# Resolves the node the slurm job is on, hits vLLM's /health, and confirms the model is listed.
# Exit code 0 = healthy, non-zero = not reachable / unhealthy.
#
# Usage: ./health.sh

set -euo pipefail

KIMI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$KIMI_DIR/lib.sh"

BASE_URL="$(resolve_base_url)" || exit 1
HEALTH_URL="${BASE_URL%/v1}/health"
MODELS_URL="$BASE_URL/models"

echo "endpoint: $BASE_URL"

# Liveness: vLLM's /health returns 200 with an empty body when the engine is ready.
if ! curl -sf -o /dev/null "$HEALTH_URL"; then
  echo "UNHEALTHY: $HEALTH_URL did not return 200" >&2
  exit 1
fi
echo "health:   OK (200 from /health)"

# Readiness: the served model id should be listed at /v1/models.
MODELS_JSON="$(curl -sf "$MODELS_URL" 2>/dev/null || true)"
if [[ -z "$MODELS_JSON" ]]; then
  echo "WARNING: /health OK but /v1/models did not respond" >&2
  exit 1
fi

if echo "$MODELS_JSON" | grep -q '"kimi-k3"'; then
  echo "model:    kimi-k3 is being served"
  echo "HEALTHY"
  exit 0
else
  echo "WARNING: 'kimi-k3' not found in /v1/models response:" >&2
  echo "$MODELS_JSON" >&2
  exit 1
fi
