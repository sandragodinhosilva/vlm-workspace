#!/bin/bash
# Startup script for monitoring app with debugging

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "Starting VLM Monitoring App"
echo "================================================"
echo ""
echo "Checking prerequisites..."
echo "- Python version: $(python --version)"
echo "- Current directory: $(pwd)"
echo "- App file exists: $([ -f app.py ] && echo 'Yes' || echo 'No')"
echo ""

echo "Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "✓ Cache cleared"
echo ""

echo "Ensuring logs directory exists..."
mkdir -p logs
echo "✓ Logs directory ready"
echo ""

echo "Starting app with debug logging..."
echo "Logs will be written to: logs/app_YYYYMMDD_HHMMSS.log"
echo ""
python app.py --port 7861 --debug "$@"
