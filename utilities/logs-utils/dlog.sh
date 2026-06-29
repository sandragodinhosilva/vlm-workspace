#!/usr/bin/env bash
# dlog — register a created dataset in ~/.claude/DATASETS.md
#
# Every dataset we materialize gets one entry: full path, how it was built
# (builder + args), source inputs, a one-line purpose, row count, and date.
# Provenance the filesystem can't infer is passed explicitly — only the builder
# knows it. Call this right after a successful build.
#
# Usage:
#   dlog --path <DATASET_DIR> \
#        --purpose "<one line: what/why>" \
#        --builder "<script + key args>" \
#        --sources "<input dataset(s)>" \
#        [--rows N]            # auto-counted via HF load if omitted & loadable
#        [--status canonical|superseded|component]   # default: canonical
#        [--superseded-by <name>]                     # the dataset that replaces this one
#
# Example:
#   dlog --path /mnt/.../1805_stage2_train_noreason_obsenforced \
#        --purpose "EXP-B Phase-1 stage-2 train (no reasoning), obs-enforced template" \
#        --builder "data_preparation/reasoning/build_stage2_train_noreason.py --template severity_v2_two_stage_severity_obs_enforced.txt" \
#        --sources "repetitions_train + 1805_oracle_obs_sft_train_categorical (joined on exercise/session/rep)" \
#        --rows 5271

set -euo pipefail

REG="$HOME/.claude/DATASETS.md"
PATH_ARG="" PURPOSE="" BUILDER="" SOURCES="" ROWS="" STATUS="canonical" SUPERSEDED_BY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --path)          PATH_ARG="$2";      shift 2;;
    --purpose)       PURPOSE="$2";       shift 2;;
    --builder)       BUILDER="$2";       shift 2;;
    --sources)       SOURCES="$2";       shift 2;;
    --rows)          ROWS="$2";          shift 2;;
    --status)        STATUS="$2";        shift 2;;  # canonical | superseded | component (default: canonical)
    --superseded-by) SUPERSEDED_BY="$2"; shift 2;;  # name of the replacing dataset
    *) echo "dlog: unknown arg '$1'" >&2; exit 2;;
  esac
done

case "$STATUS" in
  canonical|superseded|component) ;;
  *) echo "dlog: --status must be canonical|superseded|component (got '$STATUS')" >&2; exit 2;;
esac

[ -z "$PATH_ARG" ] && { echo "dlog: --path is required" >&2; exit 2; }
[ -z "$PURPOSE" ]  && { echo "dlog: --purpose is required" >&2; exit 2; }

# Auto-count rows if not given and the dataset is an HF Arrow dir.
# Delegates to count_rows.py, which SUMS num_rows across splits for a DatasetDict
# (len()/DatasetDict.num_rows would mis-report the split COUNT) and falls back to
# per-split subdirs. Distinct "?" sentinel on failure (never a silent 0/happy path).
if [ -z "$ROWS" ] && [ -d "$PATH_ARG" ]; then
  ROWS="$(/home/sgsilva/vlm-post-training-home-venv/bin/python \
    "$(dirname "$0")/count_rows.py" "$PATH_ARG" 2>/dev/null || echo "?")"
fi

DATE="$(date +%Y-%m-%d)"

# Initialize the registry with a header if missing.
if [ ! -f "$REG" ]; then
  cat > "$REG" <<'HDR'
# Datasets registry

Every dataset materialized on the cluster, newest first. Logged via
`~/utilities/logs-utils/dlog.sh`. Each entry: path · purpose · builder · sources · rows · date.

---
HDR
fi

# Prepend the new entry directly under the `---` separator (newest first).
ENTRY="$(cat <<EOF

### \`$(basename "$PATH_ARG")\`  ($DATE)
- **Status:** $STATUS$([ -n "$SUPERSEDED_BY" ] && printf ' (superseded by `%s`)' "$SUPERSEDED_BY")
- **Path:** \`$PATH_ARG\`
- **Purpose:** $PURPOSE
- **Builder:** \`$BUILDER\`
- **Sources:** $SOURCES
- **Rows:** $ROWS
EOF
)"

# Insert after the first '---' line.
awk -v entry="$ENTRY" '
  !done && /^---$/ { print; print entry; done=1; next }
  { print }
' "$REG" > "$REG.tmp" && mv "$REG.tmp" "$REG"

echo "dlog: registered $(basename "$PATH_ARG") ($ROWS rows) → $REG"
