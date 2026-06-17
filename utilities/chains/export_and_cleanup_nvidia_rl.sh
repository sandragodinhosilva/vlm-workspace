#!/bin/bash
# Script to export all checkpoints from common training output roots to
# /mnt/data/sgsilva/models and then optionally clean up exported directories.
#
# Usage:
#   bash export_and_cleanup_nvidia_rl.sh [--dry-run] [--delete]
#
# Options:
#   --dry-run    Show what would happen without making changes
#   --delete     Delete original result directories after successful exports

set -euo pipefail

DRY_RUN=false
DELETE_ORIGINALS=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        --delete)
            DELETE_ORIGINALS=true
            ;;
    esac
done

if [[ "$DRY_RUN" == true ]]; then
    echo "🔍 Running in DRY-RUN mode (no changes will be made)"
    echo ""
fi

if [[ "$DELETE_ORIGINALS" == true ]]; then
    echo "🗑️  DELETE MODE ENABLED: Original result directories will be deleted after export"
    echo ""
fi

# Directories
BASE_DIR="/mnt/data/sgsilva"
MODELS_DIR="$BASE_DIR/models"
PROJECT_DIR="/home/sgsilva/nemo-rl-vlm"
EXPORT_SCRIPT="$PROJECT_DIR/scripts/export_all_checkpoints.sh"

# Naming convention (single source of truth doc):
#   /mnt/data/sgsilva/utilities/MODEL_NAMING_CONVENTION.md
#
# Export name shape (compatible with export_all_checkpoints.sh + the eval
# pipeline's step_from_model_path / formatted-CSV builder):
#
#     qwen35-<size>-<task...>[-reasoning]-<date>-<runtype>-step<N>[_thinkon]
#                                                  ^^^^^^^         ^^^^^^^^
#   runtype  ∈ {sft, grpo, sft_grpo}  -- lives in the PREFIX, right before -step
#   _thinkon  -- post-step suffix, ONLY for reasoning runs (thinkoff is unmarked,
#                matching existing step357). export_all_checkpoints.sh appends
#                "-step<N>"; this wrapper renames "<...>-step<N>" -> "<...>-step<N>_thinkon"
#                afterwards (see maybe_append_thinkon).
#
# derive_model_prefix sets two globals for the caller:
#   REPLY_PREFIX     -- the prefix to pass to export_all_checkpoints.sh
#   REPLY_THINK_TOK  -- "thinkon" or "thinkoff" (drives the post-export rename)
derive_model_prefix() {
    local dir_name="$1"
    local name="$dir_name"

    # --- run-type: sft / grpo / sft_grpo (a model that went SFT then GRPO). ---
    # CRUCIAL distinction (must never be lost from the name).
    #  * grpo + an SFT init model   -> sft_grpo
    #  * grpo from raw base Qwen    -> grpo
    #  * everything else (run_vlm_sft.py) -> sft
    # Detection signals, in order:
    #  1. run dir name contains both grpo AND sft tokens (e.g. grpo_sft_...) -> sft_grpo
    #  2. grpo run whose config policy.model_name points at one of OUR sft
    #     exports (path contains "sft") -> sft_grpo; at raw base -> grpo
    #  3. grpo token only -> grpo ; else -> sft
    local run_type=""
    case "$name" in
        *grpo*sft*|*sft*grpo*) run_type="sft_grpo" ;;
        *grpo*)
            run_type="grpo"
            # try to upgrade grpo -> sft_grpo via the config init model
            local init_model
            init_model="$(grpo_init_model_for_run "$dir_name")"
            case "$init_model" in
                *sft*|*-sft-*|*_sft_*) run_type="sft_grpo" ;;
            esac
            ;;
        *) run_type="sft" ;;
    esac

    # --- thinking: reasoning runs train real <think> -> serve thinkON. ---
    # "reasoning" is also written "reas" or "thinkon"; all mean thinkON.
    # Absence => thinkoff (the unmarked default, matching step357).
    local think_tok="thinkoff"
    case "$name" in
        *reasoning*|*_reas_*|*_reas|*-reas-*|*-reas|*thinkon*|*think_on*) think_tok="thinkon" ;;
    esac

    name="${name#sft_vlm_megatron_}"
    name="${name#sft_vlm_}"
    name="${name#vlm_grpo_}"
    name="${name#grpo_sft_}"
    name="${name#sft_grpo_}"
    name="${name#grpo_}"
    name="${name#sft_}"

    # NOTE: keep the model family token as "qwen35" (no dot). Our export
    # convention (the prefix passed to export_all_checkpoints.sh) is
    # "qwen35-27b-..." NOT "qwen3.5-27b-...". An earlier version of this
    # function rewrote qwen35 -> qwen3.5, so the derived prefix never matched
    # the on-disk export names, the "already exported" check always failed,
    # and a --delete run would RE-EXPORT all checkpoints under dotted names
    # then delete the Megatron originals. Do not reintroduce the dot.
    name="$(echo "$name" | sed -E \
        -e 's/^qwen3_vl_/qwen3-vl-/' \
        -e 's/^qwen35_vl_/qwen35-vl-/' \
        -e 's/_/-/g')"

    # Consistency across experiments: drop config-scaffolding tokens that carry
    # no model meaning (local / megatron / Ngpu / vlm) and normalize the
    # reasoning token (reas -> reasoning) so the same experiment always yields
    # the same stem regardless of how its config was named. Tokenize on "-" so
    # adjacent scaffolding tokens (e.g. local-megatron) are both dropped.
    local -a _parts=() _kept=()
    IFS='-' read -ra _parts <<< "$name"
    local _p
    for _p in "${_parts[@]}"; do
        case "$_p" in
            local|megatron|vlm) continue ;;
            [0-9]gpu|[0-9][0-9]gpu) continue ;;
            reas) _kept+=("reasoning") ;;
            sft|grpo|sftgrpo|sft_grpo) continue ;;  # drop stray type tokens; we re-add canonically
            thinkon|thinkoff) continue ;;   # drop stray think tokens; handled post-step
            "") ;;
            *) _kept+=("$_p") ;;
        esac
    done
    name="$(IFS='-'; echo "${_kept[*]}")"

    # Append the run-type token at the tail of the PREFIX (right before the
    # "-step<N>" that export_all_checkpoints.sh will add).
    name="${name}-${run_type}"

    REPLY_PREFIX="$name"
    REPLY_THINK_TOK="$think_tok"
    echo "$name"
}

# Best-effort: find the GRPO config's policy.model_name (init/base model) for a
# run dir, to distinguish grpo (from base) vs sftgrpo (from an SFT ckpt).
# Returns "" if no config is found (caller then keeps plain "grpo").
grpo_init_model_for_run() {
    local run="$1"
    local cfg
    # Only an EXACT config-name match (run dir == config stem). A fuzzy find
    # would wrongly bind a generic recipe (e.g. vlm_grpo_qwen35_4b_thrive) to an
    # unrelated config and mis-tag grpo as sftgrpo. If the run name carries the
    # signal itself (grpo_sft_...), that's already handled by the case above.
    for cfg in \
        "$PROJECT_DIR/examples/configs/${run}.yaml" \
        "$PROJECT_DIR/examples/configs/recipes/vlm/${run}.yaml"; do
        [ -f "$cfg" ] || continue
        grep -E "^\s*model_name:" "$cfg" | head -1 | sed -E 's/.*model_name:\s*//; s/\s*#.*//'
        return 0
    done
    echo ""
}

# Rename "<...>-step<N>" -> "<...>-step<N>_thinkon" after export, for reasoning
# runs only. thinkoff stays unmarked (matches existing step357 convention).
maybe_append_thinkon() {
    local models_dir="$1" prefix="$2" step="$3" think_tok="$4"
    [ "$think_tok" = "thinkon" ] || return 0
    local src="$models_dir/${prefix}-step${step}"
    local dst="$models_dir/${prefix}-step${step}_thinkon"
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}  [DRY-RUN] Would rename -> ${prefix}-step${step}_thinkon${NC}"
        else
            mv "$src" "$dst" && echo -e "${GREEN}  ✓ tagged _thinkon: ${prefix}-step${step}_thinkon${NC}"
        fi
    fi
}

resolve_results_dir() {
    local -a candidates=(
        "${RESULTS_DIR:-}"
        "$BASE_DIR/results"
        "$BASE_DIR/checkpoints"
        "$BASE_DIR/nvidia-rl/results"
        "$BASE_DIR/nemo-rl-vlm.backup/results"
    )
    local candidate

    for candidate in "${candidates[@]}"; do
        [ -n "$candidate" ] || continue
        [ -d "$candidate" ] || continue

        if find "$candidate" -mindepth 1 -maxdepth 2 -type d -name "step_*" -print -quit | grep -q .; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

RESULTS_DIR="$(resolve_results_dir || true)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Validation
if [ -z "$RESULTS_DIR" ] || [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}✗ ERROR: Could not find a checkpoint/results directory under $BASE_DIR${NC}"
    echo "Checked:"
    echo "  - $BASE_DIR/results"
    echo "  - $BASE_DIR/checkpoints"
    echo "  - $BASE_DIR/nvidia-rl/results"
    echo "  - $BASE_DIR/nemo-rl-vlm.backup/results"
    exit 1
fi

if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${RED}✗ ERROR: Models directory not found: $MODELS_DIR${NC}"
    exit 1
fi

if [ ! -f "$EXPORT_SCRIPT" ]; then
    echo -e "${RED}✗ ERROR: Export script not found: $EXPORT_SCRIPT${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  NVIDIA-RL Checkpoint Export & Cleanup Tool              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Using source directory: $RESULTS_DIR"
echo "Using export script:    $EXPORT_SCRIPT"
echo ""

# Step 1: Discover all result directories with checkpoints
echo -e "${BLUE}[Step 1/3] Discovering checkpoint directories...${NC}"
RESULT_DIRS=$(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [ -z "$RESULT_DIRS" ]; then
    echo -e "${YELLOW}⚠ No result directories found in $RESULTS_DIR${NC}"
    exit 0
fi

# Count directories
DIR_COUNT=$(echo "$RESULT_DIRS" | wc -l)
echo -e "${GREEN}✓ Found $DIR_COUNT training result directories${NC}"
echo ""

# Step 2: Export each directory
echo -e "${BLUE}[Step 2/3] Processing exports...${NC}"
FAILED_DIRS=()
EXPORTED_DIRS=()
SKIPPED_DIRS=()
ALREADY_EXPORTED_DIRS=()

for RESULT_DIR in $RESULT_DIRS; do
    DIR_NAME=$(basename "$RESULT_DIR")
    
    # Check if directory has checkpoints
    CHECKPOINT_COUNT=$(find "$RESULT_DIR" -maxdepth 1 -type d -name "step_*" | wc -l)
    
    if [ "$CHECKPOINT_COUNT" -eq 0 ]; then
        echo -e "${YELLOW}⊘ Skipping $DIR_NAME (no checkpoints found)${NC}"
        SKIPPED_DIRS+=("$DIR_NAME")
        continue
    fi
    
    echo ""
    echo -e "${BLUE}Processing: $DIR_NAME${NC}"
    echo "  Checkpoints found: $CHECKPOINT_COUNT"
    
    # Derive the export prefix from the training run name. Call directly (not in
    # a $(...) subshell) so REPLY_PREFIX / REPLY_THINK_TOK globals propagate.
    derive_model_prefix "$DIR_NAME" >/dev/null
    MODEL_PREFIX="$REPLY_PREFIX"
    THINK_TOK="$REPLY_THINK_TOK"
    THINK_SUFFIX=""
    [ "$THINK_TOK" = "thinkon" ] && THINK_SUFFIX="_thinkon"
    echo "  Export prefix:     $MODEL_PREFIX  (runtype+think -> ...-step<N>${THINK_SUFFIX})"

    # Check if all checkpoints for this model already exist in destination.
    # Match the final on-disk name, including the post-step _thinkon suffix.
    ALL_EXIST=true
    for STEP_DIR in $(find "$RESULT_DIR" -maxdepth 1 -type d -name "step_*" | sort -V); do
        STEP_NUM=$(basename "$STEP_DIR" | sed 's/step_//')
        HF_OUTPUT_PATH="$MODELS_DIR/${MODEL_PREFIX}-step${STEP_NUM}${THINK_SUFFIX}"
        if [ ! -d "$HF_OUTPUT_PATH" ]; then
            ALL_EXIST=false
            break
        fi
    done
    
    if [ "$ALL_EXIST" = true ]; then
        echo -e "${YELLOW}  ⊘ All checkpoints already exported${NC}"
        
        # Delete original if --delete flag is set
        if [ "$DELETE_ORIGINALS" = true ]; then
            if [ "$DRY_RUN" = true ]; then
                DIR_SIZE=$(du -sh "$RESULT_DIR" | cut -f1)
                echo -e "${YELLOW}  [DRY-RUN] Would delete: $RESULT_DIR (${DIR_SIZE})${NC}"
            else
                DIR_SIZE=$(du -sh "$RESULT_DIR" | cut -f1)
                echo -e "${YELLOW}  → Deleting original result directory (${DIR_SIZE})...${NC}"
                rm -rf "$RESULT_DIR"
                echo -e "${GREEN}  ✓ Deleted${NC}"
            fi
        fi
        
        ALREADY_EXPORTED_DIRS+=("$DIR_NAME")
        continue
    fi
    
    # Run export
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}  [DRY-RUN] Would export to: $MODELS_DIR${NC}"
        EXPORTED_DIRS+=("$DIR_NAME")
    else
        echo "  → Running export..."
        if cd "$PROJECT_DIR" && bash "$EXPORT_SCRIPT" "$RESULT_DIR" "$MODELS_DIR" "$MODEL_PREFIX" 2>&1 | tail -20; then
            echo -e "${GREEN}  ✓ Export successful${NC}"
            EXPORTED_DIRS+=("$DIR_NAME")

            # For reasoning runs, tag each exported step dir with the post-step
            # _thinkon suffix (thinkoff stays unmarked).
            if [ "$THINK_TOK" = "thinkon" ]; then
                for STEP_DIR in $(find "$RESULT_DIR" -maxdepth 1 -type d -name "step_*" | sort -V); do
                    STEP_NUM=$(basename "$STEP_DIR" | sed 's/step_//')
                    maybe_append_thinkon "$MODELS_DIR" "$MODEL_PREFIX" "$STEP_NUM" "$THINK_TOK"
                done
            fi

            # Delete original if --delete flag is set and export was successful
            if [ "$DELETE_ORIGINALS" = true ]; then
                if [ "$DRY_RUN" = true ]; then
                    DIR_SIZE=$(du -sh "$RESULT_DIR" | cut -f1)
                    echo -e "${YELLOW}  [DRY-RUN] Would delete: $RESULT_DIR (${DIR_SIZE})${NC}"
                else
                    DIR_SIZE=$(du -sh "$RESULT_DIR" | cut -f1)
                    echo -e "${YELLOW}  → Deleting original result directory (${DIR_SIZE})...${NC}"
                    rm -rf "$RESULT_DIR"
                    echo -e "${GREEN}  ✓ Deleted${NC}"
                fi
            fi
        else
            echo -e "${RED}  ✗ Export failed${NC}"
            FAILED_DIRS+=("$DIR_NAME")
        fi
    fi
done

echo ""
echo -e "${BLUE}[Summary after export]${NC}"
echo -e "  ${GREEN}✓ Newly exported: ${#EXPORTED_DIRS[@]}${NC}"
echo -e "  ${GREEN}✓ Already exported: ${#ALREADY_EXPORTED_DIRS[@]}${NC}"
echo -e "  ${RED}✗ Failed: ${#FAILED_DIRS[@]}${NC}"
echo -e "  ${YELLOW}⊘ Skipped: ${#SKIPPED_DIRS[@]}${NC}"

if [ ${#FAILED_DIRS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed directories:${NC}"
    for dir in "${FAILED_DIRS[@]}"; do
        echo -e "  ${RED}✗ $dir${NC}"
    done
    echo ""
    echo -e "${RED}WARNING: Some exports failed. NOT cleaning up results directory.${NC}"
    exit 1
fi

# Step 3: Cleanup
echo ""
echo -e "${BLUE}[Step 3/3] Cleanup results directory...${NC}"

if [ "$DELETE_ORIGINALS" = true ]; then
    echo -e "${GREEN}✓ Per-directory cleanup handled during export (--delete enabled)${NC}"
elif [ "$(basename "$RESULTS_DIR")" != "results" ]; then
    echo -e "${YELLOW}⊘ Skipping top-level cleanup for shared directory: $RESULTS_DIR${NC}"
    echo "  Use --delete to remove exported run directories individually."
elif [ "$DRY_RUN" = true ]; then
    RESULTS_SIZE=$(du -sh "$RESULTS_DIR" | cut -f1)
    echo -e "${YELLOW}[DRY-RUN] Would delete: $RESULTS_DIR${NC}"
    echo "  Size that would be freed: $RESULTS_SIZE"
else
    RESULTS_SIZE=$(du -sh "$RESULTS_DIR" | cut -f1)
    
    # Double-check before deletion
    echo -e "${YELLOW}⚠ About to delete: $RESULTS_DIR${NC}"
    echo "  Size to be freed: $RESULTS_SIZE"
    echo ""
    
    read -p "Are you sure you want to delete the results directory? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  → Deleting results directory..."
        rm -rf "$RESULTS_DIR"
        echo -e "${GREEN}  ✓ Results directory deleted${NC}"
        echo -e "${GREEN}  ✓ Space freed: $RESULTS_SIZE${NC}"
    else
        echo -e "${YELLOW}  ⊘ Cleanup cancelled${NC}"
    fi
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  ✓ Process Complete                                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
