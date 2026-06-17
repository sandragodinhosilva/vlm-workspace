#!/bin/bash
# lint_model_name.sh — assert a checkpoint-dir basename derives to a CONVENTION-COMPLIANT
# export prefix, BEFORE you launch SFT/GRPO or run the exporter.
#
# Single source of truth = MODEL_NAMING_CONVENTION.md, ENFORCED by derive_model_prefix in
# export_and_cleanup_nvidia_rl.sh. This linter REUSES that exact function (sourced, not
# reimplemented) so it can never drift from the exporter, then validates the derived prefix
# against the canonical shape:  qwen35-<size>-<task...>[-reasoning]-<runtype>
#   <size>    ∈ 4b | 27b           <runtype> ∈ sft | grpo | sft_grpo
#   family    = qwen35 (NO dot)     reasoning present iff thinkon
#
# Usage:
#   lint_model_name.sh sft_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union
#   lint_model_name.sh /mnt/data/sgsilva/checkpoints/<ckpt_dir>          # basename taken
#   lint_model_name.sh --config examples/configs/<cfg>.yaml             # reads checkpoint_dir from the yaml
# Exit 0 = compliant (prints derived prefix + think tok); exit 1 = violation (prints why).

set -uo pipefail
EXPORTER="/home/sgsilva/utilities/chains/export_and_cleanup_nvidia_rl.sh"

if [[ "${1:-}" == "--config" ]]; then
    cfg="${2:?--config needs a path}"
    # pull checkpoint_dir from the yaml; basename is what the exporter derives from
    ckpt="$(grep -E '^\s*checkpoint_dir:' "$cfg" | head -1 | sed -E 's/.*checkpoint_dir:\s*//; s/\s*#.*//; s/^["'\'']//; s/["'\'']\s*$//')"
    [[ -z "$ckpt" ]] && { echo "LINT FAIL: no checkpoint_dir in $cfg" >&2; exit 1; }
    DIR_NAME="$(basename "$ckpt")"
else
    DIR_NAME="$(basename "${1:?usage: lint_model_name.sh <ckpt_dir|--config cfg.yaml>}")"
fi

# Source ONLY derive_model_prefix from the exporter (stub its one dependency).
grpo_init_model_for_run() { echo ""; }   # linter has no live config init; sft/grpo-token detection still works
# shellcheck disable=SC1090
eval "$(sed -n '/^derive_model_prefix() {/,/^}/p' "$EXPORTER")"

derive_model_prefix "$DIR_NAME" >/dev/null
PREFIX="$REPLY_PREFIX"
THINK="$REPLY_THINK_TOK"

fail() { echo "❌ LINT FAIL ($DIR_NAME -> $PREFIX): $1" >&2; exit 1; }

# 1. canonical shape: qwen35-<size>-<task...>-<runtype>  (lowercase, hyphen-separated, no dot/underscore)
[[ "$PREFIX" =~ ^qwen35-(4b|27b)-[a-z0-9-]+-(sft|grpo|sft_grpo)$ ]] \
    || fail "prefix does not match qwen35-(4b|27b)-<task>-(sft|grpo|sft_grpo)"
# 2. no dotted family, no stray underscores (would break the exporter's already-exported check)
[[ "$PREFIX" == *"qwen3.5"* ]] && fail "dotted family 'qwen3.5' — must be 'qwen35'"
[[ "$PREFIX" == *_* && "$PREFIX" != *sft_grpo* ]] && fail "stray underscore (only sft_grpo may contain one)"
# 2b. an already-exported dir (basename contains -step<N>) must NOT be linted: derive_model_prefix
#     expects the TRAINING-run dir (never has -stepN) and mangles an exported name into
#     ...-step<N>-<runtype>. Fail loudly so a mis-fed exported dir can't pass with a malformed prefix.
[[ "$DIR_NAME" =~ -step[0-9]+(_|$) ]] && fail "input looks like an ALREADY-EXPORTED dir ('-step<N>' in the name) — lint the TRAINING checkpoint_dir, not an export"
[[ "$PREFIX" =~ -step[0-9]+- ]] && fail "derived prefix has an embedded '-step<N>' before the runtype (malformed)"
# 3. no scaffolding leaked through
for bad in local megatron vlm; do
    [[ "$PREFIX" == *"-$bad-"* || "$PREFIX" == *"-$bad" ]] && fail "scaffolding token '$bad' survived"
done
# 4. reasoning <-> thinkon coherence
if [[ "$PREFIX" == *reasoning* && "$THINK" != "thinkon" ]]; then fail "'reasoning' in name but think=$THINK"; fi
if [[ "$THINK" == "thinkon" && "$PREFIX" != *reasoning* ]]; then fail "think=thinkon but no 'reasoning' token"; fi

echo "✅ LINT OK"
echo "   ckpt dir : $DIR_NAME"
echo "   prefix   : ${PREFIX}-step<N>$([[ "$THINK" == thinkon ]] && echo '_thinkon')"
echo "   runtype  : $(echo "$PREFIX" | grep -oE '(sft_grpo|sft|grpo)$')   thinking: $THINK"
