#!/usr/bin/env bash
# reasoner_sweep.sh — run ONE stage-2 reasoner across the saved stage-1 obs of EVERY board model,
# producing fresh two-stage results that fill the board's vo_s2_* columns.
#
# WHY: the two-stage VO numbers historically used a BASELINE Qwen3.5-27B as the stage-2 reasoner.
# To re-score with a better reasoner (e.g. the pmartins sft2812), we DON'T re-run inference on the
# models under test — we reuse their persisted stage-1 observations (obs_*.json, which carry the
# per-question parsed_answers) and feed each through `evaluate.py --two-stage --precomputed-visual-obs`
# with --model = the NEW reasoner. Output `stage2_<token>_<think>.json` is named so the master
# compiler's VO_FILE_TO_MODEL map joins it to the right board row.
#
# The reasoner is served ONCE on its own slot; this script loops the obs files against it (serial,
# since they share the one reasoner). Per the 2026-06-04 decision the stage-2 reasoner is normally
# served thinkOFF regardless of the stage-1 branch — but that's YOUR call at serve time; this script
# just points --server-url at whatever you serve.
#
# Usage:
#   1. Serve the reasoner on its OWN port (NOT localhost:8000 if that's the 397B):
#        # pmartins sft2812 needs the vlm-post-training venv (TokenizersBackend):
#        QWEN35_VENV=/home/sgsilva/vlm-post-training-home-venv \
#        /home/sgsilva/vlm-evaluation/start_vllm_server.sh \
#          /mnt/data/pmartins/vlm_ckpts/.../step_2812/hf 4 65536 <PORT>
#   2. Run the sweep:
#        REASONER=/mnt/data/pmartins/vlm_ckpts/.../step_2812/hf \
#        REASONER_URL=http://<host>:<PORT>/v1 \
#        REASONER_TAG=reasoner_sft2812 \
#        /home/sgsilva/utilities/eval/reasoner_sweep.sh
#   3. It recompiles the board at the end (vo_s2_* columns fill).
#
# Env:
#   REASONER       (required) served checkpoint path of the stage-2 reasoner (= --model).
#   REASONER_URL   (required) http://<host>:<port>/v1 of the reasoner server.
#   REASONER_TAG   (optional) short label appended to each stage2 filename. DEFAULT (empty) = REPLACE
#                  mode: writes the canonical stage2_<token>_<think>.json the compiler reads, so the new
#                  reasoner BECOMES the board vo_s2. The prior canonical stage2_*_v2.json files are
#                  auto-backed-up to _campaign/<date>_prev_reasoner/ first (non-destructive safety).
#                  Set a tag to instead write side-by-side files (board keeps the old numbers).
#   ONLY           (optional) space-separated list of obs stems to limit the sweep (substring match).
#   DRYRUN=1       (optional) print the commands without running.
#   MAX_TOKENS     (optional) stage-2 generation budget. Default 16384. The reasoner's THINKING mode
#                  is set at SERVE time (ENABLE_THINKING), NOT here — the sweep just queries whatever
#                  is served. A thinkON reasoner needs the larger budget (16384 fits); a thinkoff one
#                  could use less. NB: a thinkON reasoner is a DIFFERENT reasoner than the historical
#                  thinkoff base-27B (2026-06-04) — its stage2 numbers are a new experiment.
set -uo pipefail

VPT=/home/sgsilva/vlm-post-training
PY=/home/sgsilva/vlm-post-training-home-venv/bin/python
VO_RUNS=/mnt/data/sgsilva/results/visual_obs/runs
EVAL_PY="$VPT/eval/evaluate.py"
EVAL_VO_PY="$VPT/eval/evaluate_vo.py"
TEST_DIR=/mnt/data/shared/vlm/data/human_annotation_datasets/1105_not_reviewed/repetitions_test
COMPILER=/home/sgsilva/utilities/eval/compile_eval_results.py
MAX_TOKENS="${MAX_TOKENS:-16384}"
TAG_SUFFIX="${REASONER_TAG:+_${REASONER_TAG}}"

: "${REASONER:?set REASONER=<served stage-2 reasoner ckpt path>}"
: "${REASONER_URL:?set REASONER_URL=http://<host>:<port>/v1}"

# ---- targets are DISCOVERED, not hardcoded (modular: no model names in this script) ----
# The inputs are simply the obs_*.json files on disk (each = one model's saved stage-1 answers).
# For each, the stage2 OUTPUT is stage2_<obs-stem>_<think>.json — the obs-stem IS the join key the
# master compiler resolves to a board model via master_models.json `vo_path` / VO_FILE_TO_MODEL.
# So: glob obs files → derive (obs-stem, think, output-name) at runtime. A new model needs NOTHING
# here — only its obs_*.json on disk (from eval_all.sh visualobs) + its master_models.json entry.
# MODEL_CFG is the single source of truth (read only to VALIDATE that each obs stem maps to a known
# board model, so we don't silently produce an orphan stage2 the board can't join).
MODEL_CFG=/home/sgsilva/utilities/eval/master_models.json

# Build the (obs_file, obs_stem, think, out_name) work-list by globbing obs_*.json.
mapfile -t MAP < <("$PY" - "$VO_RUNS" "$MODEL_CFG" <<'PYEOF'
import json, sys, glob, os
vo_runs, cfg_path = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
# vo_path leaves + pattern substrings = the known board models (for validation only).
known = []
for e in cfg.get("models", []):
    if e.get("vo_path"):
        known.append((os.path.basename(e["vo_path"].rstrip("/")).lower(), e.get("display","")))
    known.append((e["pattern"].lower(), e.get("display","")))
for p in sorted(glob.glob(os.path.join(vo_runs, "obs_*.json"))):
    name = os.path.basename(p)
    stem = name[4:].rsplit("_think", 1)[0]      # obs_<stem>_think<mode>.json
    think = "thinkon" if "thinkon" in name.lower() else "thinkoff"
    sl = stem.lower()
    # validate: does this obs stem correspond to a known board model? (substring either direction)
    disp = next((d for tok, d in known if tok and (tok in sl or sl in tok or
                 os.path.basename(sl) in tok)), "")
    out_stem = stem                              # stage2_<obs-stem>_<think>.json (compiler resolves it)
    print(f"{stem}|{think}|{out_stem}|{disp}")
PYEOF
)

echo "=== reasoner sweep (targets discovered from obs_*.json + master_models.json) ==="
echo "  reasoner : $REASONER"
echo "  url      : $REASONER_URL"
echo "  tag      : ${REASONER_TAG:-<none — writes canonical stage2 names, REPLACES board s2>}"
echo "  obs dir  : $VO_RUNS"
echo "  config   : $MODEL_CFG"
echo "  found    : ${#MAP[@]} obs file(s) to sweep"
echo

# ---- v1 PRESERVATION (REPLACE mode only) ----
# Naming convention (user 2026-06-22): "v2 = new reasoner, v1 = as it was". In REPLACE mode (no
# REASONER_TAG) the new sweep overwrites the canonical stage2_<token>_<think>{,_v2}.json. Before that,
# copy the EXISTING (old baseline-27B reasoner) files to a dated v1 backup so they're preserved.
V1_BAK="/home/sgsilva/utilities/eval/_campaign/202606/reasoner_base27b_v1_$(date +%Y%m%d)"
if [[ -z "$TAG_SUFFIX" && "${DRYRUN:-0}" != 1 ]]; then
  mkdir -p "$V1_BAK"
  echo ">>> REPLACE mode: preserving existing (old-reasoner) stage2 files as v1 -> $V1_BAK"
fi

n_ok=0; n_skip=0; n_fail=0; outputs=()
for entry in "${MAP[@]}"; do
  IFS='|' read -r obs_stem think out_stem disp <<<"$entry"
  # ONLY filter
  if [[ -n "${ONLY:-}" ]]; then
    match=0; for o in $ONLY; do [[ "$obs_stem" == *"$o"* ]] && match=1; done
    [[ $match -eq 1 ]] || continue
  fi
  # validation: an obs stem that maps to NO board model would produce a stage2 the compiler can't
  # join (orphan) — warn loudly rather than silently sweep it.
  if [[ -z "$disp" ]]; then
    echo "[warn] obs stem '$obs_stem' matches NO master_models.json entry — stage2 would be an ORPHAN."
    echo "       add its master_models.json entry (with vo_path) first, OR it'll never reach the board."
  fi
  obs_file="$VO_RUNS/obs_${obs_stem}_${think}.json"
  if [[ ! -f "$obs_file" ]]; then
    echo "[skip] no obs yet: $(basename "$obs_file")  (run its VO eval first)"
    n_skip=$((n_skip+1)); continue
  fi
  out="$VO_RUNS/stage2_${out_stem}_${think}${TAG_SUFFIX}.json"
  # REPLACE mode: copy the existing old-reasoner base + _v2 into the v1 backup BEFORE overwriting.
  if [[ -z "$TAG_SUFFIX" && "${DRYRUN:-0}" != 1 ]]; then
    for old in "$VO_RUNS/stage2_${out_stem}_${think}.json" "$VO_RUNS/stage2_${out_stem}_${think}_v2.json"; do
      [[ -f "$old" ]] && cp -p "$old" "$V1_BAK/$(basename "$old")" && echo "    v1-backup: $(basename "$old")"
    done
  fi
  echo ">>> [$obs_stem] obs -> stage2 (reasoner=${REASONER_TAG:-NEW, canonical/REPLACE})"
  echo "    in : $(basename "$obs_file")"
  echo "    out: $(basename "$out")"
  cmd=( "$PY" "$EVAL_PY"
        --test-dataset-dir "$TEST_DIR"
        --two-stage --precomputed-visual-obs "$obs_file"
        --model "$REASONER" --server-url "$REASONER_URL"
        --max-tokens "$MAX_TOKENS"
        --output-file "$out" --resume )
  if [[ "${DRYRUN:-0}" == 1 ]]; then
    printf '    DRYRUN: '; printf '%q ' "${cmd[@]}"; echo; continue
  fi
  if ( cd "$VPT" && "${cmd[@]}" ); then
    # verify completeness: evaluated_samples must be 1181
    n=$("$PY" -c "import json,sys;print(json.load(open('$out')).get('metadata',{}).get('evaluated_samples','?'))" 2>/dev/null)
    echo "    -> evaluated_samples=$n (expect 1181)"
    [[ "$n" == 1181 ]] || echo "    [WARN] not 1181 — re-run with --resume to top off before trusting."
    n_ok=$((n_ok+1)); outputs+=("$out")
  else
    echo "    [FAIL] evaluate.py errored for $obs_stem"; n_fail=$((n_fail+1))
  fi
  echo
done

# ---- rescore (evaluate_vo) the stage2 outputs so the compiler's _v2 tier picks them ----
if [[ "${DRYRUN:-0}" != 1 && ${#outputs[@]} -gt 0 ]]; then
  echo ">>> evaluate_vo rescore (error-name-mismatch fix) on the new stage2 files"
  # evaluate_vo globs stage2_*.json in the runs dir; restrict to ones we just wrote by glob if tagged.
  glob="stage2_*${TAG_SUFFIX}.json"; [[ -z "$TAG_SUFFIX" ]] && glob="stage2_*.json"
  ( cd "$VPT" && "$PY" "$EVAL_VO_PY" --results-dir "$VO_RUNS" --glob "$glob" ) \
    && echo "    rescore OK" || echo "    [WARN] rescore failed; v1 stage2 json kept"
fi

echo
echo "=== sweep done: $n_ok ok, $n_skip skipped (no obs yet), $n_fail failed ==="
if [[ "${DRYRUN:-0}" != 1 && $n_ok -gt 0 ]]; then
  echo ">>> recompiling board (vo_s2_* columns fill from the new stage2 files)"
  "$PY" "$COMPILER" 2>&1 | grep -E "wrote|allowlist|WARN" || true
fi
echo
if [[ -z "$TAG_SUFFIX" ]]; then
  echo "NOTE: REPLACE mode — the NEW reasoner is now the board vo_s2 (v2). The OLD baseline-27B"
  echo "      reasoner files were preserved as v1 in: $V1_BAK"
else
  echo "NOTE: TAG mode — new files written as stage2_<token>_<think>${TAG_SUFFIX}.json (side by side)."
  echo "      The board s2 still reads the untagged canonical (old reasoner). Run with REASONER_TAG"
  echo "      empty to REPLACE the board s2 with the new reasoner (old set auto-saved as v1)."
fi
