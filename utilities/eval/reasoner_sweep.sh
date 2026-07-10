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
#        /home/sgsilva/utilities/serve/start_vllm_server.sh \
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
#   VO_TEST_1105 / VO_TEST_1806 / VO_N_1105 / VO_N_1806
#                  (optional) per-cohort GT test dir + expected N overrides. The cohort is derived
#                  per-obs from the _1105_/_1806_ tag in the obs filename (untagged -> 1105).
#   MIN_COMPLETE   (optional) ABSOLUTE accept threshold override; default = 99% of the cohort's N.
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
# ---- cohort-aware GT routing (2026-07-02) ----
# The obs filename carries a cohort tag (_1105_/_1806_, from eval_all.sh VO_COHORT_TAG). The GT test
# dir + expected N MUST follow that tag — a fixed TEST_DIR scored 1806 obs against 1105 human GT
# (cohort mismatch, invalid stage2). Defaults below are per-cohort and env-overridable (mirrors
# eval_all.sh's VO_TEST pattern); an untagged obs stem falls back to 1105 (legacy single-cohort).
VO_TEST_1105="${VO_TEST_1105:-/mnt/data/shared/vlm/data/human_annotation_datasets/1105_not_reviewed/repetitions_test}"
VO_TEST_1806="${VO_TEST_1806:-/mnt/data/shared/vlm/data/human_annotation_datasets/1806_after_format_review_diverse_reasoning/repetitions_test}"
VO_N_1105="${VO_N_1105:-1181}"
VO_N_1806="${VO_N_1806:-2260}"
COMPILER=/home/sgsilva/utilities/eval/compile_eval_results.py
MAX_TOKENS="${MAX_TOKENS:-16384}"
# Client concurrency into evaluate.py. The default (10 for a --server-url run) overwhelms a HEAVY
# thinkON 27B reasoner: 10 parallel long-trace generations saturate it, requests exceed the 3-retry
# budget → "Failed to get response" (the 2026-06-22 sweep meltdown on the dedicated gpu:4 server).
# Throttle to 4 by default; bump only if the server's num_requests_waiting stays ~0 under load.
MAX_WORKERS="${MAX_WORKERS:-4}"
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
  # cohort routing: derive GT test dir + expected N from the obs stem's cohort tag.
  # 2026-07-06 audit fix (P2.2/P2.7): the old bare `*) cohort=1105` default silently routed ANY
  # untagged-for-1806 stem to 1105 GT — including a future THIRD cohort's obs (e.g. a `_1907_` tag),
  # which would then be scored against the WRONG ground truth with no warning. Detect a recognizable
  # but UNWIRED 4-digit cohort tag first and hard-fail on it; only a genuinely untagged (no 4-digit
  # tag at all — the legacy single-cohort naming) stem falls through to the 1105 default.
  if [[ "$obs_stem" =~ _1806(_|$) ]]; then
    cohort=1806; TEST_DIR="$VO_TEST_1806"; expect_n="$VO_N_1806"
  elif [[ "$obs_stem" =~ _1105(_|$) ]]; then
    cohort=1105; TEST_DIR="$VO_TEST_1105"; expect_n="$VO_N_1105"
  elif [[ "$obs_stem" =~ _([0-9]{4})(_|$) ]]; then
    echo "[FAIL] $obs_stem: unrecognized cohort tag '${BASH_REMATCH[1]}' (only 1105/1806 wired to a GT source) — refusing to guess"
    n_fail=$((n_fail+1)); continue
  else
    cohort=1105; TEST_DIR="$VO_TEST_1105"; expect_n="$VO_N_1105"  # genuinely untagged legacy stem
  fi
  if [[ ! -d "$TEST_DIR" ]]; then
    echo "[FAIL] cohort $cohort test dir missing: $TEST_DIR — skipping $obs_stem (would score vs nothing)"
    n_fail=$((n_fail+1)); continue
  fi
  out="$VO_RUNS/stage2_${out_stem}_${think}${TAG_SUFFIX}.json"
  # PREFLIGHT (2026-06-22): per-obs, show how many samples WILL be recomputed (universe − already
  # done) WITHOUT running anything. universe = reps in the obs file; done = per_sample_results in the
  # existing stage2 (a --resume run re-queries only the missing/previously-failed ones). Read-only.
  if [[ "${PREFLIGHT:-0}" == 1 ]]; then
    "$PY" - "$obs_file" "$out" "$obs_stem" <<'PY'
import json,sys,os
obs_f,out_f,stem=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(obs_f)); universe=sum(1 for s in d.values() if isinstance(s,dict)
            for k in s if "repetition" in k.lower())
done=0; failed=0
if os.path.exists(out_f):
    o=json.load(open(out_f)); done=len(o.get("per_sample_results") or [])
    failed=(o.get("metadata") or {}).get("failed_samples",0)
todo=universe-done
print(f"    PREFLIGHT {stem:40s} universe={universe:5d}  done={done:5d}  TO_RECOMPUTE={todo:5d}  (prev_failed={failed})")
PY
    continue
  fi
  # REPLACE mode: copy the existing old-reasoner base + _v2 into the v1 backup BEFORE overwriting.
  if [[ -z "$TAG_SUFFIX" && "${DRYRUN:-0}" != 1 ]]; then
    for old in "$VO_RUNS/stage2_${out_stem}_${think}.json" "$VO_RUNS/stage2_${out_stem}_${think}_v2.json"; do
      [[ -f "$old" ]] && cp -p "$old" "$V1_BAK/$(basename "$old")" && echo "    v1-backup: $(basename "$old")"
    done
  fi
  echo ">>> [$obs_stem] obs -> stage2 (reasoner=${REASONER_TAG:-NEW, canonical/REPLACE})"
  echo "    in : $(basename "$obs_file")"
  echo "    out: $(basename "$out")"
  echo "    GT : cohort $cohort -> $TEST_DIR (expect N=$expect_n)"
  cmd=( "$PY" "$EVAL_PY"
        --test-dataset-dir "$TEST_DIR"
        --two-stage --precomputed-visual-obs "$obs_file"
        --model "$REASONER" --server-url "$REASONER_URL"
        --max-tokens "$MAX_TOKENS"
        --max-workers "$MAX_WORKERS"
        --output-file "$out" --resume )
  if [[ "${DRYRUN:-0}" == 1 ]]; then
    printf '    DRYRUN: '; printf '%q ' "${cmd[@]}"; echo; continue
  fi
  # RESUME-UNTIL-COMPLETE (2026-06-22): transient server-overload failures (cold/contended
  # reasoner → "Failed to get response" / a length-capped runaway → "Failed to parse scores")
  # are NOT persisted to per_sample_results, so each --resume re-queries ONLY the missing/failed
  # samples (evaluate.py:2419 — failures go to failed_samples[], never to existing_results). We
  # want a FULL eval, so loop --resume until evaluated_samples==the cohort's N or attempts exhaust.
  # Convergence guard: if an attempt adds ZERO new samples (n unchanged), stop — a stuck obs
  # won't loop forever (distinct sentinel, never a silent short pass). [[feedback_eval_gotchas]]
  # MIN_COMPLETE: the thinkON reasoner has an intrinsic ~10% token-repetition-collapse tail (verified
  # 2026-06-23) — a handful of reps never parse, so demanding an exact N grinds for hours on the
  # last stochastic stragglers. Default = 99% of the cohort's expected N (was fixed 1170 when the
  # sweep was 1105-only); the board's VO Eval N column shows the real N/failed so a partial is never
  # hidden. Keeps the STUCK guard. MIN_COMPLETE env overrides ABSOLUTE (careful in mixed-cohort runs).
  min_complete="${MIN_COMPLETE:-$(( expect_n * 99 / 100 ))}"
  attempts="${RESUME_ATTEMPTS:-6}"; n=0; prev=-1; a=0
  while (( a < attempts )); do
    a=$((a+1))
    echo "    [resume attempt $a/$attempts] (have $n/$expect_n, accept >=$min_complete)"
    ( cd "$VPT" && "${cmd[@]}" ) || echo "    [warn] evaluate.py returned non-zero on attempt $a (partial save still topped off below)"
    n=$("$PY" -c "import json;print(json.load(open('$out')).get('metadata',{}).get('evaluated_samples',0))" 2>/dev/null || echo 0)
    fl=$("$PY" -c "import json;print(json.load(open('$out')).get('metadata',{}).get('failed_samples',0))" 2>/dev/null || echo 0)
    echo "    -> evaluated_samples=$n  failed=$fl  (target $expect_n, accept >=$min_complete)"
    (( n >= min_complete )) && break
    if [[ "$n" == "$prev" ]]; then
      echo "    [STUCK] attempt $a added 0 new samples (n=$n) — server may be down or these reps consistently fail. Stopping retries for this obs."
      break
    fi
    prev="$n"
  done
  if (( n >= min_complete )); then
    echo "    [OK] $obs_stem accepted ($n/$expect_n, >=$min_complete)"; n_ok=$((n_ok+1)); outputs+=("$out")
    # ---- RUN CARD (stabilization step 4, 2026-07-10): stamp identity on the stage2 result
    # so the board routes it card-first (no vo_tokens needed, no filename parsing). The VO
    # model under test isn't known to this script directly — recover it from the obs file's
    # own card (new eval_all runs) or its .owner sidecar (realpath, post-2026-06-22 runs).
    # TAG_SUFFIX mode = experimental/non-canonical files → NO card (deliberately off-board
    # unless explicitly tokened); legacy obs without card/.owner → filename routing as before.
    if [[ -z "$TAG_SUFFIX" ]]; then
      _ck=""
      [[ -f "${obs_file}.card.json" ]] && _ck=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('checkpoint_path','') or '')" "${obs_file}.card.json" 2>/dev/null)
      [[ -z "$_ck" && -f "${obs_file}.owner" ]] && _ck="$(cat "${obs_file}.owner" 2>/dev/null)"
      # Card cohort = the STEM's tag ('' for a legacy untagged stem) — NOT the GT-routing
      # cohort above: an untagged legacy stem scores vs 1105 GT but has ALWAYS joined the
      # BARE row; a card claiming cohort=1105 would re-key it and split the model's row.
      _card_cohort=""
      [[ "$obs_stem" =~ _1806(_|$) ]] && _card_cohort="1806"
      [[ "$obs_stem" =~ _1105(_|$) ]] && _card_cohort="1105"
      if [[ -n "$_ck" ]]; then
        for _t in "$out" "${out%.json}_v2.json"; do
          cat > "${_t}.card.json" <<CARDEOF || echo "    [card WARN] could not write ${_t}.card.json"
{
  "card_version": 1,
  "checkpoint_path": "${_ck}",
  "served_id": "${_ck}",
  "axis": "stage2",
  "arm": null,
  "obs_source": null,
  "reasoner": "${REASONER}",
  "cohort": "${_card_cohort}",
  "test_set": "${TEST_DIR}",
  "expected_n": ${expect_n},
  "thinking": "${think#think}",
  "run_id": "reasoner_sweep_${obs_stem}",
  "job_id": "${SLURM_JOB_ID:-}",
  "ts": "$(date -Is)"
}
CARDEOF
        done
        echo "    [card] stage2 cards written (checkpoint=$(basename "$_ck"))"
      else
        echo "    [card] no obs card/.owner for $obs_stem — stage2 stays on filename/vo_tokens routing (legacy)"
      fi
    fi
  else
    echo "    [INCOMPLETE] $obs_stem stalled at $n/$expect_n (<$min_complete) after $a attempts — NOT counting as done (re-run later against a warm server to top off)."
    n_fail=$((n_fail+1))
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
