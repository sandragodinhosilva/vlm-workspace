#!/usr/bin/env bash
# eval_all.sh — modular eval driver for an ALREADY-SERVED model.
#
# Runs any subset of: aux (multimodal aux-tasks), benchmarks (VSI-Bench/MMMU/Video-MME/IFBench),
# visualobs (visual-obs single-stage). Each stage uses its OWN venv and writes to its OWN
# canonical results root; this script only orchestrates. The model server (and, for two-stage
# visual-obs, a separate reasoner) must already be up — pass --base-url.
#
# Thinking mode is AUTODETECTED by probing the endpoint (override with --thinking on|off).
# Hard rules honored: home-venv for aux/VO; literal paths; no node assumptions; results ->
# canonical roots under /mnt/data/sgsilva + aux_tasks/evals.
#
# Usage:
#   eval_all.sh --model <PATH> --base-model qwen3.5-4b \
#     --stages aux,visualobs,benchmarks \
#     --base-url http://localhost:8000/v1 \
#     --train-group-id mix_12k_2906 --run-id step1299_testset2906 --tag mix_12k_2906_2906testset \
#     [--thinking on|off] [--testset 2906] [--max-samples N]
#
# Stage notes:
#   aux         -> eval_multimodal_post_sft.sh --testset-2906 (default; pass --testset 1506 for the
#                  superseded benchmark) (needs --train-group-id + --run-id)
#   benchmarks  -> AUTO-generates a temp reasoning:<mode> config from --model, runs
#                  benchmarks/scripts/run_eval.py, deletes the temp config. Skips/resumes per its
#                  own logic. (Pass --skip-videomme etc. through --bench-extra if desired.)
#   visualobs   -> evaluate.py single-stage on the 1181-rep test split. WARNS if --model does not
#                  look like a visual-obs/oracle checkpoint (wrong model family).

set -uo pipefail

# ---- logging (initialized after args so $TAG is available) ----
source /home/sgsilva/utilities/logs-utils/log_run.sh
# ---- paths (literal) ----
VPT=/home/sgsilva/vlm-post-training
VPT_PY=/home/sgsilva/vlm-post-training-home-venv/bin/python
BENCH_DIR=/home/sgsilva/benchmarks
BENCH_RUN=/home/sgsilva/benchmarks/scripts/run_eval.py
BENCH_PY=/home/sgsilva/benchmarks/SIBench-VSR/.venv/bin/python
# VO cohort/schema are ENV-OVERRIDABLE (added 2026-07-01 for the vobs2906 4-variant bake-off, which
# needs 2906 schema — categorical AND angle — on BOTH 1105 and 1806, none of which the hardcoded
# 1105-categorical defaults cover). Override per (cohort, schema) leaf via the sbatch env contract:
#   VO_TEST           test split dir            (1105: …/1105_not_reviewed/repetitions_test)
#   VO_PROCESSED_DIR  obs-gen processed dir     (1105: the test-only symlink set; 1806: the processed dir)
#   VO_GT_CAT         human-GT for agreement    (per cohort/schema oracle)
#   VO_SCHEMA         --visual-obs-variant      (categorical|angle; default categorical)
#   VO_OBS_FILE       explicit --visual-obs-file (2906 schema JSON; empty => resolve from variant)
#   VO_SESSIONS_FROM  --sessions-from filter    (1806 test-scoping; empty for 1105 which uses its symlink dir)
# All default to the historical 1105-categorical values so existing board runs are unchanged.
VO_TEST="${VO_TEST:-/mnt/data/shared/vlm/data/human_annotation_datasets/1105_not_reviewed/repetitions_test}"
VO_OUT=/mnt/data/sgsilva/results/visual_obs/runs   # reorg 2026-06-17 (old visual_obs_runs/ back-compat-symlinked)
# GT visual-obs (human) for the agreement stage — per-rep entries carry `human_error_severities`
# (the HUMAN ground truth) + folder_name/repetition_id. Agreement = model-vs-GT (NOT vs oracle).
VO_GT_CAT="${VO_GT_CAT:-/mnt/data/shared/vlm/data/human_annotation_datasets/1105_not_reviewed_visual_obs/oracle/oracle_397b_1105_categorical_test.json}"
VO_SCHEMA="${VO_SCHEMA:-categorical}"       # --visual-obs-variant (categorical|angle)
VO_OBS_FILE="${VO_OBS_FILE:-}"              # explicit --visual-obs-file (2906 schema); empty => resolve from variant
VO_SESSIONS_FROM="${VO_SESSIONS_FROM:-}"    # --sessions-from (1806 test-scoping); empty => none (1105)
AGREE_PY=/home/sgsilva/vlm-post-training/visual_obs/analyze_observation_agreement.py   # moved from data_preparation/ (reorg)
# Agreement needs the model's STAGE-1 OBSERVATIONS (shape {ex:{rep:{parsed_answers}}}), NOT the
# single-stage SEVERITY json evaluate.py emits ({metadata,metrics,per_sample_results:[...]}) —
# feeding the latter as --a crashes the agreement script (AttributeError: 'list' has no 'keys').
# generate_visual_observations_human.py produces the correct obs shape. PROCESSED_DIR is the
# TEST-ONLY symlink set (132 samples / exactly the 1181 test reps) — the canonical recipe's dir
# (memory: reference_visual_obs_eval_commands + bash history), NOT the full train+test processed
# dir (that would over-generate ~4.5x; the join discards the surplus). --visual-obs-file +
# categorical variant match the recipe. --resume → stable per-model obs file.
VO_OBS_GEN=/home/sgsilva/vlm-post-training/visual_obs/generate_visual_observations_human.py   # moved from data_preparation/ (reorg)
# evaluate_vo: re-scores the single-stage SEVERITY json with the error-NAME-MISMATCH fix
# (feedback_eval_gotchas §4) WITHOUT re-running the model. The compiler's _vo_tier PREFERS *_v2.json,
# so without this pass a fresh single-stage VO run is scored under the OLD buggy name-join logic
# while neighbors use the fix. Reorg 2026-06-19: moved to eval/ + renamed evaluate_vo.py (was
# data_preparation/results/evaluate_v2.py), co-located with the evaluate.py it modifies.
# Agreement (stage-1, index-based) does NOT need it.
VO_EVAL_V2=/home/sgsilva/vlm-post-training/eval/evaluate_vo.py
VO_PROCESSED_DIR="${VO_PROCESSED_DIR:-/mnt/data/sgsilva/datasets/1105/1105_test_processed_symlinks}"   # moved under 1105/ (was datasets/ root)
# (the categorical question file resolves automatically from --visual-obs-variant categorical →
#  repo-root visual_obs/visual_observations_categorical.json; no explicit --visual-obs-file needed.
#  For 2906 schema pass VO_OBS_FILE=…/visual_obs/visual_observations_{categorical,angle}_2906.json.)

# ---- args ----
MODEL=""; BASE_MODEL="qwen3.5-4b"; STAGES=""; BASE_URL="http://localhost:8000/v1"
THINKING=""; TRAIN_GROUP_ID=""; RUN_ID=""; TAG=""; TESTSET="2906"; MAX_SAMPLES=""
BENCH_EXTRA=""; PREFLIGHT=0; JUDGE_BASE_URL=""; JUDGE_MODEL=""; BENCH_MAX_TOKENS=32768; BENCH_MAX_TOKENS_SET=0
SERVE=0; TP=""; MAX_LEN=""; SERVE_WAIT=1800; KEEP_SERVER=0   # --serve: launch+teardown our own vLLM
FULL_REBUILD=0   # --full-rebuild: end-of-run board rebuild uses --full-scan (picks up exporter/
# compiler CODE changes); default is --incremental (fast; only the new run's rows land — correct
# when only DATA changed, which is the common case).
SERVE_VENV="/home/sgsilva/qwen3.5-serving-home-venv"   # --serve-venv override for ckpts needing another
# stack (e.g. pmartins transformers-5.x 'TokenizersBackend' tokenizers -> vlm-post-training-home-venv)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --judge-base-url) JUDGE_BASE_URL="$2"; shift 2;;
    --judge-model) JUDGE_MODEL="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --base-model) BASE_MODEL="$2"; shift 2;;
    --stages) STAGES="$2"; shift 2;;
    --base-url) BASE_URL="$2"; shift 2;;
    --thinking) THINKING="$2"; shift 2;;
    --train-group-id) TRAIN_GROUP_ID="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --testset) TESTSET="$2"; shift 2;;
    --max-samples) MAX_SAMPLES="$2"; shift 2;;
    --bench-extra) BENCH_EXTRA="$2"; shift 2;;
    --bench-max-tokens) BENCH_MAX_TOKENS="$2"; BENCH_MAX_TOKENS_SET=1; shift 2;;  # cap ALL benchmarks (VSI config + MMMU/VideoMME run_eval); thinkon runaway -> fail fast
    --preflight) PREFLIGHT=1; shift;;   # validate everything + exit WITHOUT running any eval
    --serve) SERVE=1; shift;;           # launch our OWN vLLM, run, then kill it on exit (sbatch-friendly)
    --serve-venv) SERVE_VENV="$2"; shift 2;;  # QWEN35_VENV for the serve (pmartins -> vlm-post-training-home-venv)
    --tp) TP="$2"; shift 2;;            # tensor-parallel size (default: GPUs allocated to this job, see --serve below)
    --max-len) MAX_LEN="$2"; shift 2;;  # served max_model_len (default by base-model)
    --serve-wait) SERVE_WAIT="$2"; shift 2;;  # seconds to wait for server health (default 1800)
    --keep-server) KEEP_SERVER=1; shift;;  # leave OUR vLLM running after eval (reuse it; node frees only at job end)
    --full-rebuild) FULL_REBUILD=1; shift;;  # end-of-run board rebuild = --full-scan (use after an exporter/compiler code change)
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -z "$MODEL" || -z "$STAGES" ]] && { echo "ERROR: --model and --stages are required." >&2; exit 2; }
[[ "$BASE_URL" != */v1 ]] && BASE_URL="${BASE_URL%/}/v1"
# Run name disambiguates by THINKING mode (same ckpt gives different numbers on/off) and stages,
# so on/off runs of one model never collapse to one log name. LOG_CMD records the REAL invocation
# (model/stages/thinking/tp), not the bare $0, so the log header is reproducible.
_run_name="eval_all_${TAG:-${RUN_ID:-notag}}_think${THINKING:-NA}_$(echo "$STAGES" | tr ',' '-')"
# Record --bench-extra too: its --skip-* flags are the signal eval_status.sh uses to know WHICH
# benchmarks actually run (e.g. an IFBench-only run skips VSI/MMMU/Video-MME) → correct ETA.
LOG_CMD="$0 --model $MODEL --base-model $BASE_MODEL --stages $STAGES --thinking ${THINKING:-NA} --base-url $BASE_URL${TP:+ --tp $TP}${RUN_ID:+ --run-id $RUN_ID}${BENCH_EXTRA:+ --bench-extra \"$BENCH_EXTRA\"}"
export LOG_CMD
LOG=$(log_start eval "$_run_name")
# Tee stdout+stderr to BOTH the dated $LOG and the original fds, so the SLURM .out/.err still
# mirror everything for live `tail -f` (an `exec >> $LOG` alone leaves the slurm out EMPTY).
# The catch tee normally introduces — its proc-sub subshell suppresses log_start's BUILT-IN EXIT
# trap, so the `==== RUN END ====` footer never writes (crashed runs look clean = silent fail) —
# is sidestepped by NOT relying on that trap: we install our OWN explicit EXIT trap below, which
# fires in this shell regardless of the tee subshell (verified). So we get both: slurm mirror + footer.
exec > >(tee -a "$LOG") 2>&1
# log_start's OWN trap also can't fire here (LOG was captured via command substitution → its
# _LOG_RUN_PID != this shell's $$ → its PID-guarded trap skips finalize). Our explicit trap writes
# the footer for EVERY exit path (the 9 error-exits below, crashes, signals, normal end); log_end
# is idempotent so a later explicit call is a harmless no-op.
trap 'log_end "$LOG" "$?"' EXIT

# ---- --serve: validate inputs that MUST be known before the server exists ----
if [[ "$SERVE" == 1 ]]; then
  # thinking can't be probed before the server is up -> must be explicit, and it sets ENABLE_THINKING.
  [[ "$THINKING" == on || "$THINKING" == off ]] || {
    echo "ERROR: --serve requires --thinking on|off (can't autodetect before the server is up; it sets ENABLE_THINKING)." >&2; exit 2; }
  # serve params: explicit --tp wins; else default to the GPUs ACTUALLY allocated to this job
  # (CUDA_VISIBLE_DEVICES count) so a packed gpu:4 job uses TP=4, a full-node job uses TP=8.
  # A hardcoded TP=8 on a 4-GPU alloc fails to start the server. MAX_LEN stays per-base-model.
  if [[ -z "$TP" ]]; then
    n_gpu="$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | tr ',' '\n' | grep -c .)"
    TP="${n_gpu:-8}"
  fi
  if [[ -z "$MAX_LEN" ]]; then
    case "$BASE_MODEL" in
      *27b*) MAX_LEN=65536 ;;
      *)     MAX_LEN=32768 ;;   # 4b/9b default
    esac
  fi
  # derive host:port from --base-url (we serve there, then talk to it there)
  SERVE_PORT="$(printf '%s' "$BASE_URL" | sed -E 's#https?://[^:/]+:?([0-9]*)/.*#\1#')"
  [[ -z "$SERVE_PORT" ]] && SERVE_PORT=8000
  [[ ! -d "$MODEL" ]] && { echo "ERROR: --serve needs a model PATH on disk (got: $MODEL)" >&2; exit 2; }
  [[ -x /home/sgsilva/utilities/serve/start_vllm_server.sh ]] || { echo "ERROR: --serve needs start_vllm_server.sh (not found/executable at /home/sgsilva/utilities/serve/start_vllm_server.sh)" >&2; exit 2; }
fi

# ---- LONG paths (>120 chars, e.g. external 225-char pmartins ckpts): serve+eval via a SHORT
# SYMLINK under models/_ext/. Every downstream script (serve, aux per-task log, VLMEvalKit result
# slug, the request model field) then sees a short path automatically — one fix instead of patching
# each script's filename builder. The compiler's _norm_path resolves the symlink back to the real
# path, so the join is unchanged. SERVED_ID == the path actually used everywhere below. ----
SERVED_ID="$MODEL"
if [[ ${#MODEL} -gt 120 ]]; then
  short_name="${RUN_ID:-$(basename "$(dirname "$MODEL")")_$(basename "$MODEL")}"
  short_name="$(printf '%s' "$short_name" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-60)"
  SERVED_ID="/mnt/data/sgsilva/models/_ext/${short_name}"
  mkdir -p /mnt/data/sgsilva/models/_ext
  find /mnt/data/sgsilva/models/_ext -maxdepth 1 -xtype l -delete 2>/dev/null  # prune dangling (deleted-ckpt) symlinks
  ln -sfn "$MODEL" "$SERVED_ID"
  [[ -f "$SERVED_ID/config.json" ]] || { echo "ERROR: short symlink $SERVED_ID does not resolve to a model (config.json missing)" >&2; exit 1; }
  echo "==> long model path (${#MODEL} chars) -> short symlink: $SERVED_ID -> $MODEL"
fi

# ---- autodetect thinking mode (reuses the run_eval.py probe shape) ----
if [[ -z "$THINKING" ]]; then
  echo "==> Autodetecting thinking mode at $BASE_URL ..."
  detected="$("$VPT_PY" - "$BASE_URL" "$SERVED_ID" <<'PY'
import json,sys,urllib.request
base,model=sys.argv[1],sys.argv[2]
url=base.rstrip("/")+"/chat/completions"
# served model id = the model path string; if a different id is registered, fall back to it
try:
    with urllib.request.urlopen(urllib.request.Request(base.rstrip("/")+"/models"),timeout=30) as r:
        ids=[m["id"] for m in json.loads(r.read())["data"]]
    if model not in ids and ids: model=ids[0]
except Exception: pass
payload=json.dumps({"model":model,"messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":256,"temperature":0}).encode()
req=urllib.request.Request(url,data=payload,headers={"Content-Type":"application/json","Authorization":"Bearer sk-dummy"})
try:
    with urllib.request.urlopen(req,timeout=120) as resp: data=json.loads(resp.read())
    msg=data["choices"][0]["message"]; content=msg.get("content","") or ""
    reasoning=msg.get("reasoning_content") or msg.get("reasoning") or ""
    print("on" if (reasoning or "<think>" in content) else "off")
except Exception as e:
    print("ERROR:"+str(e))
PY
)"
  if [[ "$detected" == on || "$detected" == off ]]; then
    THINKING="$detected"; echo "    detected thinking=$THINKING"
  else
    echo "ERROR: could not autodetect thinking mode ($detected). Pass --thinking on|off." >&2; exit 1
  fi
fi
[[ "$THINKING" == on || "$THINKING" == off ]] || { echo "ERROR: --thinking must be on|off" >&2; exit 2; }

# AUTO-CAP benchmark max-tokens for thinkON: a thinkon-27B rambles to max_tokens on hard MMMU/
# Video-MME/IFBench items (~30min/sample at 32768 → Video-MME ≈ days, ~27% non-responses; IFBench's
# long-output constraints — "≥170 unique words" etc — are also a slow tail). If the user didn't pass
# --bench-max-tokens explicitly, drop the default 32768 → 16384 when thinking=on so runaways fail
# fast (~2-4min) while real answers (well under 16384) are untouched. Explicit --bench-max-tokens
# always wins. thinkOFF keeps the full budget (IFBench's slow tail is then the pole — expect it).
if [[ "$BENCH_MAX_TOKENS_SET" == 0 && "$THINKING" == on ]]; then
  BENCH_MAX_TOKENS=16384
  echo "==> thinkON + no explicit --bench-max-tokens: auto-capping benchmarks to $BENCH_MAX_TOKENS (runaway guard)"
fi

have_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# ---- --serve: launch OUR OWN vLLM, wait for health, kill it on exit (sbatch-friendly) ----
# Lets one sbatch job serve+eval+teardown so the node frees when the job ends. Teardown kills
# ONLY the PID we launched (hard rule: never pkill-by-pattern — could hit another user's vLLM).
OUR_SERVER_PID=""
teardown_server() {
  [[ -z "$OUR_SERVER_PID" ]] && return 0
  if kill -0 "$OUR_SERVER_PID" 2>/dev/null; then
    echo ">>> Tearing down our vLLM server (PID $OUR_SERVER_PID) + its process group"
    kill -INT "-$OUR_SERVER_PID" 2>/dev/null || kill -INT "$OUR_SERVER_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$OUR_SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "-$OUR_SERVER_PID" 2>/dev/null || true
  fi
}
# EXIT handler: respect --keep-server on a NORMAL exit (reuse the server); INT/TERM (crash/cancel)
# ALWAYS tear down so a killed job frees the node — never leave a stray 8-GPU server behind.
# ALSO writes the run-log footer: this trap REPLACES the bare `trap 'log_end…' EXIT` set at the
# top (a second `trap … EXIT` clobbers the first), so under --serve the footer must be written
# HERE or it's lost. Capture $? FIRST (teardown/echo would overwrite it).
on_exit() {
  local rc=$?
  if [[ "$KEEP_SERVER" == 1 ]]; then
    echo ">>> --keep-server: leaving vLLM (PID $OUR_SERVER_PID) running at $BASE_URL"
    echo "    served id: $SERVED_ID   kill it: kill -INT -$OUR_SERVER_PID   (node frees at job end regardless)"
  else
    teardown_server
  fi
  log_end "$LOG" "$rc"   # idempotent; writes ==== RUN END ==== footer (status/exit/duration)
}
if [[ "$SERVE" == 1 ]]; then
  [[ "$PREFLIGHT" == 1 ]] && { echo "ERROR: --serve and --preflight are mutually exclusive (preflight launches nothing)." >&2; exit 2; }
  trap on_exit EXIT
  trap teardown_server INT TERM
  ENABLE_BIT="$([[ "$THINKING" == on ]] && echo 1 || echo 0)"
  # log name: prefer RUN_ID (unique) so pmartins '.../step_N/hf' paths don't all collide on 'hf'.
  # Write under logs/eval/serve/<DATE>/ (NOT the logs root — strays there are flagged by the Stop
  # hook). Dated subdir + jobid/timestamp suffix so reruns of one config DON'T overwrite each
  # other (the flat name was reused every launch → each new serve clobbered the prior log).
  SERVE_TAG="${RUN_ID:-$(basename "$MODEL")}"
  # RUN_ID may already carry the thinking mode (…_thinkoff); only append _think<mode> if absent,
  # to avoid the doubled '…_thinkoff_thinkoff' seen before.
  case "$SERVE_TAG" in *think${THINKING}*) _serve_think="" ;; *) _serve_think="_think${THINKING}" ;; esac
  _serve_date="$(date -u +%Y-%m-%d)"; _serve_stamp="$(date -u +%H%M%S)"
  _serve_id="${SLURM_JOB_ID:-p$$}"
  mkdir -p "/mnt/data/sgsilva/logs/eval/serve/${_serve_date}"
  SERVE_LOG="/mnt/data/sgsilva/logs/eval/serve/${_serve_date}/eval_all_serve_${SERVE_TAG}${_serve_think}__${_serve_id}_${_serve_stamp}.log"
  echo "==> --serve: launching vLLM  model=$(basename "$SERVED_ID")  TP=$TP  max_len=$MAX_LEN  port=$SERVE_PORT  thinking=$THINKING"
  echo "    serve venv: $SERVE_VENV"
  echo "    serve log: $SERVE_LOG   (host $(hostname))"
  # serve the SHORT path ($SERVED_ID; == $MODEL for normal paths, the _ext symlink for long ones)
  # so the server registers a short id and every downstream filename stays under the 255-char limit.
  # own process group (setsid) so teardown can signal the whole vLLM tree, not just the launcher
  setsid env ENABLE_THINKING="$ENABLE_BIT" QWEN35_VENV="$SERVE_VENV" \
    /home/sgsilva/utilities/serve/start_vllm_server.sh "$SERVED_ID" "$TP" "$MAX_LEN" "$SERVE_PORT" \
    >"$SERVE_LOG" 2>&1 &
  OUR_SERVER_PID=$!
  echo "    server PID (process group): $OUR_SERVER_PID  — monitor: tail -f $SERVE_LOG"
  # poll /v1/models until the served id appears, OR the launcher dies, OR we time out
  echo "==> waiting up to ${SERVE_WAIT}s for server health at $BASE_URL ..."
  ready=0
  for ((waited=0; waited<SERVE_WAIT; waited+=10)); do
    if ! kill -0 "$OUR_SERVER_PID" 2>/dev/null; then
      echo "ERROR: vLLM launcher exited during startup — see $SERVE_LOG" >&2; tail -20 "$SERVE_LOG" >&2; exit 1
    fi
    if "$VPT_PY" - "$BASE_URL" <<'PY' 2>/dev/null
import json,sys,urllib.request
base=sys.argv[1]
try:
    with urllib.request.urlopen(base.rstrip("/")+"/models",timeout=8) as r:
        sys.exit(0 if json.loads(r.read()).get("data") else 1)
except Exception: sys.exit(1)
PY
    then ready=1; echo "    server healthy after ~${waited}s"; break; fi
    sleep 10
  done
  [[ "$ready" == 1 ]] || { echo "ERROR: server not healthy within ${SERVE_WAIT}s — see $SERVE_LOG" >&2; tail -20 "$SERVE_LOG" >&2; exit 1; }
fi

# ---- PREFLIGHT: validate everything cheaply, then (if --preflight) exit before any eval ----
run_preflight() {
  local ok=1
  echo ""; echo "===== PREFLIGHT ====="
  # 1. server reachable + does the served id match --model?
  local served; served="$("$VPT_PY" - "$BASE_URL" "$SERVED_ID" <<'PY'
import json,sys,urllib.request
base,model=sys.argv[1],sys.argv[2]
try:
    with urllib.request.urlopen(urllib.request.Request(base.rstrip("/")+"/models"),timeout=20) as r:
        d=json.loads(r.read())["data"][0]
    print(f"{d['id']}|{d.get('max_model_len','?')}")
except Exception as e:
    print("ERR:"+str(e))
PY
)"
  if [[ "$served" == ERR:* ]]; then
    echo "  [FAIL] server unreachable at $BASE_URL ($served)"; ok=0
  else
    local sid="${served%%|*}" smax="${served##*|}"
    echo "  [ok]   server up: id=$sid  max_model_len=$smax"
    # HARD GATE (was a non-fatal WARN): a served id != expected is the PORT-collision data-corruption
    # signature — the eval would query the WRONG model's server and log plausible garbage. Refuse.
    if [[ "$sid" != "$SERVED_ID" ]]; then
      echo "  [FAIL] served id != expected ('$sid' vs '$SERVED_ID') — refusing to eval the WRONG model"
      echo "         (likely a shared-PORT collision: another job's server is bound to $BASE_URL)."
      ok=0
    fi
    # 2. served max_model_len > planned eval max-tokens (aux 16384/8192, VO 32768/4096, bench 32768)
    local emax; emax="$([[ "$THINKING" == on ]] && echo 32768 || echo 16384)"
    if [[ "$smax" =~ ^[0-9]+$ ]]; then
      (( smax > emax )) && echo "  [ok]   max_model_len $smax > planned eval max-tokens $emax" \
                        || { echo "  [FAIL] max_model_len $smax NOT > eval max-tokens $emax (every request will 400)"; ok=0; }
    fi
  fi
  echo "  [ok]   thinking mode = $THINKING"
  # 3b. master_models.json allowlist: is THIS model listed? The board is allowlist-gated, so a
  # model that isn't listed runs to COMPLETED but stays INVISIBLE on the master CSV (the 94484
  # miss). WARN (not FAIL) — the eval is still valid, but you almost always want the row. Add the
  # entry NOW (before the hours of compute) rather than rediscovering the gap after. Matches by the
  # same case-insensitive substring rule the compiler uses (pattern in served PATH or basename).
  local allowlist="$(dirname "$0")/master_models.json"
  if [[ -f "$allowlist" ]]; then
    if "$VPT_PY" - "$allowlist" "$MODEL" <<'PY'
import json,sys
al,model=sys.argv[1],sys.argv[2].lower()
pats=[m.get("pattern","").lower() for m in json.load(open(al)).get("models",[])]
sys.exit(0 if any(p and p in model for p in pats) else 1)
PY
    then echo "  [ok]   master_models.json: '$(basename "$MODEL")' is allowlisted (will appear on the board)"
    else echo "  [WARN] master_models.json: '$(basename "$MODEL")' NOT allowlisted — eval will run but stay INVISIBLE on the board."
         echo "         Add a {pattern,display,group} entry to $allowlist NOW, then re-run (or recompile after)."
    fi
  fi
  # 4/5/6. per-stage prerequisites
  if have_stage aux; then
    [[ -n "$TRAIN_GROUP_ID" && -n "$RUN_ID" ]] && echo "  [ok]   aux: train-group-id + run-id present" \
      || { echo "  [FAIL] aux: --train-group-id and --run-id required"; ok=0; }
    [[ -e "$VPT/aux_tasks/sft/eval_multimodal_post_sft.sh" ]] || { echo "  [FAIL] aux driver missing"; ok=0; }
    if [[ "$TESTSET" == 1506 ]]; then
      [[ -d /mnt/data/shared/vlm/data/merged_aux_datasets/multimodal_reduced_testset_1506 ]] \
        && echo "  [ok]   aux: testset_1506 present" || { echo "  [FAIL] testset_1506 missing"; ok=0; }
    fi
    if [[ "$TESTSET" == 2906 ]]; then
      [[ -d /mnt/data/shared/vlm/data/merged_aux_datasets/multimodal_reduced_testset_2906 ]] \
        && echo "  [ok]   aux: testset_2906 present" || { echo "  [FAIL] testset_2906 missing"; ok=0; }
    fi
    # OUTPUT MANIFEST (aux): the per-run tree + the rich eval_matrix the board reads.
    _aux_run="${RUN_ID:-<run-id>}"
    _aux_dir="$VPT/aux_tasks/evals/$BASE_MODEL/multimodal/${TRAIN_GROUP_ID:-<group>}/final_sft/$_aux_run"
    echo "  aux OUTPUT MANIFEST:"
    echo "    $([[ -d "$_aux_dir" ]] && echo '↻ EXISTS  ' || echo '+ NEW     ') $_aux_dir/"
    echo "                 ^ per-run tree: video/text/image legs + results/multimodal_*.json (the BOARD aux source) [via aux_tasks/sft/eval_multimodal_post_sft.sh → aggregate_multimodal_eval.py]"
    echo "    ↻ APPEND    /mnt/data/sgsilva/results/aux/eval_matrix.csv  (+ eval_matrix_${BASE_MODEL}.csv)"
    echo "                 ^ rich aux master — a NEW row appended for run-id '$_aux_run' (the board's Aux columns) [via aux_tasks/evals/export_eval_matrix.py]"
  fi
  if have_stage benchmarks; then
    [[ -e "$BENCH_RUN" && -x "$BENCH_PY" ]] && echo "  [ok]   benchmarks: run_eval.py + venv present" \
      || { echo "  [FAIL] benchmarks: run_eval.py or SIBench venv missing (symlink /home/sgsilva/benchmarks?)"; ok=0; }
    # OUTPUT MANIFEST (benchmarks): per-bench result dir (keyed by the benchmark DISPLAY = run-id/tag)
    # + the summary CSVs the board reads. The display dir name is whatever run_eval.py uses; show the root.
    echo "  benchmarks OUTPUT MANIFEST  [via benchmarks/scripts/run_eval.py; judged via run_judge_all.py + rescore_*.py]:"
    for _b in mmmu_val video_mme vsibench; do
      echo "    + RESULT    /mnt/data/sgsilva/results/benchmarks/$_b/<display>/   (+ ${_b}_judged/ if --judge-*)"
    done
    echo "    + RESULT    /mnt/data/sgsilva/results/benchmarks/ifbench/<display>/   (text-only, RULE-scored — NO judge/_judged)"
    echo "    ↻ APPEND    /mnt/data/sgsilva/results/benchmarks/summary.csv  (+ summary_judge.csv if judged)"
    echo "                 ^ the BOARD MMMU/Video-MME/VSI/IF-Bench source (compiler prefers summary_judge.csv) [via benchmarks/scripts/collect_results.py]"
    # ---- POISONED-REUSE GUARD (2026-06-22): a prior run whose server died leaves prediction files
    # that are 100% "Failed to obtain answer via API.". --reuse silently REUSES them, re-scores the
    # all-failed predictions, and reports `benchmarks: OK` with ZERO valid rows — the board cell then
    # goes (correctly) BLANK while the run looks successful (the 4B-baseline VSI carried this across 3
    # runs: T20260618→20→21). Detect it BEFORE launch: any reuse-target *_score.xlsx that is
    # >=90% API-failure → [FAIL] (delete the poisoned T*/ dir and re-run WITHOUT --reuse). [[feedback_eval_gotchas]]
    # ifbench included (2026-07-06 audit fix, P1.4): it was excluded from this loop, so a dead-server
    # IFBench cache re-scored to a low-but-plausible number with NO poisoned-reuse warning (unlike
    # vsibench/mmmu/video_mme, which at least get flagged). IFBench also has no judge-rescue fallback
    # (rule-scored only), so a poisoned cache there is a permanently wrong number until caught here.
    _bench_disp="$(basename "${SERVED_ID:-${RUN_ID:-$MODEL}}")-think${THINKING}"
    for _b in vsibench mmmu_val video_mme ifbench; do
      _bdir="/mnt/data/sgsilva/results/benchmarks/$_b/$_bench_disp"
      [[ -d "$_bdir" ]] || continue
      _poison="$("$VPT_PY" - "$_bdir" <<'PY'
import sys,glob,os
try: import pandas as pd
except Exception: sys.exit(0)   # can't check -> stay silent (don't false-FAIL)
bdir=sys.argv[1]; FAIL="Failed to obtain answer via API"
worst=None
# IFBench result files are named "*_IFBench.xlsx"/"*_IFBench_result.xlsx", not "*_score.xlsx"
# (VLMEvalKit's naming for the other 3 benchmarks) — without this pattern the guard silently
# never matched any ifbench file (2026-07-06 audit fix, P1.4).
_globs = ["*_score.xlsx", "*_IFBench.xlsx", "*_IFBench_result.xlsx"]
_files = set()
for _g in _globs:
    _files.update(glob.glob(os.path.join(bdir,"**",_g),recursive=True))
    _files.update(glob.glob(os.path.join(bdir,_g)))
for f in sorted(_files):
    try: df=pd.read_excel(f)
    except Exception: continue
    col="prediction" if "prediction" in df.columns else None
    if not col or len(df)==0: continue
    frac=df[col].astype(str).str.contains(FAIL,na=False).mean()
    if worst is None or frac>worst[0]: worst=(frac,os.path.basename(f))
if worst and worst[0]>=0.90:
    print(f"{worst[0]*100:.0f} {worst[1]}")
PY
)"
      if [[ -n "$_poison" ]]; then
        echo "  [FAIL] benchmarks: POISONED reuse cache in $_bdir"
        echo "         ${_poison%% *}% of predictions = 'Failed to obtain answer via API' (e.g. ${_poison#* })."
        echo "         --reuse would re-score these failures as OK and leave the board cell BLANK."
        echo "         FIX: rm -rf $_bdir/T*  then re-run the benchmark stage WITHOUT --reuse."
        ok=0
      fi
    done
  fi
  if have_stage visualobs; then
    [[ -d "$VO_TEST" ]] && echo "  [ok]   visualobs: 1181-rep test dir present" || { echo "  [FAIL] visualobs test dir missing"; ok=0; }
    # driver + agreement scripts must exist (a reorg of data_preparation/ would otherwise abort the
    # stage mid-run with no preflight warning — the class-1 moved-script failure mode).
    [[ -f "$VPT/eval/evaluate.py" ]] || { echo "  [FAIL] visualobs: evaluate.py missing"; ok=0; }
    [[ -f "$AGREE_PY" ]] || { echo "  [FAIL] visualobs: analyze_observation_agreement.py missing"; ok=0; }
    [[ -f "$VO_OBS_GEN" ]] || { echo "  [FAIL] visualobs: obs generator ($VO_OBS_GEN) missing — agreement stage can't run"; ok=0; }
    [[ -d "$VO_PROCESSED_DIR" ]] && echo "  [ok]   visualobs: agreement processed-dir present" \
      || echo "  [WARN] visualobs: agreement processed-dir missing ($VO_PROCESSED_DIR) — agreement will SKIP"
    [[ -f "$VO_GT_CAT" ]] || { echo "  [FAIL] visualobs: agreement GT file ($VO_GT_CAT) missing"; ok=0; }
    case "$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')" in
      *oracle*|*visual*obs*|*vo3d*) : ;;
      *) echo "  [WARN] visualobs: '$(basename "$MODEL")' not a visual-obs/oracle ckpt — severity eval may not apply" ;;
    esac
    # ---- OUTPUT MANIFEST: every file this run WILL create, with a per-file status so a collision is
    # visible BEFORE launch (2026-06-22). Stem = same derivation as the run (full served basename).
    # At preflight SERVED_ID may be unset (server not up for --serve) → fall back to MODEL/RUN_ID.
    _pf_stem="$(basename "${SERVED_ID:-${RUN_ID:-$MODEL}}")"; [[ -n "${VO_COHORT_TAG:-}" ]] && _pf_stem="${_pf_stem}_${VO_COHORT_TAG}"
    _pf_real="$("$VPT_PY" -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$MODEL" 2>/dev/null)"
    echo "  visualobs OUTPUT MANIFEST (stem='$_pf_stem') — FULL PATHS so you can inspect/cat them:"
    _pf_made=()
    for _pf_pair in \
      "singlestage raw severity scores [via eval/evaluate.py]|$VO_OUT/${_pf_stem}_singlestage_think${THINKING}.json" \
      "↳ auto-rescored from the raw above (no re-run; error-name-mismatch fix) — THE BOARD vo_s1 value [via eval/evaluate_vo.py]|$VO_OUT/${_pf_stem}_singlestage_think${THINKING}_v2.json" \
      "stage-1 obs: per-question answers — the reasoner input [via visual_obs/generate_visual_observations_human.py]|$VO_OUT/obs_${_pf_stem}_think${THINKING}.json" \
      "agreement vs human GT — THE BOARD vo_agree value [via visual_obs/analyze_observation_agreement.py]|$VO_OUT/agreement_${_pf_stem}_think${THINKING}.json"; do
      _lbl="${_pf_pair%%|*}"; _pf="${_pf_pair##*|}"
      if [[ -f "$_pf" ]]; then
        _own=""
        case "$_pf" in
          *singlestage*) _own="$("$VPT_PY" -c "import json,sys;print((json.load(open(sys.argv[1])).get('metadata') or {}).get('model',''))" "$_pf" 2>/dev/null)";;
          *obs_*) [[ -f "${_pf}.owner" ]] && _own="$(cat "${_pf}.owner" 2>/dev/null)";;
        esac
        if [[ -n "$_own" && -n "$_pf_real" && "$("$VPT_PY" -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$_own" 2>/dev/null)" != "$_pf_real" ]]; then
          echo "    ⚠ COLLISION  $_pf"
          echo "                 ^ $_lbl — OWNED BY $_own — run WILL refuse to overwrite"; ok=0
        else
          echo "    ↻ RESUME     $_pf"
          echo "                 ^ $_lbl — exists (same model); --resume tops it off"
        fi
      else
        echo "    + NEW        $_pf"
        echo "                 ^ $_lbl"
      fi
      _pf_made+=("$_pf")
    done
    echo "    + LATER      $VO_OUT/stage2_${_pf_stem}_think${THINKING}{,_v2}.json"
    echo "                 ^ two-stage — THE BOARD vo_s2 value — only when a reasoner sweep runs over the obs above [via utilities/eval/reasoner_sweep.sh → eval/evaluate.py --two-stage, then eval/evaluate_vo.py rescore]"
    echo "  inspect after the run:  ls -la ${_pf_made[*]} 2>/dev/null"
    # ---- BOARD ROUTING simulation (stabilization step 2, 2026-07-10): the SAME resolve_vo()
    # the compiler uses, run over this launch's PLANNED filenames — shows the exact row key +
    # display entry each artifact will land on, BEFORE any compute. This run's singlestage/
    # agreement artifacts also get run-cards (step 4) so they route without vo_tokens; a
    # "⚠ NO pattern matches" line below still means the row gets DROPPED by the allowlist —
    # add the master_models.json entry in the SAME turn as this launch (feedback §7).
    echo ""
    echo "  BOARD ROUTING (simulated with compile_eval_results.py --route):"
    "$VPT_PY" "$(dirname "$0")/compile_eval_results.py" --route \
        "$VO_OUT/${_pf_stem}_singlestage_think${THINKING}.json" \
        "$VO_OUT/agreement_${_pf_stem}_think${THINKING}.json" \
        "$VO_OUT/stage2_${_pf_stem}_think${THINKING}.json" 2>/dev/null | sed 's/^/    /' \
      || echo "    ⚠ ROUTING: at least one artifact above would be INVISIBLE on the board — add the master_models.json entry (pattern + vo_tokens) NOW, not after the run."
  fi
  # ---- ALWAYS-WRITTEN outputs (any stage): the unified board + this run's log ----
  echo "  ALWAYS OUTPUT MANIFEST (every run, all stages):"
  echo "    ↻ REBUILD   /mnt/data/sgsilva/results/master/eval_master.csv  (+ _${BASE_MODEL##qwen3.5-} split)"
  echo "                 ^ the unified board — recompiled from all sources at the end of the run [via utilities/eval/compile_eval_results.py]"
  echo "    + LOG       /mnt/data/sgsilva/logs/eval/<date>/eval_all_<run>_think${THINKING}_<stages>__<jobid>_<ts>.log"
  [[ "$SERVE" == 1 ]] && echo "    + SERVELOG  /mnt/data/sgsilva/logs/eval/serve/<date>/eval_all_serve_<run>__<jobid>_<ts>.log"
  echo "===== PREFLIGHT $([[ $ok == 1 ]] && echo PASS || echo FAIL) ====="
  return $((1 - ok))
}

run_preflight || { echo "Preflight failed — fix the [FAIL]s above before running." >&2; exit 1; }
if [[ "$PREFLIGHT" == 1 ]]; then
  echo "(--preflight: validation only, no eval launched.)"; exit 0
fi

echo "=================================================="
echo " eval_all: model=$MODEL"
echo "          base_model=$BASE_MODEL  thinking=$THINKING  base_url=$BASE_URL"
echo "          stages=$STAGES"
echo "=================================================="

declare -a STAGE_RESULTS=()

# ---------------- aux ----------------
if have_stage aux; then
  echo ""; echo ">>> STAGE: aux (multimodal aux-tasks, testset $TESTSET)"
  if [[ -z "$TRAIN_GROUP_ID" || -z "$RUN_ID" ]]; then
    echo "[aux] SKIP: --train-group-id and --run-id are required for the aux stage." >&2
    STAGE_RESULTS+=("aux: SKIPPED (missing train-group-id/run-id)")
  else
    # eval_multimodal_post_sft.sh's --enable-thinking expects true|false (NOT on|off).
    et="$([[ "$THINKING" == on ]] && echo true || echo false)"
    vmax="$([[ "$THINKING" == on ]] && echo 16384 || echo 8192)"
    # avoid a doubled _thinkon/_thinkoff if the caller already put it in --run-id/--tag.
    base_run="${RUN_ID%_think${THINKING}}"; base_run="${base_run%_thinkon}"; base_run="${base_run%_thinkoff}"
    base_tag="${TAG:-$base_run}"; base_tag="${base_tag%_think${THINKING}}"; base_tag="${base_tag%_thinkon}"; base_tag="${base_tag%_thinkoff}"
    aux_tag="${base_tag}_think${THINKING}"
    aux_run="${base_run}_think${THINKING}"
    args=( "$VPT/aux_tasks/sft/eval_multimodal_post_sft.sh"
      --model "$SERVED_ID" --base-model "$BASE_MODEL"
      --train-group-id "$TRAIN_GROUP_ID" --eval-family final_sft
      --server-url "${BASE_URL%/v1}" --api-base "${BASE_URL%/v1}"
      --enable-thinking "$et"
      --max-tokens "$vmax" --video-max-tokens "$vmax"
      --tag "$aux_tag" --run-id "$aux_run" )
    [[ "$TESTSET" == 1506 ]] && args+=( --testset-1506 )
    [[ "$TESTSET" == 2906 ]] && args+=( --testset-2906 )
    [[ -n "$MAX_SAMPLES" ]] && args+=( --max-samples "$MAX_SAMPLES" )
    ( cd "$VPT" && "${args[@]}" ) \
      && STAGE_RESULTS+=("aux: OK -> $VPT/aux_tasks/evals/$BASE_MODEL/multimodal/$TRAIN_GROUP_ID/final_sft/$aux_run/") \
      || STAGE_RESULTS+=("aux: FAILED")
  fi
fi

# ---------------- visualobs (single-stage) ----------------
# ORDER: aux -> visualobs -> benchmarks. Benchmarks run LAST because they are the slowest
# (thinkon-27B runaway tail on MMMU/Video-MME) and the most likely to be cut short — so the
# cheap/fast axes (aux, VO) land FIRST and a kill loses only the benchmark tail, not VO/aux.
if have_stage visualobs; then
  echo ""; echo ">>> STAGE: visualobs (single-stage, 1181-rep test)"
  case "$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')" in
    *oracle*|*visual*obs*|*vo3d*) : ;;
    *) echo "[visualobs] ⚠ WARNING: '$(basename "$MODEL")' does not look like a visual-obs/oracle"
       echo "            checkpoint. Single-stage severity eval on the 1181-rep test may not apply"
       echo "            to this model family. Proceeding anyway (you selected it)." ;;
  esac
  # stem from SERVED_ID (the short symlink / run-id), NOT $MODEL — basename($MODEL) is 'hf' for
  # every pmartins '.../step_N/hf', which COLLIDES all their VO output files (A overwrote B once).
  # DO NOT strip the qwen35-<N>b- base-model prefix: it is exactly what distinguishes a 4B from a 27B
  # sibling (e.g. qwen35-4b-mix-12k-1506-sft-step1299 vs qwen35-27b-...). Stripping it collided the
  # two and the 27B run overwrote the 4B's VO (data loss, 2026-06-22). The full basename is already
  # unique — pmartins resolves to a unique _ext/<run_id> symlink, not 'hf'. Keep the full basename.
  stem="$(basename "$SERVED_ID")"
  # VO_COHORT_TAG (2026-07-01): the obs/singlestage/agreement filenames derive from $stem (the served
  # model basename) — which is IDENTICAL for two cohorts of the same model (e.g. the vobs2906 bake-off
  # ran each 4B on BOTH 1105 and 1806). Without a cohort suffix the second cohort's run CLOBBERS the
  # first's obs (cohort contamination). Append the tag so each cohort gets its own file:
  #   obs_<model>_1105_thinkoff.json vs obs_<model>_1806_thinkoff.json. Empty => unchanged (single-cohort).
  [[ -n "${VO_COHORT_TAG:-}" ]] && stem="${stem}_${VO_COHORT_TAG}"
  # ---- RUN CARD writer (stabilization step 4, 2026-07-10): every routed VO artifact gets a
  # <file>.card.json sidecar stating its identity (checkpoint/axis/cohort/thinking/test set)
  # AT GENERATION TIME. The board compiler routes card-first (resolve_vo), so a carded file
  # reaches its row with NO vo_tokens entry and NO filename parsing — the invisible-row and
  # wrong-row-merge classes end here for eval_all-produced artifacts. Never fatal.
  write_card() {
    local _t="$1" _axis="$2"
    cat > "${_t}.card.json" <<CARDEOF || { echo "[card WARN] could not write ${_t}.card.json"; return 0; }
{
  "card_version": 1,
  "checkpoint_path": "${SERVED_ID}",
  "served_id": "${SERVED_ID}",
  "base_model": "${BASE_MODEL:-}",
  "axis": "${_axis}",
  "arm": null,
  "obs_source": null,
  "reasoner": null,
  "cohort": "${VO_COHORT_TAG:-}",
  "test_set": "${VO_TEST:-}",
  "expected_n": null,
  "thinking": "${THINKING}",
  "run_id": "${RUN_ID:-}",
  "job_id": "${SLURM_JOB_ID:-}",
  "ts": "$(date -Is)"
}
CARDEOF
  }
  vmax="$([[ "$THINKING" == on ]] && echo 32768 || echo 4096)"
  mkdir -p "$VO_OUT"
  # ---- COLLISION GUARD (defense-in-depth, 2026-06-22): never overwrite ANOTHER model's VO file.
  # If the target stem already exists AND its metadata.model is a DIFFERENT served path than ours,
  # a stem collision would silently clobber that model's data (what happened to 4B-vs-27B mix-12k).
  # Abort the stage LOUDLY instead — the operator picks a distinct run-id/stem. --resume is safe
  # ONLY when the existing file is the SAME model (a genuine resume), which this allows.
  _vo_target="$VO_OUT/${stem}_singlestage_think${THINKING}.json"
  _vo_collision=0
  if [[ -f "$_vo_target" ]]; then
    _existing_model="$("$VPT_PY" -c "import json,sys;print((json.load(open(sys.argv[1])).get('metadata') or {}).get('model',''))" "$_vo_target" 2>/dev/null)"
    _norm() { "$VPT_PY" -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$1" 2>/dev/null; }
    if [[ -n "$_existing_model" && "$(_norm "$_existing_model")" != "$(_norm "$SERVED_ID")" ]]; then
      echo "  [FAIL] visualobs: STEM COLLISION — $_vo_target already holds a DIFFERENT model:"
      echo "         existing: $_existing_model"
      echo "         this run: $SERVED_ID"
      echo "         Refusing to overwrite (would lose the other model's VO). Use a distinct --run-id/stem."
      # NOTE: must contain the literal substring "FAILED" (not "FAIL") — the exit-code check
      # at the bottom of this script greps STAGE_RESULTS for *FAILED*; "FAIL (...)" alone was
      # silently passing the run as exit 0 while the whole visualobs stage was refused
      # (2026-07-06 audit fix, P2.4).
      STAGE_RESULTS+=("visualobs: FAILED (stem collision — would overwrite $_existing_model)")
      _vo_collision=1
    fi
  fi
  vargs=( "$VPT_PY" "$VPT/eval/evaluate.py"
    --test-dataset-dir "$VO_TEST"
    --model "$SERVED_ID" --server-url "$BASE_URL"
    --max-tokens "$vmax"
    --output-file "$_vo_target" --resume )
  [[ "$THINKING" == off ]] && vargs+=( --disable-thinking )
  [[ -n "$MAX_SAMPLES" ]] && vargs+=( --max-samples "$MAX_SAMPLES" )
  vo_obs="$_vo_target"
  if [[ "${_vo_collision:-0}" == 1 ]]; then
    echo "  visualobs: SKIPPED (collision guard tripped — see [FAIL] above)"
  elif ( cd "$VPT" && "${vargs[@]}" ); then
    STAGE_RESULTS+=("visualobs: OK -> $vo_obs")
    write_card "$vo_obs" singlestage
    # ---- evaluate_v2 RESCORE (error-name-mismatch fix; no model re-run). Writes <stem>_..._v2.json
    # next to the original (never overwrites); the compiler prefers the _v2 tier. Scope the glob to
    # THIS run's single-stage file so we don't rescore the whole dir every time. Non-fatal: a rescore
    # failure leaves the (older-logic) singlestage json in place rather than failing the run. ----
    echo ""; echo ">>> visualobs: evaluate_v2 rescore (error-name-mismatch fix)"
    if ( cd "$VPT" && "$VPT_PY" "$VO_EVAL_V2" --results-dir "$VO_OUT" --glob "${stem}_singlestage_think${THINKING}.json" ); then
      STAGE_RESULTS+=("visualobs_v2: OK -> ${vo_obs%.json}_v2.json")
      write_card "${vo_obs%.json}_v2.json" singlestage
    else
      STAGE_RESULTS+=("visualobs_v2: WARN (rescore failed; singlestage json kept)")
    fi
    # ---- AGREEMENT (auto, model stage-1 obs vs HUMAN GT — no reasoner; the comparable single-stage
    # VO metric). model-vs-GT via --gt-source (human_error_severities); ±1 ordinal tolerance.
    # Two steps (canonical recipe = memory reference_visual_obs_eval_commands step a→b):
    # (1) generate the model's stage-1 OBSERVATIONS over the 1181 test reps (the correct
    # {ex:{rep:{parsed_answers}}} shape — NOT the single-stage severity json above, which crashes
    # the agreement script); (2) compare obs vs GT. --resume to a stable per-model file so a re-run
    # / partial doesn't redo finished reps. max-tokens per recipe (4096 off / 32768 on). ----
    echo ""; echo ">>> STAGE: agreement — step 1/2: generate stage-1 observations (1181 test reps)"
    obs_out="$VO_OUT/obs_${stem}_think${THINKING}.json"
    # COLLISION GUARD for obs (2026-06-22): obs files carry NO metadata.model, so we track ownership
    # via a sidecar <obs>.owner. If an existing obs is owned by a DIFFERENT model, --resume would reuse
    # (and the agreement would score the WRONG model's obs — exactly the 4B/27B mix-12k contamination).
    # Refuse; the operator picks a distinct stem. A matching/absent owner = safe (genuine resume / fresh).
    _obs_owner="${obs_out}.owner"; _served_real="$("$VPT_PY" -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$SERVED_ID" 2>/dev/null)"
    if [[ -f "$obs_out" && -f "$_obs_owner" && "$(cat "$_obs_owner" 2>/dev/null)" != "$_served_real" ]]; then
      echo "  [FAIL] agreement: obs STEM COLLISION — $obs_out is owned by a DIFFERENT model:"
      echo "         owner:    $(cat "$_obs_owner" 2>/dev/null)"
      echo "         this run: $_served_real"
      echo "         Refusing to reuse/overwrite. Use a distinct --run-id/stem."
      # Must contain "FAILED" (see the visualobs collision-guard fix above for why).
      STAGE_RESULTS+=("agreement: FAILED (obs stem collision)")
      _vo_collision=1
    fi
    if [[ "${_vo_collision:-0}" != 1 ]]; then echo "$_served_real" > "$_obs_owner"; fi
    if [[ "${_vo_collision:-0}" != 1 && -d "$VO_PROCESSED_DIR" ]]; then
      omax="$([[ "$THINKING" == on ]] && echo 32768 || echo 4096)"
      oargs=( "$VPT_PY" "$VO_OBS_GEN"
        --model "$SERVED_ID" --server-url "$BASE_URL"
        --processed-dir "$VO_PROCESSED_DIR"
        --visual-obs-variant "$VO_SCHEMA"
        --max-tokens "$omax" --max-workers 16
        --output-file "$obs_out" --resume )
      # 2906 schema: pass the explicit question file (categorical/angle _2906.json) — else the
      # generator resolves the OLD repo-root categorical file from the variant name. (feedback: 2906)
      [[ -n "$VO_OBS_FILE" ]] && oargs+=( --visual-obs-file "$VO_OBS_FILE" )
      # 1806 test-scoping: restrict the processed-dir walk to the test split's session_ids (added
      # 2026-06-30, symmetric with --exclude-csv). Empty for 1105 (its symlink dir is already scoped).
      [[ -n "$VO_SESSIONS_FROM" ]] && oargs+=( --sessions-from "$VO_SESSIONS_FROM" )
      [[ "$THINKING" == off ]] && oargs+=( --disable-thinking )
      [[ -n "$MAX_SAMPLES" ]] && oargs+=( --limit "$MAX_SAMPLES" )
      if ( cd "$VPT" && "${oargs[@]}" ); then
        # obs-gen (the reasoner input) ALWAYS runs above; the agreement COMPARISON below is skippable
        # (VO_SKIP_AGREEMENT=1) for cohorts with no schema-matched GT — e.g. 1105 has no 2906 oracle,
        # so comparing 2906 model-obs against the old-schema GT would be a mismatched, misleading number.
        # Added 2026-07-01 for the vobs2906 bake-off (skip on 1105, run on 1806 vs the GT-on oracle).
        if [[ "${VO_SKIP_AGREEMENT:-0}" == 1 ]]; then
          echo "[agreement] SKIP: VO_SKIP_AGREEMENT=1 (obs-gen kept; no schema-matched GT for this cohort)" >&2
          STAGE_RESULTS+=("agreement: SKIPPED (VO_SKIP_AGREEMENT=1); obs-gen OK -> $obs_out")
        else
        echo ">>> STAGE: agreement — step 2/2: obs vs human GT"
        agree_out="$VO_OUT/agreement_${stem}_think${THINKING}.json"
        aargs=( "$VPT_PY" "$AGREE_PY"
          --a "$obs_out" --b "$VO_GT_CAT" --gt-source "$VO_GT_CAT"
          --label-a model --label-b gt --categorical-tolerance 1
          --output "$agree_out" )
        if ( cd "$VPT" && "${aargs[@]}" ); then
          STAGE_RESULTS+=("agreement: OK -> $agree_out")
          write_card "$agree_out" agreement
        else
          STAGE_RESULTS+=("agreement: FAILED")
        fi
        fi
      else
        STAGE_RESULTS+=("agreement: FAILED (obs generation)")
      fi
    else
      echo "[agreement] SKIP: processed-dir not found ($VO_PROCESSED_DIR)" >&2
      STAGE_RESULTS+=("agreement: SKIPPED (no processed-dir)")
    fi
  else
    STAGE_RESULTS+=("visualobs: FAILED")
  fi
fi

# ---------------- benchmarks (LAST — slowest, most likely to be cut short) ----------------
if have_stage benchmarks; then
  echo ""; echo ">>> STAGE: benchmarks (VSI-Bench / MMMU-val / Video-MME / IFBench)"
  # --max-samples does NOT apply here: run_eval.py / VLMEvalKit has no sample-limit flag, so the
  # benchmarks stage ALWAYS runs the full sets. Warn loudly rather than silently ignoring the cap
  # (a silent no-op once made a "smoke" run execute all 300 IFBench prompts). [[feedback_no_silent_fail]]
  if [[ -n "$MAX_SAMPLES" ]]; then
    echo "  ⚠ --max-samples=$MAX_SAMPLES is IGNORED for benchmarks (run_eval.py has no sample cap)." >&2
    echo "    The benchmarks stage runs FULL sets. To smoke-test cheaply, use --stages aux (cap applies there)" >&2
    echo "    or run a single short benchmark via --skip-* for the long ones." >&2
  fi
  reason_bool="$([[ "$THINKING" == on ]] && echo true || echo false)"
  # display = basename of SERVED_ID (the short symlink for long paths) — UNIQUE per run. Using the
  # raw $MODEL basename would be 'hf' for every pmartins '.../step_N/hf', colliding all their
  # benchmark result dirs + the master display. SERVED_ID basename = e.g. grpo_step492_thinkon.
  disp="$(basename "$SERVED_ID")-think${THINKING}"
  # TEMP config (deleted after the run) — no configs/ accumulation needed. compile_eval_results.py
  # recovers display_name -> served path by decoding the result-tree model_slug, NOT from configs.
  cfg="/mnt/data/sgsilva/tmp/_eval_all_bench_${disp}.json"
  mkdir -p /mnt/data/sgsilva/tmp
  "$VPT_PY" - "$cfg" "$SERVED_ID" "$disp" "$reason_bool" "$BASE_URL" "$BENCH_MAX_TOKENS" <<'PY'
import json,sys
cfg,model,disp,reason,base,maxtok=sys.argv[1:7]
short="bench-"+disp
out={"display_name":disp,"reasoning":(reason=="true"),
 "model":{short:{"class":"OpenAIWrapper","model":model,
   "api_base":base.rstrip("/")+"/chat/completions","key":"sk-dummy",
   "temperature":0,"max_tokens":int(maxtok),"img_detail":"high",
   "system_prompt":"End your response with \\boxed{X} on the last line where X is the option letter."}},
 "data":{t:{"class":"SIBench","dataset":t} for t in
   ["Counting","Height","Existence","Object_Localization","Spatial_Relation"]}}
json.dump(out,open(cfg,"w"),indent=4)
print("wrote",cfg)
PY
  bargs=( "$BENCH_PY" "$BENCH_RUN" --config "$cfg" --base-url "$BASE_URL" --max-tokens "$BENCH_MAX_TOKENS" )
  # Optional judge rescore (parsing-rescue): catches right-but-unparsed \boxed{} / prose answers,
  # writes *_judged + summary_judge.csv. Needs a SEPARATE judge server (NOT the model-under-test).
  # compile_eval_results.py prefers summary_judge.csv when present.
  if [[ -n "$JUDGE_BASE_URL" && -n "$JUDGE_MODEL" ]]; then
    bargs+=( --judge-base-url "$JUDGE_BASE_URL" --judge-model "$JUDGE_MODEL" )
    echo "    judge: $JUDGE_MODEL @ $JUDGE_BASE_URL"
  else
    echo "    judge: SKIPPED (pass --judge-base-url + --judge-model to enable the rescore)"
  fi
  [[ -n "$BENCH_EXTRA" ]] && bargs+=( $BENCH_EXTRA )
  ( cd "$BENCH_DIR" && "${bargs[@]}" ) \
    && STAGE_RESULTS+=("benchmarks: OK -> $BENCH_DIR/results/{vsibench,mmmu_val,video_mme,ifbench}/$disp/") \
    || STAGE_RESULTS+=("benchmarks: FAILED")
  rm -f "$cfg"
fi

# ---------------- re-export aux eval_matrix + compile board (one wrapper) ----------------
# The aux stage writes only per-RUN aggregates; eval_matrix.csv (the compiler's PRIMARY aux input)
# is built separately and is "only up to date after re-exporting". Without it a finished aux run is
# INVISIBLE to the board (the 94484 miss). rebuild_board.sh does the FULL correct sequence: regen
# the COMBINED matrix AND each per-base file (a multi-base export writes ONLY the combined file, so
# the per-base files — which the compiler reads FIRST/PRIMARY — must be regenerated separately or
# they go STALE and SHADOW the combined values), runs the staleness guard, then compiles. We use
# Default --incremental (per-run: only the new aux run lands; fast). Pass --full-rebuild to this
# script when an exporter/compiler CODE change must propagate to ALL rows (incremental reuses the
# cache and would skip unchanged rows). Backup IS taken either way: every eval run snapshots the key
# CSVs to results/_backups/<ts>/ BEFORE touching them, so any board rebuild is reversible and the
# AFTER diff shows exactly what changed. See [[feedback_backup_before_mutating]].
_rebuild_mode="--incremental"; [[ "$FULL_REBUILD" == 1 ]] && _rebuild_mode="--full-scan"
echo ""; echo ">>> Rebuilding aux eval_matrix + master board (rebuild_board.sh $_rebuild_mode; backup + diff)"
"$(dirname "$0")/rebuild_board.sh" $_rebuild_mode \
  && STAGE_RESULTS+=("board: OK (backed up) -> /mnt/data/sgsilva/results/master/eval_master.csv") \
  || STAGE_RESULTS+=("board: WARN/FAILED (matrix or compile issue — check rebuild_board output)")

echo ""
echo "=================================================="
echo " eval_all SUMMARY  (model=$(basename "$MODEL"), thinking=$THINKING)"
echo "=================================================="
for r in "${STAGE_RESULTS[@]}"; do echo "  $r"; done

# Exit with failed RC if ANY stage recorded FAILED — the EXIT trap above reads $? and writes the
# footer as status=failed/done accordingly (so a half-broken run no longer logs as clean).
_eval_rc=0
for r in "${STAGE_RESULTS[@]}"; do [[ "$r" == *FAILED* ]] && _eval_rc=1; done
exit "$_eval_rc"
