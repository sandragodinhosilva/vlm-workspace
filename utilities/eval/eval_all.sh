#!/usr/bin/env bash
# eval_all.sh — modular eval driver for an ALREADY-SERVED model.
#
# Runs any subset of: aux (multimodal aux-tasks), benchmarks (VSI-Bench/MMMU/Video-MME),
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
#     --train-group-id mix_12k_1506 --run-id step1299_testset1506 --tag mix_12k_1506_1506testset \
#     [--thinking on|off] [--testset 1506] [--max-samples N]
#
# Stage notes:
#   aux         -> eval_multimodal_post_sft.sh --testset-1506 (needs --train-group-id + --run-id)
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
VO_TEST=/mnt/data/shared/vlm/data/human_annotation_datasets/1105_not_reviewed/repetitions_test
VO_OUT=/mnt/data/sgsilva/results/visual_obs/runs   # reorg 2026-06-17 (old visual_obs_runs/ back-compat-symlinked)
# GT visual-obs (human) for the agreement stage — per-rep entries carry `human_error_severities`
# (the HUMAN ground truth) + folder_name/repetition_id. Agreement = model-vs-GT (NOT vs oracle).
VO_GT_CAT=/mnt/data/shared/vlm/data/human_annotation_datasets/1105_not_reviewed_visual_obs/oracle/oracle_397b_1105_categorical_test.json
AGREE_PY=/home/sgsilva/vlm-post-training/data_preparation/analyze_observation_agreement.py

# ---- args ----
MODEL=""; BASE_MODEL="qwen3.5-4b"; STAGES=""; BASE_URL="http://localhost:8000/v1"
THINKING=""; TRAIN_GROUP_ID=""; RUN_ID=""; TAG=""; TESTSET="1506"; MAX_SAMPLES=""
BENCH_EXTRA=""; PREFLIGHT=0; JUDGE_BASE_URL=""; JUDGE_MODEL=""; BENCH_MAX_TOKENS=32768
SERVE=0; TP=""; MAX_LEN=""; SERVE_WAIT=1800; KEEP_SERVER=0   # --serve: launch+teardown our own vLLM
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
    --bench-max-tokens) BENCH_MAX_TOKENS="$2"; shift 2;;  # cap ALL benchmarks (VSI config + MMMU/VideoMME run_eval); thinkon runaway -> fail fast
    --preflight) PREFLIGHT=1; shift;;   # validate everything + exit WITHOUT running any eval
    --serve) SERVE=1; shift;;           # launch our OWN vLLM, run, then kill it on exit (sbatch-friendly)
    --serve-venv) SERVE_VENV="$2"; shift 2;;  # QWEN35_VENV for the serve (pmartins -> vlm-post-training-home-venv)
    --tp) TP="$2"; shift 2;;            # tensor-parallel size (default by base-model)
    --max-len) MAX_LEN="$2"; shift 2;;  # served max_model_len (default by base-model)
    --serve-wait) SERVE_WAIT="$2"; shift 2;;  # seconds to wait for server health (default 1800)
    --keep-server) KEEP_SERVER=1; shift;;  # leave OUR vLLM running after eval (reuse it; node frees only at job end)
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -z "$MODEL" || -z "$STAGES" ]] && { echo "ERROR: --model and --stages are required." >&2; exit 2; }
[[ "$BASE_URL" != */v1 ]] && BASE_URL="${BASE_URL%/}/v1"
LOG=$(log_start eval "eval_all_${TAG:-${RUN_ID:-notag}}")
exec > >(tee -a "$LOG") 2>&1

# ---- --serve: validate inputs that MUST be known before the server exists ----
if [[ "$SERVE" == 1 ]]; then
  # thinking can't be probed before the server is up -> must be explicit, and it sets ENABLE_THINKING.
  [[ "$THINKING" == on || "$THINKING" == off ]] || {
    echo "ERROR: --serve requires --thinking on|off (can't autodetect before the server is up; it sets ENABLE_THINKING)." >&2; exit 2; }
  # serve params: explicit flags win, else per-base-model defaults.
  if [[ -z "$TP" || -z "$MAX_LEN" ]]; then
    case "$BASE_MODEL" in
      *27b*) : "${TP:=8}"; : "${MAX_LEN:=65536}" ;;
      *)     : "${TP:=8}"; : "${MAX_LEN:=32768}" ;;   # 4b/9b default
    esac
  fi
  # derive host:port from --base-url (we serve there, then talk to it there)
  SERVE_PORT="$(printf '%s' "$BASE_URL" | sed -E 's#https?://[^:/]+:?([0-9]*)/.*#\1#')"
  [[ -z "$SERVE_PORT" ]] && SERVE_PORT=8000
  [[ ! -d "$MODEL" ]] && { echo "ERROR: --serve needs a model PATH on disk (got: $MODEL)" >&2; exit 2; }
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
on_exit() {
  if [[ "$KEEP_SERVER" == 1 ]]; then
    echo ">>> --keep-server: leaving vLLM (PID $OUR_SERVER_PID) running at $BASE_URL"
    echo "    served id: $SERVED_ID   kill it: kill -INT -$OUR_SERVER_PID   (node frees at job end regardless)"
  else
    teardown_server
  fi
}
if [[ "$SERVE" == 1 ]]; then
  [[ "$PREFLIGHT" == 1 ]] && { echo "ERROR: --serve and --preflight are mutually exclusive (preflight launches nothing)." >&2; exit 2; }
  trap on_exit EXIT
  trap teardown_server INT TERM
  ENABLE_BIT="$([[ "$THINKING" == on ]] && echo 1 || echo 0)"
  # log name: prefer RUN_ID (unique) so pmartins '.../step_N/hf' paths don't all collide on 'hf'.
  # Write under logs/eval/serve/ (NOT the logs root — strays there are flagged by the Stop hook).
  SERVE_TAG="${RUN_ID:-$(basename "$MODEL")}"
  mkdir -p /mnt/data/sgsilva/logs/eval/serve
  SERVE_LOG="/mnt/data/sgsilva/logs/eval/serve/eval_all_serve_${SERVE_TAG}_think${THINKING}.log"
  echo "==> --serve: launching vLLM  model=$(basename "$SERVED_ID")  TP=$TP  max_len=$MAX_LEN  port=$SERVE_PORT  thinking=$THINKING"
  echo "    serve venv: $SERVE_VENV"
  echo "    serve log: $SERVE_LOG   (host $(hostname))"
  # serve the SHORT path ($SERVED_ID; == $MODEL for normal paths, the _ext symlink for long ones)
  # so the server registers a short id and every downstream filename stays under the 255-char limit.
  # own process group (setsid) so teardown can signal the whole vLLM tree, not just the launcher
  setsid env ENABLE_THINKING="$ENABLE_BIT" QWEN35_VENV="$SERVE_VENV" \
    /home/sgsilva/vlm-evaluation/start_vllm_server.sh "$SERVED_ID" "$TP" "$MAX_LEN" "$SERVE_PORT" \
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
    [[ "$sid" == "$SERVED_ID" ]] || echo "  [WARN] served id != expected ('$sid' vs '$SERVED_ID') — eval model must equal the served id or requests 404"
    # 2. served max_model_len > planned eval max-tokens (aux 16384/8192, VO 32768/4096, bench 32768)
    local emax; emax="$([[ "$THINKING" == on ]] && echo 32768 || echo 16384)"
    if [[ "$smax" =~ ^[0-9]+$ ]]; then
      (( smax > emax )) && echo "  [ok]   max_model_len $smax > planned eval max-tokens $emax" \
                        || { echo "  [FAIL] max_model_len $smax NOT > eval max-tokens $emax (every request will 400)"; ok=0; }
    fi
  fi
  echo "  [ok]   thinking mode = $THINKING"
  # 4/5/6. per-stage prerequisites
  if have_stage aux; then
    [[ -n "$TRAIN_GROUP_ID" && -n "$RUN_ID" ]] && echo "  [ok]   aux: train-group-id + run-id present" \
      || { echo "  [FAIL] aux: --train-group-id and --run-id required"; ok=0; }
    [[ -e "$VPT/aux_tasks/sft/eval_multimodal_post_sft.sh" ]] || { echo "  [FAIL] aux driver missing"; ok=0; }
    if [[ "$TESTSET" == 1506 ]]; then
      [[ -d /mnt/data/shared/vlm/data/merged_aux_datasets/multimodal_reduced_testset_1506 ]] \
        && echo "  [ok]   aux: testset_1506 present" || { echo "  [FAIL] testset_1506 missing"; ok=0; }
    fi
  fi
  if have_stage benchmarks; then
    [[ -e "$BENCH_RUN" && -x "$BENCH_PY" ]] && echo "  [ok]   benchmarks: run_eval.py + venv present" \
      || { echo "  [FAIL] benchmarks: run_eval.py or SIBench venv missing (symlink /home/sgsilva/benchmarks?)"; ok=0; }
  fi
  if have_stage visualobs; then
    [[ -d "$VO_TEST" ]] && echo "  [ok]   visualobs: 1181-rep test dir present" || { echo "  [FAIL] visualobs test dir missing"; ok=0; }
    case "$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')" in
      *oracle*|*visual*obs*|*vo3d*) : ;;
      *) echo "  [WARN] visualobs: '$(basename "$MODEL")' not a visual-obs/oracle ckpt — severity eval may not apply" ;;
    esac
  fi
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
  stem="$(basename "$SERVED_ID" | sed 's/^qwen35-[0-9]*b-//')"
  vmax="$([[ "$THINKING" == on ]] && echo 32768 || echo 4096)"
  mkdir -p "$VO_OUT"
  vargs=( "$VPT_PY" "$VPT/data_preparation/evaluate.py"
    --test-dataset-dir "$VO_TEST"
    --model "$SERVED_ID" --server-url "$BASE_URL"
    --max-tokens "$vmax"
    --output-file "$VO_OUT/${stem}_singlestage_think${THINKING}.json" --resume )
  [[ "$THINKING" == off ]] && vargs+=( --disable-thinking )
  [[ -n "$MAX_SAMPLES" ]] && vargs+=( --max-samples "$MAX_SAMPLES" )
  vo_obs="$VO_OUT/${stem}_singlestage_think${THINKING}.json"
  if ( cd "$VPT" && "${vargs[@]}" ); then
    STAGE_RESULTS+=("visualobs: OK -> $vo_obs")
    # ---- AGREEMENT (auto, stage-1 obs vs HUMAN GT — no reasoner; the comparable single-stage VO
    # metric). model-vs-GT via --gt-source (human_error_severities); ±1 ordinal tolerance. ----
    echo ""; echo ">>> STAGE: agreement (stage-1 obs vs human GT)"
    agree_out="$VO_OUT/agreement_${stem}_think${THINKING}.json"
    aargs=( "$VPT_PY" "$AGREE_PY"
      --a "$vo_obs" --b "$VO_GT_CAT" --gt-source "$VO_GT_CAT"
      --label-a model --label-b gt --categorical-tolerance 1
      --output "$agree_out" )
    ( cd "$VPT" && "${aargs[@]}" ) \
      && STAGE_RESULTS+=("agreement: OK -> $agree_out") \
      || STAGE_RESULTS+=("agreement: FAILED")
  else
    STAGE_RESULTS+=("visualobs: FAILED")
  fi
fi

# ---------------- benchmarks (LAST — slowest, most likely to be cut short) ----------------
if have_stage benchmarks; then
  echo ""; echo ">>> STAGE: benchmarks (VSI-Bench / MMMU-val / Video-MME)"
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
    && STAGE_RESULTS+=("benchmarks: OK -> $BENCH_DIR/results/{vsibench,mmmu_val,video_mme}/$disp/") \
    || STAGE_RESULTS+=("benchmarks: FAILED")
  rm -f "$cfg"
fi

# ---------------- compile master CSV (additive; never touches per-stage outputs) ----------------
echo ""; echo ">>> Compiling unified master CSV (read-only union/join on served path)"
"$VPT_PY" "$(dirname "$0")/compile_eval_results.py" \
  && STAGE_RESULTS+=("master: OK -> /mnt/data/sgsilva/results/master/eval_master.csv") \
  || STAGE_RESULTS+=("master: FAILED")

echo ""
echo "=================================================="
echo " eval_all SUMMARY  (model=$(basename "$MODEL"), thinking=$THINKING)"
echo "=================================================="
for r in "${STAGE_RESULTS[@]}"; do echo "  $r"; done
