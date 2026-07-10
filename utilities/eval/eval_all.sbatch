#!/bin/bash
#SBATCH --job-name=eval-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=1200G
#SBATCH --output=/mnt/data/sgsilva/logs/eval/slurm/eval_all_slurm-%j.out
#SBATCH --error=/mnt/data/sgsilva/logs/eval/slurm/eval_all_slurm-%j.err
#
# DEFAULT = HALF NODE (gpu:4, 96 CPU, 1200G) so TWO gpu:4 evals PACK onto one 8-GPU/192-CPU/
# 2489G node (partition OverSubscribe=YES:4). The old full-node default left half the GPUs idle:
# --cpus-per-task=192 + no --mem grabbed ALL CPUs+RAM, so nothing else could land even with 4
# free GPUs.  PORT is auto-derived from the allocated GPUs below so two co-located jobs DON'T
# collide on 8000 (no manual PORT= needed when packing).
#   - gpu:4 job (default)          : just submit; two pack per node automatically.
#   - FULL-NODE job (TP=8 / 397B)  : sbatch --gres=gpu:8 --cpus-per-task=192 --mem=2400G ...
#     (the 27B/397B aux+serve need all 8 GPUs; ask for the whole node explicitly.)
#
# Serve a checkpoint, run all requested eval stages, compile the unified master CSV, then
# tear down the server — all in ONE sbatch job. The node frees when the job ends.
# eval_all.sh --serve owns the vLLM lifecycle (launch + health-wait + trap-kill on exit; it
# kills ONLY the PID it started, never pkill-by-pattern).
#
# SUBMIT VIA THE WRAPPER (preferred — puts .out/.err in a dated subdir):
#   /home/sgsilva/utilities/eval/sbatch_eval_all.sh
# Or manually: export vars first, then:
#   export MODEL=/mnt/data/sgsilva/models/qwen35-4b-mix-12k-1506-sft-step1299
#   export BASE_MODEL=qwen3.5-4b STAGES=aux,visualobs,benchmarks THINKING=off
#   export TRAIN_GROUP_ID=mix_12k_1506 RUN_ID=step1299_full_thinkoff
#   sbatch --export=ALL /home/sgsilva/utilities/eval/eval_all.sbatch
# (--export=ALL forwards your whole exported env, so STAGES' commas are safe.)
#
# Required env: MODEL, BASE_MODEL, STAGES, THINKING (on|off — must match the SFT target).
# aux stage also needs TRAIN_GROUP_ID + RUN_ID. Optional: TP, MAX_LEN, PORT, MAX_SAMPLES,
# JUDGE_BASE_URL, JUDGE_MODEL, SERVE_WAIT, TESTSET, BENCH_MAX_TOKENS, BENCH_EXTRA.
#
# ⚠ THIS SCRIPT IS ENV-DRIVEN, NOT FLAG-DRIVEN. Pass config as ENV VARS (STAGES=..., not --stages).
#   A missing required var fails in <10s at the :? guard below (no GPU wasted) — but DON'T guess the
#   command: the canonical, copy-paste form lives in the /eval-vlm skill. Use it.
#
# OPTIONAL vo_s2 chain (the two-stage board value, which eval_all.sh does NOT produce on its own):
#   set WITH_VO_S2=1 ONLY_STEM=<unique obs-stem substring, e.g. aux12k-union-reasoning-sft-step4125>
#   → after the stages, serves the sft2812 reasoner (thinkON, REASONER_PORT default 8150) and sweeps
#   ONLY that model's obs → vo_s2 → auto-recompiles master. Override REASONER / REASONER_PORT if needed.
#   GPU handling (fixed 2026-06-25): the reasoner TP auto-derives from THIS job's allocation
#   (CUDA_VISIBLE_DEVICES count → REASONER_TP, override with REASONER_TP=N); before serving it WAITS
#   up to 120s for the just-torn-down model server's GPU memory to release, then serves with
#   ALLOW_BUSY_GPUS=1 (safe — the only "busy" device is our own draining alloc, never a co-tenant).
#   So request the SAME --gres for the reasoner as the eval used (gpu:4 for 27B); no separate node.
#
# CANONICAL 5-axis thinkON 27B example (mirror this EXACTLY; only change MODEL/RUN_ID/group/ONLY_STEM):
#   cd /home/sgsilva/vlm-post-training && unset PORT
#   MODEL=/mnt/data/sgsilva/models/<ckpt> BASE_MODEL=qwen3.5-27b STAGES=aux,visualobs,benchmarks \
#   THINKING=on TRAIN_GROUP_ID=<group> RUN_ID=<run>_thinkon BENCH_MAX_TOKENS=32768 \
#   WITH_VO_S2=1 ONLY_STEM=<unique-substring> \
#   /home/sgsilva/utilities/eval/sbatch_eval_all.sh --gres=gpu:4 --job-name=eval-27b-<short>
# (thinkOFF: THINKING=off, drop BENCH_MAX_TOKENS; 4 axes only: drop WITH_VO_S2 + ONLY_STEM.)
set -uo pipefail

# ---- required ----
: "${MODEL:?set MODEL=/mnt/data/sgsilva/models/<ckpt>}"
: "${BASE_MODEL:?set BASE_MODEL=qwen3.5-4b|qwen3.5-9b|qwen3.5-27b}"
: "${STAGES:?set STAGES=aux,visualobs,benchmarks (any subset)}"
: "${THINKING:?set THINKING=on|off (must match the SFT target)}"
# ---- optional (defaults) ----
# PORT: derived from SLURM_JOB_ID so two co-located packed jobs CANNOT collide. The old GPU-bank
# heuristic was BROKEN for packing — SLURM remaps CUDA_VISIBLE_DEVICES to 0-based PER JOB (both
# packed gpu:4 jobs see "0,1,2,3" → both picked 8000 → the 2nd eval queried the 1st's server →
# WRONG-MODEL garbage scores; war story: 94198/94199 on worker-4, 4B-on hit the 27B server → video
# 2%). job-id parity gives each job a distinct port with no manual bookkeeping. Explicit PORT= still
# overrides (e.g. to attach a known port). Interactive (no SLURM_JOB_ID) falls back to 8000.
# LEAK GUARD: under SLURM, an INHERITED PORT (almost always a stale export leaked via --export=ALL,
# NOT a deliberate pin) silently overrides the job-id scheme → two packed jobs both reuse the leaked
# 8000 → wrong-model collision (the F2.2 preflight catches it, but the run is wasted). So under SLURM
# we IGNORE an inherited PORT and always use the job-id port, warning loudly. To deliberately pin a
# port (rare — e.g. attach a known server), set PORT_PIN=<n>, which wins.
if [[ -n "${PORT_PIN:-}" ]]; then
  PORT="$PORT_PIN"
  echo "  PORT: pinned via PORT_PIN=$PORT"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _jobport=$(( 8000 + (SLURM_JOB_ID % 100) ))   # 8000-8099, distinct per job → collision-free when packed
  if [[ -n "${PORT:-}" && "${PORT}" != "${_jobport}" ]]; then
    echo "  ⚠ WARN: ignoring INHERITED PORT=${PORT} (stale --export=ALL leak?) — using job-id port ${_jobport}." >&2
    echo "          (a deliberate pin must use PORT_PIN=<n>; an inherited PORT under SLURM is treated as a leak.)" >&2
  fi
  PORT="$_jobport"
else
  PORT="${PORT:-8000}"   # interactive (no SLURM_JOB_ID): honor an explicit PORT, else 8000
fi
SERVE_WAIT="${SERVE_WAIT:-1800}"
BASE_URL="http://localhost:${PORT}/v1"

echo "===== eval_all.sbatch ====="
echo "  node      : $(hostname)"
echo "  job id    : ${SLURM_JOB_ID:-<interactive>}"
echo "  model     : $MODEL"
echo "  base      : $BASE_MODEL   stages: $STAGES   thinking: $THINKING"
echo "  serve     : port $PORT  TP=${TP:-default}  max_len=${MAX_LEN:-default}  wait=${SERVE_WAIT}s"
echo "  group/run : ${TRAIN_GROUP_ID:-<none>} / ${RUN_ID:-<none>}"
# --export=ALL forwards the WHOLE env, so a stale exported value silently re-targets this run.
# Echo the leak-prone vars so a leftover (esp. SERVE_VENV/TP/TESTSET/BENCH_MAX_TOKENS) is visible
# in the log BEFORE the run starts. SERVE_VENV must be UNSET for bare Qwen (pmartins-only).
echo "  env-leak watch: SERVE_VENV=${SERVE_VENV:-<unset>}  TESTSET=${TESTSET:-<default 2906>}  BENCH_MAX_TOKENS=${BENCH_MAX_TOKENS:-<default>}  TAG=${TAG:-<none>}  MAX_SAMPLES=${MAX_SAMPLES:-<none>}"
# VO_* cohort vars are read by eval_all.sh DIRECTLY from env (no flag translation) — a stale
# exported VO_TEST/VO_COHORT_TAG silently re-cohorts the whole visualobs stage (job 111364,
# 2026-07-10: submitted without overrides → silent 1105 default under an '[1806]' label). Echo
# them ALWAYS; scream when the visualobs stage is requested with NO explicit cohort wiring.
echo "  env-leak watch (VO): VO_COHORT_TAG=${VO_COHORT_TAG:-<unset>}  VO_TEST=${VO_TEST:-<unset → 1105 DEFAULT>}  VO_GT_CAT=${VO_GT_CAT:-<unset>}  VO_SCHEMA=${VO_SCHEMA:-<unset>}  VO_PROCESSED_DIR=${VO_PROCESSED_DIR:-<unset>}  VO_SESSIONS_FROM=${VO_SESSIONS_FROM:-<unset>}  VO_OBS_FILE=${VO_OBS_FILE:-<unset>}"
if [[ ",${STAGES}," == *",visualobs,"* && -z "${VO_TEST:-}" ]]; then
  echo "  ⚠⚠ visualobs stage requested with VO_TEST UNSET → eval_all.sh will SILENTLY default to the" >&2
  echo "     SUPERSEDED 1105 bank (N=1181). If you meant the current cohort (e.g. 1806), STOP and" >&2
  echo "     resubmit with the full VO_* set (see the canonical submit in /eval-vlm)." >&2
  echo "     [This becomes a HARD FAIL when testsets.json + required --cohort land — plan step 5.]" >&2
fi
# BENCH_EXTRA leak guard: a stale `--skip-vsibench/--skip-mmmu/--skip-videomme` leaked via
# --export=ALL silently drops benchmarks from a fresh run (the mix12k/mixreas miss 2026-06-20:
# leaked --skip-vsibench --skip-mmmu → only Video-MME ran, MMMU+VSI blank with NO error). Make it
# LOUD + explicit: if the benchmarks stage is requested AND BENCH_EXTRA carries a skip, print a
# banner naming exactly which benchmarks WON'T run, so a leak can't masquerade as a clean run.
if [[ ",${STAGES}," == *",benchmarks,"* && "${BENCH_EXTRA:-}" == *--skip-* ]]; then
  echo "  ⚠⚠ BENCH_EXTRA=\"${BENCH_EXTRA}\" will SKIP benchmarks in this run:" >&2
  [[ "${BENCH_EXTRA}" == *--skip-vsibench* ]] && echo "       - VSI-Bench  WILL NOT RUN" >&2
  [[ "${BENCH_EXTRA}" == *--skip-mmmu*     ]] && echo "       - MMMU-val   WILL NOT RUN" >&2
  [[ "${BENCH_EXTRA}" == *--skip-videomme* ]] && echo "       - Video-MME  WILL NOT RUN" >&2
  echo "     If you did NOT intend this, it's a stale --export=ALL leak → 'unset BENCH_EXTRA' and resubmit." >&2
fi
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/    gpu /'

# ---- build the eval_all.sh command (only pass flags that are set) ----
CMD=( /home/sgsilva/utilities/eval/eval_all.sh
  --model "$MODEL" --base-model "$BASE_MODEL" --stages "$STAGES"
  --serve --thinking "$THINKING" --base-url "$BASE_URL" --serve-wait "$SERVE_WAIT" )
[[ -n "${TP:-}" ]]              && CMD+=( --tp "$TP" )
[[ -n "${MAX_LEN:-}" ]]         && CMD+=( --max-len "$MAX_LEN" )
[[ -n "${SERVE_VENV:-}" ]]      && CMD+=( --serve-venv "$SERVE_VENV" )  # pmartins -> vlm-post-training-home-venv
[[ "${KEEP_SERVER:-0}" == 1 ]]  && CMD+=( --keep-server )  # leave vLLM up after eval (reuse; rarely useful in sbatch since node frees at job end)
[[ "${FULL_REBUILD:-0}" == 1 ]] && CMD+=( --full-rebuild )  # end-of-run board rebuild = --full-scan (after an exporter/compiler code change)
[[ -n "${BENCH_MAX_TOKENS:-}" ]] && CMD+=( --bench-max-tokens "$BENCH_MAX_TOKENS" )  # cap thinkon runaways (auto-caps to 16384 if unset; see eval_all.sh) so Video-MME finishes in hours not days
[[ -n "${BENCH_EXTRA:-}" ]]      && CMD+=( --bench-extra "$BENCH_EXTRA" )  # e.g. "--skip-videomme" (skip the slow runaway-tail benchmark)
[[ -n "${TRAIN_GROUP_ID:-}" ]]  && CMD+=( --train-group-id "$TRAIN_GROUP_ID" )
[[ -n "${RUN_ID:-}" ]]          && CMD+=( --run-id "$RUN_ID" )
[[ -n "${TAG:-}" ]]             && CMD+=( --tag "$TAG" )
[[ -n "${TESTSET:-}" ]]         && CMD+=( --testset "$TESTSET" )
[[ -n "${MAX_SAMPLES:-}" ]]     && CMD+=( --max-samples "$MAX_SAMPLES" )
[[ -n "${JUDGE_BASE_URL:-}" ]]  && CMD+=( --judge-base-url "$JUDGE_BASE_URL" )
[[ -n "${JUDGE_MODEL:-}" ]]     && CMD+=( --judge-model "$JUDGE_MODEL" )

echo "  exec: ${CMD[*]}"
echo "==========================="
"${CMD[@]}"
RC=$?
echo "===== eval_all.sbatch done (rc=$RC) — model server torn down by eval_all --serve ====="

# ---- OPTIONAL vo_s2 chain (set WITH_VO_S2=1) — the two-stage board value. eval_all does NOT
#      produce vo_s2 (it needs a separate sft2812-reasoner sweep over the saved stage-1 obs). When
#      requested, serve the reasoner here (model server already down → GPUs free) and sweep ONLY this
#      model's obs; reasoner_sweep auto-recompiles the master. Reads obs from disk, so model server
#      is NOT needed. Off by default — nothing changes for existing runs. ----
if [[ "$RC" == 0 && "${WITH_VO_S2:-0}" == 1 ]]; then
  : "${ONLY_STEM:?WITH_VO_S2=1 needs ONLY_STEM=<unique obs-stem substring, e.g. aux12k-union-reasoning-sft-step4125>}"
  REASONER="${REASONER:-/mnt/data/pmartins/vlm_ckpts/sft_qwen_35_27b_1805_not_reviewed_diverse_reasoning_merged_rep_fe_comparison_3_mix_reas_12k_full_2005/step_2812/hf}"
  REASONER_PORT="${REASONER_PORT:-8150}"  # OUTSIDE the model's 8000-8099 job-id-port band (no collision)

  # ---- GPU/worker handling (rewritten 2026-06-25) ----------------------------------------
  # The reasoner runs in THIS job's SLURM allocation, so nvidia-smi only sees OUR --gres GPUs.
  # Three things bit us before (job 99326): (1) TP hardcoded 4 — must match the allocation;
  # (2) the model server we just tore down hadn't released its GPU memory yet, so the serve
  # preflight saw "3/4 free" and refused; (3) no busy-tolerant fallback even though the GPUs
  # are OURS. Fix: derive TP from the allocation, WAIT for our own teardown to settle, then
  # allow-busy (the only "busy" device is our draining server, never another user).
  # Derive TP from the GPUs SLURM gave us (CUDA_VISIBLE_DEVICES count), default 4.
  REASONER_TP="${REASONER_TP:-}"
  if [[ -z "$REASONER_TP" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      REASONER_TP=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
    elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
      REASONER_TP="$SLURM_GPUS_ON_NODE"
    else
      REASONER_TP=4
    fi
  fi
  # Wait for OUR just-torn-down model server's GPU memory to actually free (teardown is async;
  # vLLM can hold 200GB+ for tens of seconds after SIGTERM). Poll nvidia-smi until the max
  # used-MiB across our visible GPUs drops below 5 GB, or 120s elapse — THEN serve.
  echo "===== WITH_VO_S2: waiting for prior server's GPU memory to release (TP=$REASONER_TP) ====="
  for s in $(seq 1 24); do
    maxused=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    maxused="${maxused:-0}"
    if [[ "$maxused" -lt 5000 ]]; then echo "  GPUs settled (max ${maxused} MiB used) after $((s*5))s"; break; fi
    [[ "$s" == 24 ]] && echo "  ⚠ GPUs still show ${maxused} MiB after 120s — proceeding with ALLOW_BUSY_GPUS (our own draining alloc)"
    sleep 5
  done

  echo "===== WITH_VO_S2: serving reasoner $REASONER on :$REASONER_PORT (TP=$REASONER_TP, thinkON) ====="
  # ALLOW_BUSY_GPUS=1 is SAFE here: this is our dedicated SLURM allocation — any residual "busy"
  # reading is our own draining server, never a co-tenant. (We waited above; this is the backstop.)
  ALLOW_BUSY_GPUS=1 ENABLE_THINKING=1 QWEN35_VENV=/home/sgsilva/vlm-post-training-home-venv \
    /home/sgsilva/utilities/serve/start_vllm_server.sh "$REASONER" "$REASONER_TP" 65536 "$REASONER_PORT" &
  RPID=$!
  trap 'kill "$RPID" 2>/dev/null' EXIT INT TERM
  for i in $(seq 1 150); do
    curl -sf "http://localhost:$REASONER_PORT/v1/models" >/dev/null 2>&1 && { echo "  reasoner up (${i}x10s)"; break; }
    # if the server proc died (e.g. OOM at this TP), stop waiting the full 25 min
    kill -0 "$RPID" 2>/dev/null || { echo "  ⚠ reasoner process exited early (PID $RPID gone)"; break; }
    sleep 10
  done
  if curl -sf "http://localhost:$REASONER_PORT/v1/models" >/dev/null 2>&1; then
    echo "===== WITH_VO_S2: reasoner_sweep ONLY=$ONLY_STEM (vo_s2 → auto-recompile master) ====="
    REASONER="$REASONER" REASONER_URL="http://localhost:$REASONER_PORT/v1" ONLY="$ONLY_STEM" \
      /home/sgsilva/utilities/eval/reasoner_sweep.sh
    RC=$?
    echo "===== WITH_VO_S2: reasoner_sweep rc=$RC ====="
  else
    echo "  ⚠ reasoner never came up — vo_s2 SKIPPED (aux/bench/vo_s1/agree already done)." >&2
    RC=3
  fi
fi

echo "===== eval_all.sbatch FINISHED (rc=$RC) — node releasing ====="
exit $RC
