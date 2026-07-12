#!/bin/bash
#SBATCH --job-name=expb-gateb-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --exclude=worker-30,worker-31
#SBATCH --output=/mnt/data/sgsilva/logs/eval/slurm-expb-gateb-%j.out
#SBATCH --error=/mnt/data/sgsilva/logs/eval/slurm-expb-gateb-%j.err

# EXP-B GATE-B GT-obs eval, one checkpoint per job (campaign one-off, 202607).
# Serves the EXP-B stage-2 reasoner thinkON (its SFT targets carry real <think>) on 2 GPUs,
# waits for health, runs the two-stage severity eval on the 1806 test split with the 2906
# categorical_k5majority GT observations as stage-1 (--precomputed-visual-obs skips the
# stage-1 model call), then tears the server down so the node frees.
#   CKPT=<exported HF dir> sbatch <this>
# Prompt identity: --stage2-stance ondemand renders the trained user-turn byte-identically
# (verified 2,157/2,157; include-options is auto-forced OFF by evaluate.py).
# GATE B target: beat fixed-sft2812 2-call vo_s2 = 55.1 on GT-obs.

# MODE=gtobs (default): single-stage on the test BUILD (GT obs inlined in the stored prompt).
# MODE=modelobs: two-stage stance eval on repetitions_test with MODEL stage-1 obs:
#   OBS_JSON=<obs_*.json from the stage-1 model> OBS_TAG=<short stem for the output filename>
#   Descriptions come byte-exact from the build via --stage2-desc-build (commit 4595fd0);
#   the ~103 reps without obs coverage hard-error per sample (counted, never silent).
# MODE=selfloop: two-stage on repetitions_test, NO --precomputed-visual-obs — the SAME EXP-B
#   checkpoint answers its own stage-1 categorical Q/A prompt (build_stage1_prompt, live model
#   call, one extra query per rep) then consumes those self-generated obs in its own stage-2
#   stance prompt. This checkpoint was never trained on the stage-1 task; quality is unmeasured.
#   Distinct row from gtobs/modelobs (2 rows, 2 strategies) via OBS_TAG=selfloop.
# MODE=singlestage (2026-07-06): the SINGLE-STAGE VO family every other board model gets — plain
#   eval/evaluate.py, NO --two-stage, NO obs of any kind, model emits severity directly from the
#   video using the dataset's own stored prompt (byte-exact, same template as every other
#   single-stage board row; verified 07-06 against the 1105 test set's prompt — only exercise-
#   specific content differs per cohort, template identical). This is an intentional OFF-TRAINING
#   probe: EXP-B was SFT'd on a stance/two-stage prompt expecting an obs block, so this measures
#   whether it retains any severity-judgment ability without its trained scaffolding at all.
#
# NON-EXP-B checkpoints (2026-07-09): MODE=gtobs is fully generic — it just serves $CKPT and
# scores it on the test BUILD's byte-exact stored prompt (exercise desc + GT obs inlined), no
# EXP-B-specific logic in that path. To point this at e.g. a 397B apples-to-apples comparison,
# override the serving shape (27B TP2/131072 defaults, unchanged for all existing callers) —
# 397B canonical per /serve-vllm skill = TP8/262144 (full node; ~99GB/GPU weights at TP8,
# ~169GB/GPU headroom for KV cache at B300's 275GB/GPU — comfortable, not memory-constrained):
#   MODEL_TAG=<short id, e.g. 397b> SERVE_GPUS=8 SERVE_MAXLEN=262144 CKPT=<path> \
#     sbatch --gres=gpu:8 --mem=2400G --cpus-per-task=192 <this>
# MODEL_TAG also fixes the output filename: the old $STEP extraction (grep -oE 'step[0-9]+')
# returns EMPTY for a checkpoint dirname with no stepNNN (e.g. Qwen3.5-397B-A17B), which would
# have silently written to a malformed/collapsed filename — MODEL_TAG is the explicit fix.
set -uo pipefail
CKPT="${CKPT:?set CKPT=/mnt/data/sgsilva/models/qwen35-27b-expb-stage2-ondemand-sft-stepNNN}"
MODE="${MODE:-gtobs}"
SERVE_GPUS="${SERVE_GPUS:-2}"       # was hardcoded 2 (27B TP2) — override for other model sizes
SERVE_MAXLEN="${SERVE_MAXLEN:-131072}"  # was hardcoded 131072 — override for other model sizes
# STEP_TAG lets two differently-named checkpoints that share the same stepNNN suffix (e.g. the
# original inverted run vs a _flipfix retrain) write to distinct output filenames instead of
# silently colliding on the same stepNNN-derived path (caught 2026-07-09 before the flipfix
# step840 eval would have overwritten the pre-fix inverted result).
# MODEL_TAG (2026-07-09): explicit override for the filename stem — REQUIRED for any checkpoint
# whose basename has no stepNNN suffix (the auto-extraction below returns empty for those).
_STEP_AUTO="$(basename "$CKPT" | grep -oE 'step[0-9]+')"
STEP="${MODEL_TAG:-${_STEP_AUTO:?no stepNNN in checkpoint name and no MODEL_TAG set — pass MODEL_TAG=<short id>}}${STEP_TAG:+_${STEP_TAG}}"
PORT=$(( 8300 + (SLURM_JOB_ID % 100) ))
PY=/home/sgsilva/vlm-post-training-home-venv/bin/python
VPT=/home/sgsilva/vlm-post-training
# GT-obs arm v2 (2026-07-06): run SINGLE-STAGE on the EXP-B test BUILD, whose messages[1] IS the
# byte-exact trained prompt (ondemand template + 2906 GT obs inlined by build_stage2_from_raw).
# The first attempt used --two-stage --stage2-stance + --precomputed-visual-obs on repetitions_test:
# obs block + template matched, but the exercise DESCRIPTION was extracted from the original
# repetitions_test prompt, whose layout differs from the builder's build_ref_desc (~1.7k chars,
# 'Correct Movement Criteria:' placement) → prompts were NOT train-identical (outputs kept as
# *_gtobs_DESCDRIFT_thinkon.json). Single-stage on the build uses the stored prompt verbatim —
# no reconstruction, no drift. (The two-stage stance path is still needed for the MODEL-obs arm;
# its description source must be fixed to build_ref_desc before that arm runs.)
# BUILD/TESTDIR/EXPECTED_N env-overridable since 2026-07-12 (1806 re-baseline: min1s build=2142,
# min1s repetitions_test=2245). Defaults unchanged for the legacy (pre-min1s) callers.
_EXPECTED_N_ENV="${EXPECTED_N:-}"
BUILD="${BUILD:-/mnt/data/sgsilva/datasets/1806/expb_stage2_from_raw_ondemand_test_flagkeep}"
EXPECTED_N="${EXPECTED_N:-2157}"
if [ "$MODE" = "modelobs" ]; then
    OBS_JSON="${OBS_JSON:?MODE=modelobs needs OBS_JSON=<stage-1 obs json>}"
    OBS_TAG="${OBS_TAG:?MODE=modelobs needs OBS_TAG=<short output stem>}"
    OUT=/mnt/data/sgsilva/results/visual_obs/runs/stage2_expb_stage2_ondemand_${STEP}_1806_modelobs_${OBS_TAG}_thinkon.json
    TESTDIR=/mnt/data/shared/vlm/data/human_annotation_datasets/1806_after_format_review_diverse_reasoning/repetitions_test
elif [ "$MODE" = "selfloop" ]; then
    OUT=/mnt/data/sgsilva/results/visual_obs/runs/stage2_expb_stage2_ondemand_${STEP}_1806_selfloop_thinkon.json
    TESTDIR="${TESTDIR:-/mnt/data/shared/vlm/data/human_annotation_datasets/1806_after_format_review_diverse_reasoning/repetitions_test}"
    # 2260 total reps in raw repetitions_test (min1s re-baseline: pass TESTDIR=…_min1s + EXPECTED_N=2245);
    # selfloop has no GT-obs-coverage gate, so its N is the cohort N, not the build N.
    EXPECTED_N="${_EXPECTED_N_ENV:-2260}"
elif [ "$MODE" = "singlestage" ]; then
    # Filename carries "singlestage" (not "stage2_") so the compiler's vo_s1 ingestion (not vo_s2)
    # picks it up, matching every other board model's single-stage family.
    OUT=/mnt/data/sgsilva/results/visual_obs/runs/expb_stage2_ondemand_${STEP}_1806_singlestage_thinkon.json
    TESTDIR=/mnt/data/shared/vlm/data/human_annotation_datasets/1806_after_format_review_diverse_reasoning/repetitions_test
    EXPECTED_N=2260
else
    # board files carry the _1806 cohort tag (the first run's files were renamed on disk).
    # Filename prefix stays "expb_stage2_ondemand" for the default (no-MODEL_TAG) EXP-B callers —
    # unchanged, so it keeps matching the master_models.json vo_tokens already registered for
    # those runs. A non-EXP-B comparison model (MODEL_TAG set) gets an honest "compare_" prefix
    # instead — it is NOT an EXP-B checkpoint and must not look like one on disk.
    if [ -n "${MODEL_TAG:-}" ]; then
        OUT=/mnt/data/sgsilva/results/visual_obs/runs/stage2_compare_${STEP}_1806_gtobsbuild_thinkon.json
    else
        OUT=/mnt/data/sgsilva/results/visual_obs/runs/stage2_expb_stage2_ondemand_${STEP}_1806_gtobsbuild_thinkon.json
    fi
    TESTDIR=$BUILD
fi

echo "=== EXP-B GATE-B eval: $CKPT (thinkON, port $PORT, node $(hostname -s)) ==="

# 1. Serve thinkON (real <think> targets -> ENABLE_THINKING=1, no --disable-thinking)
#    SERVE_GPUS/SERVE_MAXLEN default to 2/131072 (27B) — override for other model sizes; the
#    #SBATCH --gres=gpu:2 header must be overridden to match at submit time:
#    sbatch --gres=gpu:8 <this>  (SLURM cli flags win over an in-file #SBATCH directive).
ENABLE_THINKING=1 /home/sgsilva/utilities/serve/start_vllm_server.sh "$CKPT" "$SERVE_GPUS" "$SERVE_MAXLEN" "$PORT" &
SERVE_PID=$!
cleanup() { echo "=== teardown: killing vLLM (pid $SERVE_PID) ==="; kill "$SERVE_PID" 2>/dev/null; sleep 5; pkill -P "$SERVE_PID" 2>/dev/null; }
trap cleanup EXIT

# 2. Health wait (up to 30 min for 27B load)
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "server healthy after ~$((i*10))s"; break; fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "ERROR: vLLM exited during startup"; exit 1; fi
    [ "$i" -eq 180 ] && { echo "ERROR: server not healthy after 30 min"; exit 1; }
    sleep 10
done

# 3. Eval (two passes: --resume tops off stragglers)
cd "$VPT"
EXTRA_ARGS=()
if [ "$MODE" = "modelobs" ]; then
    EXTRA_ARGS=(--two-stage --stage2-stance ondemand --stage2-desc-build "$BUILD" --precomputed-visual-obs "$OBS_JSON")
elif [ "$MODE" = "selfloop" ]; then
    # NO --precomputed-visual-obs: evaluate.py falls through to a LIVE stage-1 call
    # (build_stage1_prompt, 2906 categorical) served by this SAME checkpoint.
    EXTRA_ARGS=(--two-stage --stage2-stance ondemand --stage2-desc-build "$BUILD" --visual-obs-variant categorical)
fi
for pass in 1 2; do
    echo "=== eval pass $pass (mode=$MODE) ==="
    "$PY" eval/evaluate.py \
      --test-dataset-dir "$TESTDIR" \
      "${EXTRA_ARGS[@]}" \
      --model "$CKPT" \
      --server-url "http://127.0.0.1:${PORT}/v1" \
      --max-tokens 32768 \
      --max-workers 16 \
      --output-file "$OUT" \
      --resume
done

# 4. Completeness check (never trust the summary line alone)
N=$("$PY" -c "import json;print(json.load(open('$OUT')).get('metadata',{}).get('evaluated_samples',0))")
echo "=== evaluated_samples: $N / $EXPECTED_N ==="
[ "$N" -ge $(( EXPECTED_N * 99 / 100 )) ] || { echo "ERROR: incomplete eval ($N < 99% of $EXPECTED_N)"; exit 1; }
echo "=== DONE: $OUT ==="
